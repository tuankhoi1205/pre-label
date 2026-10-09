"""Resolve manually uploaded PCDs to exact nuScenes keyframes, without GT."""
import hashlib
import io
from pathlib import Path

import numpy as np

from .dataset import read_lidar


def read_uploaded_pcd(path):
    path = Path(path)
    return parse_uploaded_pcd(path.read_bytes(), path.name)


def parse_uploaded_pcd(content, name='uploaded PCD'):
    if not 0 < len(content) <= 32 * 1024 * 1024:
        raise ValueError(f"PCD must be nonempty and at most 32 MiB: {name}")
    offset, header = 0, {}
    for line in io.BytesIO(content):
        offset += len(line)
        if offset > 4096:
            raise ValueError(f"Invalid PCD header: {name}")
        try:
            words = line.decode('ascii').strip().split()
        except UnicodeDecodeError as exc:
            raise ValueError(f"Invalid PCD header: {name}") from exc
        if not words or words[0].startswith('#'):
            continue
        header[words[0]] = words[1:]
        if words[0] == 'DATA':
            break
    fields = header.get('FIELDS')
    if fields not in (['x', 'y', 'z'], ['x', 'y', 'z', 'intensity']):
        raise ValueError(f"Use binary float32 XYZ or XYZI PCD: {name}")
    columns = len(fields)
    if (header.get('DATA') != ['binary'] or header.get('SIZE') != ['4'] * columns
            or header.get('TYPE') != ['F'] * columns or header.get('COUNT') != ['1'] * columns):
        raise ValueError(f"Use uncompressed binary float32 PCD: {name}")
    try:
        count = int(header['POINTS'][0])
        width, height = int(header['WIDTH'][0]), int(header['HEIGHT'][0])
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(f"Invalid PCD dimensions: {name}") from exc
    body = content[offset:]
    if count <= 0 or width * height != count or len(body) != count * columns * 4:
        raise ValueError(f"PCD point count/body length mismatch: {name}")
    points = np.frombuffer(body, dtype='<f4').reshape(count, columns)
    if not np.isfinite(points).all():
        raise ValueError(f"Nonfinite PCD points: {name}")
    return points, hashlib.sha256(content).hexdigest()


def match_uploaded_pcds(nusc, root, allowed_scenes, pcd_dir):
    paths = sorted(Path(pcd_dir).glob('*.pcd'))
    if not paths:
        raise ValueError('No .pcd files in --pcd-dir')
    targets, source_hashes = {}, {}
    for path in paths:
        points, source_hashes[path] = read_uploaded_pcd(path)
        key = (points.shape[1], hashlib.sha256(points.tobytes()).hexdigest())
        if key in targets:
            raise ValueError('Duplicate uploaded point clouds; select unique keyframes')
        targets[key] = path
    matches = {}
    scenes = {s['token']: s for s in nusc.scene if s['name'] in allowed_scenes}
    for sample in nusc.sample:
        if sample['scene_token'] not in scenes:
            continue
        sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        if not sd['is_key_frame']:
            continue
        points = read_lidar(Path(root) / sd['filename'])
        for columns in {key[0] for key in targets}:
            key = (columns, hashlib.sha256(points[:, :columns].tobytes()).hexdigest())
            if key in targets:
                path = targets[key]
                if path in matches:
                    raise ValueError(f'Ambiguous nuScenes keyframe match: {path.name}')
                matches[path] = (scenes[sample['scene_token']], sample, path, source_hashes[path])
    missing = [path.name for path in paths if path not in matches]
    if missing:
        raise ValueError('PCDs do not exactly match nuScenes keyframes in the configured split/scenes: '
                         + ', '.join(missing[:10]))
    return [matches[path] for path in paths]


class NuScenesPCDResolver:
    """Build one lazy content index; resolve uploads independent of file names."""
    def __init__(self, cfg):
        from .config import dataset_root
        from .dataset import open_nuscenes
        from nuscenes.utils.splits import create_splits_scenes

        self.root = dataset_root(cfg)
        self.nusc = open_nuscenes(cfg)
        self.scenes = {scene['token']: scene for scene in self.nusc.scene}
        splits = create_splits_scenes()
        names = ('mini_train', 'mini_val') if cfg['dataset']['version'] == 'v1.0-mini' else (
            ('test',) if cfg['dataset']['version'].endswith('-test') else ('train', 'val'))
        self.split_by_scene = {scene: split for split in names for scene in splits[split]}
        self.index = {}
        for sample in self.nusc.sample:
            sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            if not sd['is_key_frame']:
                continue
            # A mounted partial dataset may not contain every keyframe.
            path = self.root / sd['filename']
            if not path.is_file():
                continue
            points = read_lidar(path)
            for columns in (3, 4):
                key = (columns, hashlib.sha256(points[:, :columns].tobytes()).hexdigest())
                self.index.setdefault(key, []).append(sample)

    def resolve(self, content):
        points, sha = parse_uploaded_pcd(content)
        columns = points.shape[1]
        key = (columns, hashlib.sha256(points.tobytes()).hexdigest())
        samples = self.index.get(key, [])
        if len(samples) != 1:
            reason = 'Ambiguous PCD' if samples else 'PCD does not exactly match a mounted nuScenes LiDAR keyframe'
            raise ValueError(reason + '; keep original XYZ/intensity and mount its raw dataset, metadata and sweeps.')
        sample = samples[0]
        scene = self.scenes[sample['scene_token']]
        sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        current = read_lidar(self.root / sd['filename'])
        if current[:, :columns].tobytes() != points.tobytes():
            raise ValueError('nuScenes LiDAR changed after the upload index was built')
        if scene['name'] not in self.split_by_scene:
            raise ValueError('Matched scene is not in a supported nuScenes split')
        return scene, sample, self.split_by_scene[scene['name']], sha
