import logging
from pathlib import Path
import time
import numpy as np
from .config import dataset_root
from .geometry import pose_matrix
from .io import digest, file_digest, read_json, write_json
from .schema import Cuboid

LOG = logging.getLogger(__name__)


def read_lidar(path):
    """nuScenes stores little-endian float32: x,y,z,intensity,ring (5 columns)."""
    path = Path(path)
    if path.stat().st_size == 0 or path.stat().st_size % 20:
        raise ValueError(f"Not a nonempty 5-float nuScenes LiDAR file: {path}")
    points = np.fromfile(path, dtype="<f4").reshape(-1, 5)
    if not np.isfinite(points).all():
        raise ValueError(f"Nonfinite LiDAR values: {path}")
    if np.any(points[:, 4] < 0) or np.any(points[:, 4] > 31) or not np.allclose(points[:, 4], np.round(points[:, 4])):
        raise ValueError(f"Invalid nuScenes 32-beam ring column: {path}")
    return points


def write_pcd(path, points):
    points = np.asarray(points, dtype="<f4")
    if points.ndim != 2 or points.shape[1] != 5 or not np.isfinite(points).all():
        raise ValueError("Expected finite N x 5 nuScenes points")
    # XYZ only: CVAT/THREE uses the raw coordinates without a viewer transform.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    header = ("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\n"
              "TYPE F F F\nCOUNT 1 1 1\n" f"WIDTH {n}\nHEIGHT 1\n"
              "VIEWPOINT 0 0 0 1 0 0 0\n" f"POINTS {n}\nDATA binary\n")
    tmp = path.with_suffix(".pcd.tmp")
    with tmp.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(points[:, :3].astype("<f4").tobytes())
    tmp.replace(path)


def open_nuscenes(cfg):
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version=cfg["dataset"]["version"], dataroot=str(dataset_root(cfg)), verbose=False)


def global_from_sensor(nusc, sample_data):
    calibration = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", sample_data["ego_pose_token"])
    return pose_matrix(ego["translation"], ego["rotation"]) @ pose_matrix(calibration["translation"], calibration["rotation"])


def selected_samples(nusc, cfg, limit=None):
    from nuscenes.utils.splits import create_splits_scenes
    splits = create_splits_scenes()
    split = cfg["dataset"]["split"]
    if split not in splits:
        raise ValueError(f"Unknown nuScenes split: {split}")
    version = cfg["dataset"]["version"]
    if (version == "v1.0-mini") != split.startswith("mini_"):
        raise ValueError("Dataset version and split do not agree")
    requested = set(cfg["dataset"].get("scenes", []))
    allowed = set(splits[split])
    if requested - allowed:
        raise ValueError(f"Scenes outside split: {sorted(requested - allowed)}")
    scenes = sorted((s for s in nusc.scene if s["name"] in allowed and (not requested or s["name"] in requested)), key=lambda s: s["name"])
    count = limit if limit is not None else cfg["dataset"]["max_frames"]
    if count <= 0:
        raise ValueError("Frame limit must be positive")
    result = []
    for scene in scenes:
        token = scene["first_sample_token"]
        while token and len(result) < count:
            sample = nusc.get("sample", token)
            result.append((scene, sample))
            token = sample["next"]
        if len(result) >= count:
            break
    if not result:
        raise ValueError("No samples selected")
    return result


def prepare(cfg, limit=None, pcd_dir=None):
    start = time.perf_counter()
    root, run = dataset_root(cfg), Path(cfg["run_dir"])
    nusc = open_nuscenes(cfg)
    uploaded = None
    if pcd_dir is None:
        selection = selected_samples(nusc, cfg, limit)
    else:
        if limit is not None:
            raise ValueError('Uploaded PCD preparation uses all files in --pcd-dir')
        # Validate version, split and requested scenes using the normal selection rules.
        selected_samples(nusc, cfg, 1)
        from nuscenes.utils.splits import create_splits_scenes
        from .uploaded import match_uploaded_pcds
        allowed = set(create_splits_scenes()[cfg['dataset']['split']])
        if cfg['dataset'].get('scenes'):
            allowed &= set(cfg['dataset']['scenes'])
        uploaded = match_uploaded_pcds(nusc, root, allowed, pcd_dir)
        selection = [(scene, sample) for scene, sample, _, _ in uploaded]
    return prepare_selection(cfg, nusc, selection, uploaded, start)


def prepare_selection(cfg, nusc, selection, uploaded=None, start=None):
    """Prepare verified selections, including PCDs resolved by the UI function."""
    start = time.perf_counter() if start is None else start
    root, run = dataset_root(cfg), Path(cfg['run_dir'])
    selection_spec = {"version": cfg["dataset"]["version"], "split": cfg["dataset"]["split"],
                      "tokens": [s["token"] for _, s in selection], "sweeps": cfg["detector"]["sweeps"]}
    if uploaded is not None:
        selection_spec['uploaded_pcds'] = [{'name': path.name, 'sha256': sha}
                                         for _, _, path, sha in uploaded]
    manifest_path = run / "manifest.json"
    if manifest_path.exists():
        old = load_manifest(run)
        if old["selection"] != selection_spec:
            raise ValueError("Selection changed. Use a new run_dir so CVAT mapping stays immutable.")
        validate_manifest(old, run, root, check_sweeps=True)
        LOG.info("Preparation already complete (%d frames)", len(old["frames"]))
        return old
    frames = []
    for index, (scene, sample) in enumerate(selection):
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        global_from_lidar = global_from_sensor(nusc, sd)
        source = root / sd["filename"]
        points = read_lidar(source)
        if uploaded is None:
            pcd = f"pcd/{index:06d}_{sample['token']}.pcd"
            write_pcd(run / pcd, points)
        else:
            import shutil
            path, sha = uploaded[index][2:]
            if file_digest(path) != sha:
                raise ValueError('Uploaded PCD changed during preparation')
            pcd = f'pcd/{path.name}'
            destination = run / pcd
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and file_digest(destination) != sha:
                raise ValueError('Run already contains a different PCD; use a new run_dir')
            if destination.resolve() != path.resolve():
                shutil.copyfile(path, destination)
            if file_digest(destination) != sha:
                raise ValueError('Copied PCD differs from the matched upload')
        sweeps = []
        previous = sd["prev"]
        while previous and len(sweeps) < cfg["detector"]["sweeps"]:
            sweep = nusc.get("sample_data", previous)
            source_path = root / sweep["filename"]
            read_lidar(source_path)
            target_from_source = np.linalg.inv(global_from_lidar) @ global_from_sensor(nusc, sweep)
            sweeps.append({"sample_data_token": sweep["token"], "lidar_file": sweep["filename"],
                           "timestamp_us": sweep["timestamp"], "target_from_source": target_from_source.tolist(),
                           "sha256": file_digest(source_path)})
            previous = sweep["prev"]
        frames.append({"frame_id": index, "sample_token": sample["token"], "sample_data_token": sd["token"],
                       "scene": scene["name"], "timestamp_us": sd["timestamp"], "lidar_file": sd["filename"],
                       "lidar_sha256": file_digest(source), "pcd_file": pcd, "pcd_sha256": file_digest(run / pcd),
                       "point_count": len(points), "global_from_lidar": global_from_lidar.tolist(), "sweeps": sweeps})
        LOG.info("Prepared %d/%d: %s, %d points, %d prior sweeps", index + 1, len(selection), sample["token"], len(points), len(sweeps))
    manifest = {"schema_version": 1, "coordinate_frame": "lidar_sensor", "selection": selection_spec, "frames": frames}
    manifest["signature"] = digest(manifest)
    write_json(manifest_path, manifest)
    write_json(run / "prepare_timing.json", {"seconds": time.perf_counter() - start, "frames": len(frames), "source": "real_nuscenes"})
    return manifest


def load_manifest(run):
    manifest = read_json(Path(run) / "manifest.json")
    unsigned = {k: v for k, v in manifest.items() if k != "signature"}
    if digest(unsigned) != manifest["signature"]:
        raise ValueError("Manifest integrity check failed")
    frames = manifest["frames"]
    if not frames or [f["frame_id"] for f in frames] != list(range(len(frames))):
        raise ValueError("Manifest frame IDs must be contiguous")
    if len({f["sample_token"] for f in frames}) != len(frames):
        raise ValueError("Duplicate samples in manifest")
    return manifest


def validate_manifest(manifest, run, root=None, check_sweeps=False):
    for frame in manifest["frames"]:
        if file_digest(Path(run) / frame["pcd_file"]) != frame["pcd_sha256"]:
            raise ValueError("PCD changed since preparation")
        if root is not None:
            if file_digest(Path(root) / frame["lidar_file"]) != frame["lidar_sha256"]:
                raise ValueError("LiDAR input changed since preparation")
            if check_sweeps:
                for sweep in frame["sweeps"]:
                    if file_digest(Path(root) / sweep["lidar_file"]) != sweep["sha256"]:
                        raise ValueError("Sweep input changed since preparation")


def ground_truth(cfg, manifest):
    """Called only by evaluate. GT never supplies inference or pre-label shapes."""
    from nuscenes.eval.detection.utils import category_to_detection_name
    nusc = open_nuscenes(cfg)
    result = []
    roi = np.asarray(cfg["evaluation"]["point_cloud_range"])
    for frame in manifest["frames"]:
        _, boxes, _ = nusc.get_sample_data(frame["sample_data_token"], use_flat_vehicle_coordinates=False)
        for box in boxes:
            name = category_to_detection_name(box.name)
            if name not in cfg["detector"]["classes"]:
                continue
            annotation = nusc.get("sample_annotation", box.token)
            if annotation["num_lidar_pts"] < cfg["evaluation"]["min_lidar_points"]:
                continue
            if np.any(box.center < roi[:3]) or np.any(box.center > roi[3:]):
                continue
            result.append(Cuboid(sample_token=frame["sample_token"], frame_id=frame["frame_id"],
                                 prediction_id=f"gt:{box.token}", class_name=name, confidence=None,
                                 center_xyz=box.center.tolist(), size_lwh=box.wlh[[1, 0, 2]].tolist(),
                                 quaternion_wxyz=box.orientation.elements.tolist(), model="nuScenes-ground-truth"))
    return result
