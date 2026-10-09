from types import SimpleNamespace

import numpy as np
import pytest

from prelabel.dataset import write_pcd
from prelabel.uploaded import match_uploaded_pcds, read_uploaded_pcd


def write_xyzi(path, points):
    points = np.asarray(points, dtype='<f4')
    path.write_bytes((f'# PCD\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n'
                      f'TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {len(points)}\nHEIGHT 1\n'
                      f'POINTS {len(points)}\nDATA binary\n').encode() + points.tobytes())


@pytest.fixture
def uploaded_fixture(tmp_path):
    root, inputs = tmp_path / 'nusc', tmp_path / 'uploads'
    root.mkdir()
    inputs.mkdir()
    points = np.array([[1, 2, 3, 99, 4], [4, 5, 6, 100, 8]], dtype='<f4')
    points.tofile(root / 'raw.bin')
    sd = {'is_key_frame': True, 'filename': 'raw.bin'}
    nusc = SimpleNamespace(scene=[{'token': 'scene-token', 'name': 'scene-1094'}],
                           sample=[{'token': 'sample', 'scene_token': 'scene-token',
                                    'data': {'LIDAR_TOP': 'lidar'}}],
                           get=lambda table, token: sd)
    return root, inputs, points, nusc, sd


@pytest.mark.parametrize('columns', [3, 4])
def test_exact_keyframe_match_keeps_uploaded_bytes_and_ignores_filename(uploaded_fixture, columns):
    root, inputs, points, nusc, _ = uploaded_fixture
    path = inputs / 'unrelated-name.pcd'
    if columns == 3:
        write_pcd(path, points)
    else:
        write_xyzi(path, points[:, :4])
    before = path.read_bytes()
    result = match_uploaded_pcds(nusc, root, {'scene-1094'}, inputs)
    assert result[0][1]['token'] == 'sample'
    assert result[0][2] == path and path.read_bytes() == before
    np.testing.assert_array_equal(read_uploaded_pcd(path)[0], points[:, :columns])


def test_changed_intensity_is_rejected_even_with_same_xyz(uploaded_fixture):
    root, inputs, points, nusc, _ = uploaded_fixture
    changed = points[:, :4].copy()
    changed[0, 3] += 1
    write_xyzi(inputs / '20.pcd', changed)
    with pytest.raises(ValueError, match='do not exactly match'):
        match_uploaded_pcds(nusc, root, {'scene-1094'}, inputs)


def test_match_rejects_other_split_and_non_keyframe(uploaded_fixture):
    root, inputs, points, nusc, sd = uploaded_fixture
    write_xyzi(inputs / '20.pcd', points[:, :4])
    with pytest.raises(ValueError, match='do not exactly match'):
        match_uploaded_pcds(nusc, root, {'another-scene'}, inputs)
    sd['is_key_frame'] = False
    with pytest.raises(ValueError, match='do not exactly match'):
        match_uploaded_pcds(nusc, root, {'scene-1094'}, inputs)


def test_duplicate_and_ambiguous_clouds_are_rejected(uploaded_fixture):
    root, inputs, points, nusc, _ = uploaded_fixture
    write_xyzi(inputs / '20.pcd', points[:, :4])
    write_xyzi(inputs / '21.pcd', points[:, :4])
    with pytest.raises(ValueError, match='Duplicate'):
        match_uploaded_pcds(nusc, root, {'scene-1094'}, inputs)
    (inputs / '21.pcd').unlink()
    nusc.sample.append({**nusc.sample[0], 'token': 'different-sample'})
    with pytest.raises(ValueError, match='Ambiguous'):
        match_uploaded_pcds(nusc, root, {'scene-1094'}, inputs)


@pytest.mark.parametrize('change', ['truncated', 'nonfinite', 'compressed'])
def test_invalid_pcd_is_rejected_before_matching(uploaded_fixture, change):
    _, inputs, points, _, _ = uploaded_fixture
    path = inputs / '20.pcd'
    if change == 'nonfinite':
        points[0, 0] = np.nan
    write_xyzi(path, points[:, :4])
    if change == 'truncated':
        path.write_bytes(path.read_bytes()[:-1])
    elif change == 'compressed':
        path.write_bytes(path.read_bytes().replace(b'DATA binary\n', b'DATA binary_compressed\n'))
    with pytest.raises(ValueError):
        read_uploaded_pcd(path)


def test_prepared_upload_preserves_pcd_and_calibrated_sweeps(uploaded_fixture, monkeypatch, tmp_path):
    import sys
    from prelabel import dataset
    from prelabel.io import file_digest

    root, inputs, points, nusc, sd = uploaded_fixture
    path = inputs / '20.pcd'
    write_xyzi(path, points[:, :4])
    points.tofile(root / 'prev.bin')
    sd.update(token='lidar', timestamp=2000000, prev='prev',
              calibrated_sensor_token='cal', ego_pose_token='ego')
    previous = {**sd, 'token': 'prev', 'filename': 'prev.bin', 'timestamp': 1750000,
                'prev': '', 'ego_pose_token': 'prev-ego'}
    records = {('sample_data', 'lidar'): sd, ('sample_data', 'prev'): previous,
               ('calibrated_sensor', 'cal'): {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]},
               ('ego_pose', 'ego'): {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]},
               ('ego_pose', 'prev-ego'): {'translation': [1, 0, 0], 'rotation': [1, 0, 0, 0]}}
    nusc.get = lambda table, token: records[table, token]
    monkeypatch.setattr(dataset, 'open_nuscenes', lambda cfg: nusc)
    monkeypatch.setattr(dataset, 'selected_samples', lambda *args: [(nusc.scene[0], nusc.sample[0])])
    monkeypatch.setitem(sys.modules, 'nuscenes.utils.splits', SimpleNamespace(
        create_splits_scenes=lambda: {'mini_val': ['scene-1094']}))
    cfg = {'dataset': {'root': str(root), 'version': 'v1.0-mini', 'split': 'mini_val', 'scenes': []},
           'detector': {'sweeps': 9}, 'run_dir': str(tmp_path / 'run')}
    result = dataset.prepare(cfg, pcd_dir=inputs)
    frame = result['frames'][0]
    destination = tmp_path / 'run' / frame['pcd_file']
    assert destination.read_bytes() == path.read_bytes()
    assert frame['pcd_sha256'] == file_digest(path) and frame['sample_token'] == 'sample'
    assert frame['sweeps'][0]['timestamp_us'] == 1750000
    np.testing.assert_allclose(np.array(frame['sweeps'][0]['target_from_source'])[:3, 3], [1, 0, 0])
    assert dataset.prepare(cfg, pcd_dir=inputs) == result
    from prelabel.io import write_json
    changed_manifest = {**result, 'coordinate_frame': 'changed-without-updating-signature'}
    write_json(destination.parent.parent / 'manifest.json', changed_manifest)
    with pytest.raises(ValueError, match='Manifest integrity'):
        dataset.prepare(cfg, pcd_dir=inputs)
    write_json(destination.parent.parent / 'manifest.json', result)
    changed = points[:, :4].copy()
    changed[0, 3] += 1
    write_xyzi(path, changed)
    with pytest.raises(ValueError, match='do not exactly match'):
        dataset.prepare(cfg, pcd_dir=inputs)
