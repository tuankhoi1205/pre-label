import numpy as np
import pytest
from prelabel.dataset import read_lidar, write_pcd
from prelabel.detector import model_input, normalize_predictions, load_predictions
from prelabel.geometry import pose_matrix, yaw_quaternion


def test_five_dimensions_and_pcd_binary(tmp_path):
    p = np.array([[1, 2, 3, 100, 7], [4, 5, 6, 255, 31]], dtype="<f4")
    source, output = tmp_path / "points.bin", tmp_path / "cloud.pcd"
    p.tofile(source)
    np.testing.assert_array_equal(read_lidar(source), p)
    write_pcd(output, p)
    header, binary = output.read_bytes().split(b"DATA binary\n")
    assert b"FIELDS x y z" in header
    assert b"WIDTH 2" in header
    np.testing.assert_array_equal(np.frombuffer(binary, dtype="<f4").reshape(-1, 3), p[:, :3])


@pytest.mark.parametrize("values", [[], [1, 2, 3, 4], [1, 2, 3, 4, 40], [1, 2, float("inf"), 4, 1], [1, 2, 3, 4, 1.5]])
def test_invalid_lidar(tmp_path, values):
    path = tmp_path / "bad.bin"
    np.array(values, dtype="<f4").tofile(path)
    with pytest.raises(ValueError):
        read_lidar(path)


def test_sweep_native_loader_contract(tmp_path):
    t = pose_matrix([3, -1, 0.5], yaw_quaternion(0.7))
    frame = {"sample_token": "s", "timestamp_us": 5000000, "lidar_file": "key.bin",
             "sweeps": [{"lidar_file": "prev.bin", "timestamp_us": 4750000, "target_from_source": t.tolist()}]}
    data = model_input(frame, tmp_path, "box-type", "box-mode")
    sweep = data["lidar_sweeps"][0]
    matrix = np.array(sweep["lidar_points"]["lidar2sensor"])
    point = np.array([2, 1, 0])
    # Independent homogeneous transform vs exact native v1.4.0 loader expression.
    np.testing.assert_allclose(point @ matrix[:3, :3] - matrix[:3, 3], (t @ [*point, 1])[:3])
    assert data["timestamp"] - sweep["timestamp"] == pytest.approx(0.25)
    frame["sweeps"] = []
    assert "lidar_sweeps" not in model_input(frame, tmp_path, None, None)


def test_thresholds_and_no_second_nms():
    frame = {"sample_token": "s", "frame_id": 0}
    raw = {"labels": [0, 0, 1], "scores": [0.6, 0.61, 0.3], "centers": [[1, 2, 3]] * 3,
           "sizes": [[4.6, 1.7, 1.5]] * 3, "yaws": [0, 0, 0.7]}
    boxes = normalize_predictions(frame, raw, ["car", "bicycle"], {"bicycle": 0.25}, 0.5, "test-model", "hash", "run")
    assert len(boxes) == 3  # Overlapping predictions are retained, NMS belongs to the detector.
    assert [b.class_name for b in boxes] == ["car", "car", "bicycle"]
    assert boxes[0].center_xyz == [1, 2, 3]  # Function consumes gravity_center, not tensor bottom center.
    assert boxes[0].prediction_id == "run:s:0"


def test_incomplete_inference_is_not_uploaded(run_fixture):
    from pathlib import Path
    cfg, _, _ = run_fixture
    (Path(cfg["run_dir"]) / "predictions" / "000001.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        load_predictions(cfg["run_dir"])


def test_prepare_builds_sensor_sweep_manifest_without_using_gt(tmp_path, monkeypatch):
    from prelabel import dataset
    root = tmp_path / "nusc"
    root.mkdir()
    for name in ("key.bin", "sweep.bin"):
        np.array([[4, 1, 2, 90, 8]], dtype="<f4").tofile(root / name)
    key_sd = {"token": "key", "filename": "key.bin", "timestamp": 2000000, "prev": "sweep",
              "calibrated_sensor_token": "cal", "ego_pose_token": "ego_key"}
    sweep_sd = {"token": "sweep", "filename": "sweep.bin", "timestamp": 1750000, "prev": "",
                "calibrated_sensor_token": "cal", "ego_pose_token": "ego_sweep"}
    cal = {"translation": [0.8, -0.2, 1.5], "rotation": yaw_quaternion(0.2)}
    ego_key = {"translation": [8, -2, 0], "rotation": yaw_quaternion(0.5)}
    ego_sweep = {"translation": [7, -1.5, 0], "rotation": yaw_quaternion(0.45)}
    tables = {"sample_data": {"key": key_sd, "sweep": sweep_sd}, "calibrated_sensor": {"cal": cal},
              "ego_pose": {"ego_key": ego_key, "ego_sweep": ego_sweep}}

    class NoGroundTruthNusc:
        def get(self, table, token):
            assert table != "sample_annotation", "prepare must never access GT"
            return tables[table][token]

    monkeypatch.setattr(dataset, "open_nuscenes", lambda cfg: NoGroundTruthNusc())
    monkeypatch.setattr(dataset, "selected_samples", lambda n, c, limit: [({"name": "scene-fixture"}, {"token": "sample", "data": {"LIDAR_TOP": "key"}})])
    cfg = {"run_dir": str(tmp_path / "run"), "dataset": {"root": str(root), "version": "v1.0-mini", "split": "mini_val"}, "detector": {"sweeps": 9}}
    manifest = dataset.prepare(cfg, limit=1)
    frame = manifest["frames"][0]
    assert frame["sample_data_token"] == "key"
    assert len(frame["sweeps"]) == 1
    t_sensor = pose_matrix(cal["translation"], cal["rotation"])
    t_key = pose_matrix(ego_key["translation"], ego_key["rotation"]) @ t_sensor
    t_sweep = pose_matrix(ego_sweep["translation"], ego_sweep["rotation"]) @ t_sensor
    np.testing.assert_allclose(frame["sweeps"][0]["target_from_source"], np.linalg.inv(t_key) @ t_sweep)
    assert dataset.prepare(cfg, limit=1) == manifest  # Resume does not rewrite selection.


@pytest.mark.detector
def test_optional_native_multisweeps(tmp_path):
    pytest.importorskip("mmdet3d")
    from mmdet3d.datasets.transforms import LoadPointsFromFile, LoadPointsFromMultiSweeps
    from mmdet3d.structures import get_box_type
    key = np.array([[2, 2, 1, 80, 3]], dtype="<f4")
    prev = np.array([[4, 1, 2, 90, 8], [0.1, 0.1, 2, 50, 2]], dtype="<f4")
    key.tofile(tmp_path / "key.bin")
    prev.tofile(tmp_path / "prev.bin")
    t = pose_matrix([3, -1, 0.5], yaw_quaternion(0.7))
    frame = {"sample_token": "s", "timestamp_us": 5000000, "lidar_file": "key.bin",
             "sweeps": [{"lidar_file": "prev.bin", "timestamp_us": 4750000, "target_from_source": t.tolist()}]}
    data = model_input(frame, tmp_path, *get_box_type("LiDAR"))
    data = LoadPointsFromFile(coord_type="LIDAR", load_dim=5, use_dim=5)(data)
    result = LoadPointsFromMultiSweeps(sweeps_num=9, use_dim=[0, 1, 2, 3, 4], pad_empty_sweeps=True, remove_close=True, test_mode=True)(data)["points"].numpy()
    assert result.shape == (2, 5)
    np.testing.assert_allclose(result[1, :3], (t @ [4, 1, 2, 1])[:3], atol=1e-6)
    assert result[0, 4] == 0
    assert result[1, 4] == pytest.approx(0.25)
