from pathlib import Path
import zipfile

import pytest

from prelabel.context import CAMERAS, prepare_camera_context
from prelabel.cvat import CVATClient
from prelabel.io import read_json, write_json


@pytest.fixture
def camera_dataset(run_fixture):
    cfg, manifest, _ = run_fixture
    root = Path(cfg["project_root"]) / "nuscenes"
    version = "v1.0-mini"
    manifest["selection"]["version"] = version
    samples, data, calibrations, sensors = [], [], [], []
    for i, channel in enumerate(("LIDAR_TOP", *CAMERAS)):
        sensors.append({"token": f"sensor-{i}", "channel": channel})
        calibrations.append({"token": f"calibration-{i}", "sensor_token": f"sensor-{i}"})
    for frame in manifest["frames"]:
        token = frame["sample_token"]
        samples.append({"token": token})
        data.append({"token": frame["sample_data_token"], "sample_token": token,
                     "calibrated_sensor_token": "calibration-0", "is_key_frame": True})
        for i, channel in enumerate(CAMERAS, 1):
            file = f"samples/{channel}/{token}.jpg"
            path = root / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic:{token}:{channel}".encode())
            record = {"token": f"{token}-{channel}", "sample_token": token, "filename": file,
                      "calibrated_sensor_token": f"calibration-{i}", "is_key_frame": True,
                      "timestamp": frame["timestamp_us"] + 1000}
            data.append(record)
            # A non-keyframe in the same channel must never replace the matching camera keyframe.
            data.append({**record, "token": record["token"] + "-sweep", "filename": "wrong.jpg", "is_key_frame": False})
    for name, records in (("sample", samples), ("sample_data", list(reversed(data))),
                          ("calibrated_sensor", calibrations), ("sensor", sensors)):
        write_json(root / version / f"{name}.json", records)
    return cfg, manifest, root


def test_camera_context_uses_sample_keyframes_and_preserves_source(camera_dataset):
    cfg, manifest, root = camera_dataset
    context = prepare_camera_context(manifest, root, cfg["run_dir"])
    assert len(context["images"]) == 12
    for record in context["images"]:
        assert record["sample_token"] == manifest["frames"][record["frame_id"]]["sample_token"]
        assert record["lidar_time_delta_us"] == 1000
        assert (Path(cfg["run_dir"]) / "context" / record["context_file"]).read_bytes() == (root / record["source_file"]).read_bytes()
    assert prepare_camera_context(manifest, root, cfg["run_dir"]) == context


def test_context_does_not_replace_a_changed_image(camera_dataset):
    cfg, manifest, root = camera_dataset
    context = prepare_camera_context(manifest, root, cfg["run_dir"])
    target = Path(cfg["run_dir"]) / "context" / context["images"][0]["context_file"]
    target.write_bytes(b"changed by user")
    with pytest.raises(ValueError, match="differs"):
        prepare_camera_context(manifest, root, cfg["run_dir"])
    assert target.read_bytes() == b"changed by user"


def test_upload_archive_has_native_3d_context_structure(camera_dataset):
    cfg, manifest, root = camera_dataset
    context = prepare_camera_context(manifest, root, cfg["run_dir"])
    client = CVATClient.__new__(CVATClient)
    client.cfg = cfg["cvat"]
    captured = []

    def request(method, path, **kwargs):
        assert method == "POST" and path == "/api/tasks/2/data"
        files = kwargs["files"]
        assert len(files) == 1 and files[0][1][2] == "application/zip"
        with zipfile.ZipFile(files[0][1][1]) as archive:
            captured.extend(archive.namelist())
            assert archive.testzip() is None
        return {"rq_id": "fixture"}

    client.request = request
    assert client.upload_data(2, manifest, cfg["run_dir"]) == {"rq_id": "fixture"}
    assert set(captured) == {f"pcd/{Path(f['pcd_file']).name}" for f in manifest["frames"]} | {r["context_file"] for r in context["images"]}
    assert all(f"related_images/{Path(f['pcd_file']).name.replace('.pcd', '_pcd')}/CAM_FRONT.jpg" in captured for f in manifest["frames"])


def test_upload_rejects_context_from_another_run(camera_dataset):
    cfg, manifest, root = camera_dataset
    prepare_camera_context(manifest, root, cfg["run_dir"])
    path = Path(cfg["run_dir"]) / "context/manifest.json"
    context = read_json(path)
    context["manifest_signature"] = "another-run"
    write_json(path, context)
    client = CVATClient.__new__(CVATClient)
    client.cfg = cfg["cvat"]
    client.request = lambda *args, **kwargs: pytest.fail("Must reject before sending any data")
    with pytest.raises(ValueError, match="different LiDAR run"):
        client.upload_data(2, manifest, cfg["run_dir"])
