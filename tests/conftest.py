from copy import deepcopy
from pathlib import Path
import pytest
import numpy as np
from prelabel.cvat import label_spec
from prelabel.dataset import write_pcd
from prelabel.geometry import yaw_quaternion
from prelabel.io import digest, file_digest, write_json
from prelabel.schema import Cuboid


@pytest.fixture
def box():
    return Cuboid("sample-0", 0, "run:sample-0:0", "car", 0.8, [4, 2, 1.5], [4.6, 1.7, 1.5], yaw_quaternion(0))


@pytest.fixture
def run_fixture(tmp_path):
    run = tmp_path / "run"
    frames = []
    for i in range(2):
        pcd = f"pcd/{i:06d}_sample-{i}.pcd"
        write_pcd(run / pcd, np.array([[1, 2, 3, 100, 5]], dtype=float))
        frames.append({"frame_id": i, "sample_token": f"sample-{i}", "sample_data_token": f"lidar-{i}",
                       "lidar_file": f"samples/{i}.bin", "pcd_file": pcd, "pcd_sha256": file_digest(run / pcd),
                       "timestamp_us": 1000000 + i, "sweeps": [], "global_from_lidar": np.eye(4).tolist()})
    manifest = {"schema_version": 1, "coordinate_frame": "lidar_sensor", "selection": {"source": "synthetic_fixture"}, "frames": frames}
    manifest["signature"] = digest(manifest)
    write_json(run / "manifest.json", manifest)
    write_json(run / "inference_state.json", {"signature": "test-inference", "run_id": "test-run", "manifest_signature": manifest["signature"]})
    boxes = [Cuboid(f"sample-{i}", i, f"test-run:sample-{i}:0", "car", 0.9, [8 + i, 2, 1], [4.6, 1.7, 1.5], yaw_quaternion(0.4), model="synthetic", checkpoint="synthetic") for i in range(2)]
    for b in boxes:
        write_json(run / "predictions" / f"{b.frame_id:06d}.json", {"inference_signature": "test-inference", "sample_token": b.sample_token,
                    "frame_id": b.frame_id, "boxes": [b.to_dict()], "raw": {"forward_seconds": 0}, "frame_seconds": 0, "source": "synthetic_fixture"})
    cfg = {"run_dir": str(run), "project_root": str(tmp_path), "detector": {"classes": ["car", "bicycle"]},
           "cvat": {"expected_version": "2.20.0", "task_id": None, "task_name": "test", "label_mapping": {}, "timeout_seconds": 10, "poll_seconds": 0},
           "evaluation": {"match_iou": 0.5, "center_tolerance_m": 0.02, "size_tolerance_m": 0.02, "rotation_tolerance_rad": 0.01}}
    return cfg, manifest, boxes


class FakeCVAT:
    """In-memory contract fixture, not a CVAT server or detector."""
    url = "http://fake-cvat"

    def __init__(self, manifest, classes=("car", "bicycle")):
        self.manifest = manifest
        self.labels_data = [{**label_spec(c), "id": i + 1, "attributes": [{**a, "id": i * 10 + j + 1} for j, a in enumerate(label_spec(c)["attributes"])]} for i, c in enumerate(classes)]
        self.task = None
        self.shapes = []
        self.version = 0
        self.create_calls = 0
        self.data_calls = 0
        self.annotation_calls = 0
        self.next_id = 101
        self.fail_after_commit = False
        self.fail_before_commit = False
        self.corrupt_readback = False
        self.tracks = []

    def check_version(self):
        return {"version": "2.20.0"}

    def labels(self, task_id):
        return deepcopy(self.labels_data)

    def annotations(self, task_id):
        shapes = deepcopy(self.shapes)
        if self.corrupt_readback and shapes:
            shapes[0]["points"][0] += 10
        return {"version": self.version, "shapes": shapes, "tracks": deepcopy(self.tracks), "tags": []}

    def all_pages(self, path, **params):
        if path == "/api/tasks":
            return [deepcopy(self.task)] if self.task else []
        if path == "/api/requests":
            return []
        raise AssertionError(path)

    def upload_data(self, task_id, manifest, run):
        self.data_calls += 1
        return {"rq_id": "rq-test"}

    def wait_request(self, rq_id):
        self.task.update(size=len(self.manifest["frames"]), dimension="3d")

    def request(self, method, path, **kwargs):
        if method == "POST" and path == "/api/tasks":
            self.create_calls += 1
            self.task = {"id": 5, "name": kwargs["json"]["name"], "size": 0, "dimension": "2d"}
            return deepcopy(self.task)
        if method == "GET" and path == "/api/tasks/5":
            return deepcopy(self.task)
        if method == "GET" and path == "/api/tasks/5/data/meta":
            # Deliberately reversed: the adapter must trust metadata, not upload order.
            return {"start_frame": 0, "frame_filter": "", "deleted_frames": [],
                    "frames": [{"name": Path(f["pcd_file"]).name} for f in reversed(self.manifest["frames"])]}
        if method == "PATCH" and path == "/api/tasks/5/annotations":
            assert kwargs["params"] == {"action": "create"}
            if self.fail_before_commit:
                self.fail_before_commit = False
                raise ConnectionError("Synthetic failure before commit")
            self.annotation_calls += 1
            shapes = deepcopy(kwargs["json"]["shapes"])
            for s in shapes:
                s["id"] = self.next_id
                self.next_id += 1
            self.shapes.extend(shapes)
            self.version += 1
            if self.fail_after_commit:
                self.fail_after_commit = False
                raise ConnectionError("Synthetic response loss after commit")
            return {"version": self.version, "shapes": deepcopy(shapes)}
        raise AssertionError((method, path, kwargs))


@pytest.fixture
def fake_cvat(run_fixture, monkeypatch):
    monkeypatch.delenv("CVAT_ORG", raising=False)
    return FakeCVAT(run_fixture[1])
