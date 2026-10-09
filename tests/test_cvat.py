from copy import deepcopy
from pathlib import Path
import pytest
from prelabel.cvat import (attrs_by_name, dry_run, export_annotations, frame_mapping,
                          label_maps, upload, validate_created, CVATClient)
from prelabel.geometry import geometry_delta
from prelabel.io import read_json, write_json


def test_upload_metadata_readback_resume_and_human_edits(run_fixture, fake_cvat):
    cfg, manifest, boxes = run_fixture
    state = upload(cfg, fake_cvat)
    assert state["phase"] == "complete"
    assert fake_cvat.create_calls == 1
    assert fake_cvat.data_calls == 1
    assert fake_cvat.annotation_calls == 1
    assert len(fake_cvat.shapes) == 2
    assert fake_cvat.shapes[0]["frame"] == 1
    assert fake_cvat.shapes[1]["frame"] == 0
    # Simulate a user edit, deletion, and addition in CVAT.
    fake_cvat.shapes[0]["points"][0] += 1
    edited = deepcopy(fake_cvat.shapes[0])
    fake_cvat.shapes.pop(1)
    manual = deepcopy(edited)
    manual.update(id=500, attributes=[], label_id=2)
    fake_cvat.shapes.append(manual)
    upload(cfg, fake_cvat)
    assert fake_cvat.annotation_calls == 1
    assert fake_cvat.shapes[0] == edited
    assert len(fake_cvat.shapes) == 2  # Deleted original is not restored.
    reviewed = export_annotations(cfg, fake_cvat)
    assert reviewed[0].sample_token == "sample-0"
    assert reviewed[0].prediction_id == boxes[0].prediction_id
    assert reviewed[0].identity_source == "annotation_id"
    assert geometry_delta(boxes[0], reviewed[0])["center_m"] == pytest.approx(1)
    assert reviewed[1].class_name == "bicycle"
    assert reviewed[1].prediction_id == "manual:5:500"
    assert read_json(Path(cfg["run_dir"]) / "baseline.json")["boxes"] == [b.to_dict() for b in boxes]


def test_response_loss_after_commit_recovers_without_duplicates(run_fixture, fake_cvat):
    cfg, _, _ = run_fixture
    fake_cvat.fail_after_commit = True
    with pytest.raises(ConnectionError):
        upload(cfg, fake_cvat)
    assert len(fake_cvat.shapes) == 2
    # Human edits can happen before the retry; these must survive recovery.
    fake_cvat.shapes[0]["points"][1] += 0.6
    upload(cfg, fake_cvat)
    assert fake_cvat.annotation_calls == 1
    assert len(fake_cvat.shapes) == 2
    assert fake_cvat.shapes[0]["points"][1] == pytest.approx(2.6)


def test_uncertain_commit_does_not_risk_restoring_a_human_deletion(run_fixture, fake_cvat):
    cfg, _, _ = run_fixture
    fake_cvat.fail_after_commit = True
    with pytest.raises(ConnectionError):
        upload(cfg, fake_cvat)
    fake_cvat.shapes.pop()
    with pytest.raises(RuntimeError, match="uncertain"):
        upload(cfg, fake_cvat)
    assert fake_cvat.annotation_calls == 1
    assert len(fake_cvat.shapes) == 1


def test_failed_readback_retains_commit_ids(run_fixture, fake_cvat):
    cfg, _, _ = run_fixture
    fake_cvat.corrupt_readback = True
    with pytest.raises(ValueError, match="geometry"):
        upload(cfg, fake_cvat)
    state = read_json(Path(cfg["run_dir"]) / "cvat_state.json")
    assert len(state["applied"]) == 2
    fake_cvat.corrupt_readback = False
    upload(cfg, fake_cvat)
    assert fake_cvat.annotation_calls == 1


def test_dryrun_is_offline_and_explicitly_uses_placeholder_ids(run_fixture):
    cfg, _, _ = run_fixture
    payload = dry_run(cfg)
    doc = read_json(Path(cfg["run_dir"]) / "dry_run.json")
    assert doc["offline_placeholder_ids"] is True
    assert all(s["label_id"] < 0 for s in payload["shapes"])
    assert not (Path(cfg["run_dir"]) / "cvat_state.json").exists()


def test_export_restored_ids_and_unknown_identity(run_fixture, fake_cvat):
    cfg, _, boxes = run_fixture
    upload(cfg, fake_cvat)
    for shape in fake_cvat.shapes:
        shape["id"] += 1000
    reviewed = export_annotations(cfg, fake_cvat, ids_recreated=True)
    assert [b.prediction_id for b in reviewed] == [b.prediction_id for b in boxes]
    assert all(b.identity_source == "prediction_attribute_after_id_loss" for b in reviewed)
    fake_cvat.shapes[0]["attributes"] = []
    reviewed = export_annotations(cfg, fake_cvat, ids_recreated=True)
    assert reviewed[0].identity_source == "unresolved_after_id_loss"


def test_duplicate_copy_identity_requires_resolution(run_fixture, fake_cvat):
    cfg, _, _ = run_fixture
    upload(cfg, fake_cvat)
    copied = deepcopy(fake_cvat.shapes[0])
    copied["id"] = 999
    fake_cvat.shapes.append(copied)
    with pytest.raises(ValueError, match="Duplicate restored"):
        export_annotations(cfg, fake_cvat)


def test_raw_export_retained_when_tracks_unsupported(run_fixture, fake_cvat):
    cfg, _, _ = run_fixture
    upload(cfg, fake_cvat)
    fake_cvat.tracks = [{"id": 1}]
    with pytest.raises(ValueError, match="standalone"):
        export_annotations(cfg, fake_cvat)
    assert (Path(cfg["run_dir"]) / "cvat_export_raw.json").exists()


@pytest.mark.parametrize("change", ["missing", "duplicate", "deleted", "step"])
def test_invalid_frame_metadata(run_fixture, change):
    _, manifest, _ = run_fixture
    meta = {"frames": [{"name": Path(f["pcd_file"]).name} for f in manifest["frames"]]}
    if change == "missing":
        meta["frames"].pop()
    elif change == "duplicate":
        meta["frames"][1] = meta["frames"][0]
    elif change == "deleted":
        meta["deleted_frames"] = [0]
    else:
        meta["frame_filter"] = "step=2"
    with pytest.raises(ValueError):
        frame_mapping(manifest, meta)


def test_target_server_change_is_rejected(run_fixture, fake_cvat):
    cfg, _, _ = run_fixture
    upload(cfg, fake_cvat)
    fake_cvat.url = "http://another-server"
    with pytest.raises(ValueError, match="server/organization"):
        upload(cfg, fake_cvat)


def test_unknown_class_mapping_rejected(fake_cvat):
    with pytest.raises(ValueError, match="one-to-one"):
        label_maps(fake_cvat.labels_data, ["car", "bicycle"], {"car": "vehicle", "bicycle": "vehicle"})


class Response:
    def __init__(self, data, status=200):
        self.data, self.status_code = data, status
        self.content = b"json"

    def json(self):
        return self.data


class Session:
    def __init__(self, responses):
        self.headers, self.responses, self.calls = {}, iter(responses), []

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return next(self.responses)


def test_client_version_token_and_no_secret_echo(monkeypatch):
    monkeypatch.setenv("CVAT_URL", "http://server")
    monkeypatch.setenv("CVAT_TOKEN", "test-secret")
    session = Session([Response({"version": "2.20.0"}), Response({"error": "test-secret"}, 403)])
    client = CVATClient({"expected_version": "2.20.0", "timeout_seconds": 10}, session)
    assert client.check_version()["version"] == "2.20.0"
    assert session.headers["Authorization"].endswith("test-secret")
    with pytest.raises(RuntimeError) as error:
        client.request("GET", "/api/tasks")
    assert "test-secret" not in str(error.value)


def test_client_rejects_unaudited_version(monkeypatch):
    monkeypatch.setenv("CVAT_URL", "http://server")
    monkeypatch.setenv("CVAT_TOKEN", "fake")
    client = CVATClient({"expected_version": "2.20.0", "timeout_seconds": 10}, Session([Response({"version": "2.21.0"})]))
    with pytest.raises(ValueError, match="audited"):
        client.check_version()
