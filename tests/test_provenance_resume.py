from pathlib import Path
import zipfile
import pytest
import numpy as np
from prelabel.assets import extract_repo
from prelabel.dataset import load_manifest
from prelabel.detector import infer
from prelabel.io import digest, file_digest, read_json, run_lock, write_json


def test_run_lock_and_release(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="locked"):
            with run_lock(tmp_path):
                pass
    assert not (tmp_path / ".lock").exists()


def test_manifest_mutation_rejected(run_fixture):
    cfg, manifest, _ = run_fixture
    manifest["frames"][0]["sample_token"] = "tampered"
    write_json(Path(cfg["run_dir"]) / "manifest.json", manifest)
    with pytest.raises(ValueError, match="integrity"):
        load_manifest(cfg["run_dir"])


def test_resume_uses_single_model_load_and_only_pending_frames(run_fixture, tmp_path, monkeypatch):
    from prelabel import detector, assets
    cfg, manifest, boxes = run_fixture
    run = Path(cfg["run_dir"])
    root = tmp_path / "dataset"
    for frame in manifest["frames"]:
        p = root / frame["lidar_file"]
        p.parent.mkdir(parents=True, exist_ok=True)
        np.array([[1, 2, 3, 80, 4]], dtype="<f4").tofile(p)
        frame["lidar_sha256"] = file_digest(p)
    manifest.pop("signature")
    manifest["signature"] = digest(manifest)
    write_json(run / "manifest.json", manifest)
    checkpoint = tmp_path / "mock-checkpoint.pth"
    checkpoint.write_bytes(b"synthetic checkpoint fixture, never loaded by torch")
    config = tmp_path / "configs" / "centerpoint" / "test_config.py"
    config.parent.mkdir(parents=True)
    config.write_text("fixture config", encoding="utf-8")
    cfg["dataset"] = {"root": str(root)}
    cfg["detector"].update(config=str(config), checkpoint=str(checkpoint), device="cuda:0", sweeps=9,
                            confidence=0.35, class_thresholds={}, backend="mmdet3d_centerpoint")
    monkeypatch.setitem(assets.MODEL_PROFILES, config.name, {"sha256_prefix": file_digest(checkpoint)[:8]})
    (run / "inference_state.json").unlink()
    for f in (run / "predictions").glob("*.json"):
        f.unlink()
    loads, calls = [], []

    class SyntheticAdapter:
        resolved_config = "synthetic adapter fixture"

        def __init__(self, d, run_id):
            loads.append(run_id)

        def predict(self, frame, root):
            calls.append(frame["frame_id"])
            return {"forward_seconds": 0.01}, [boxes[frame["frame_id"]]]

    monkeypatch.setattr(detector, "MMDet3DCenterPoint", SyntheticAdapter)
    infer(cfg)
    assert calls == [0, 1]
    assert len(loads) == 1
    infer(cfg)
    assert calls == [0, 1]  # No reload and no repeated inference after completion.
    (run / "predictions" / "000001.json").unlink()
    infer(cfg)
    assert calls == [0, 1, 1]
    assert len(loads) == 2
    cfg["detector"]["confidence"] = 0.5
    with pytest.raises(ValueError, match="changed"):
        infer(cfg)


def test_archive_cannot_escape_destination(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("mmdetection3d-1.4.0/../../outside.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_repo(archive, tmp_path / "models")


def test_evaluation_driver_writes_csv_json_with_synthetic_gt(run_fixture, fake_cvat, monkeypatch):
    # Exercise report orchestration with explicitly substituted test GT; no real benchmark claim.
    from prelabel.cvat import upload, export_annotations
    from prelabel.evaluation import evaluate
    cfg, _, boxes = run_fixture
    cfg["evaluation"].update(point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3], min_lidar_points=1)
    upload(cfg, fake_cvat)
    export_annotations(cfg, fake_cvat)
    monkeypatch.setattr("prelabel.evaluation.ground_truth", lambda c, m: boxes)
    result = evaluate(cfg)
    assert result["before"]["matched_pairs"] == 2
    assert result["after"]["mean_matched_iou"] == pytest.approx(1)
    assert result["annotation_time"]["status"] == "not_measured"
    for name in ("metrics.json", "matched_pairs.csv", "edits.csv", "quality_summary.csv", "summary.md"):
        assert (Path(cfg["run_dir"]) / "reports" / name).is_file()
