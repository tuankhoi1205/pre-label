from abc import ABC, abstractmethod
import logging
from pathlib import Path
import time
import numpy as np
from .config import dataset_root
from .dataset import load_manifest, validate_manifest
from .geometry import yaw_quaternion
from .io import digest, file_digest, read_json, write_json
from .schema import Cuboid

LOG = logging.getLogger(__name__)


class DetectorAdapter(ABC):
    @abstractmethod
    def predict(self, frame, root, *, run_id=None):
        """Return (raw JSON, list[Cuboid]); boxes in the keyframe LiDAR frame."""


def model_input(frame, root, box_type, box_mode):
    data = {"lidar_points": {"lidar_path": str(Path(root) / frame["lidar_file"])},
            "timestamp": frame["timestamp_us"] / 1e6, "sample_idx": frame["sample_token"],
            "box_type_3d": box_type, "box_mode_3d": box_mode, "axis_align_matrix": np.eye(4)}
    if frame["sweeps"]:
        sweeps = []
        for sweep in frame["sweeps"]:
            t = np.asarray(sweep["target_from_source"])
            # v1.4.0 loader consumes p @ matrix[:3,:3] - matrix[:3,3].
            # This transport field is deliberately NOT a standard inverse SE(3).
            transport = np.eye(4)
            transport[:3, :3] = t[:3, :3].T
            transport[:3, 3] = -t[:3, 3]
            sweeps.append({"timestamp": sweep["timestamp_us"] / 1e6,
                           "lidar_points": {"lidar_path": str(Path(root) / sweep["lidar_file"]),
                                            "lidar2sensor": transport.tolist()}})
        data["lidar_sweeps"] = sweeps
    # Omit lidar_sweeps when empty: the official loader then pads 9 keyframes.
    return data


def normalize_predictions(frame, raw, classes, thresholds, default_threshold, model, checkpoint, run_id):
    boxes = []
    count = len(raw["labels"])
    if any(len(raw[key]) != count for key in ("scores", "centers", "sizes", "yaws")):
        raise ValueError("Inconsistent detector output lengths")
    for i, (label, score, center, size, yaw) in enumerate(zip(raw["labels"], raw["scores"], raw["centers"], raw["sizes"], raw["yaws"])):
        if label < 0 or label >= len(classes):
            raise ValueError("Detector emitted an unknown class index")
        name = classes[label]
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Detector emitted an invalid score")
        if score < thresholds.get(name, default_threshold):
            continue
        boxes.append(Cuboid(frame["sample_token"], frame["frame_id"], f"{run_id}:{frame['sample_token']}:{i}",
                            name, float(score), list(center), list(size), yaw_quaternion(yaw),
                            model=model, checkpoint=checkpoint))
    return boxes


class MMDet3DCenterPoint(DetectorAdapter):
    def __init__(self, cfg, run_id):
        import torch
        import mmdet3d
        import mmcv
        import mmengine
        import mmdet
        from mmdet3d.apis import init_model
        from mmdet3d.utils import register_all_modules
        from mmengine.dataset import Compose
        from mmdet3d.structures import get_box_type
        from mmengine.config import Config
        from copy import deepcopy
        versions = {"mmdet3d": mmdet3d.__version__, "mmcv": mmcv.__version__, "mmengine": mmengine.__version__, "mmdet": mmdet.__version__}
        expected = {"mmdet3d": "1.4.0", "mmcv": "2.1.0", "mmengine": "0.10.5", "mmdet": "3.2.0"}
        if versions != expected or torch.__version__.split("+")[0] != "1.13.1":
            raise RuntimeError(f"Unsupported detector versions: {versions}, torch={torch.__version__}. Use the pinned Linux environment.")
        if not cfg["device"].startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("CenterPoint requires CUDA with compiled MMCV ops; see docs/INSTALL.md")
        register_all_modules(init_default_scope=True)
        config = Config.fromfile(cfg["config"])
        self.resolved_config = config.pretty_text
        pipeline = deepcopy(config.test_dataloader.dataset.pipeline)
        if len(pipeline) != 4 or pipeline[0]["type"] != "LoadPointsFromFile" or pipeline[1]["type"] != "LoadPointsFromMultiSweeps":
            raise ValueError("Unsupported test pipeline: use a documented CenterPoint profile")
        load, sweeps = pipeline[:2]
        if load["load_dim"] != 5 or load["use_dim"] != 5 or load["coord_type"] != "LIDAR":
            raise ValueError("Checkpoint must consume 5-column nuScenes LiDAR")
        if sweeps["sweeps_num"] != cfg["sweeps"] or sweeps["use_dim"] != [0, 1, 2, 3, 4] or not sweeps["pad_empty_sweeps"] or not sweeps["remove_close"]:
            raise ValueError("Unsupported sweep settings")
        sweeps["test_mode"] = True
        if list(config.class_names) != cfg["classes"]:
            raise ValueError("Configured class order differs from model config")
        self.model = init_model(config, cfg["checkpoint"], device=cfg["device"])
        metadata_classes = self.model.dataset_meta.get("classes")
        if metadata_classes is not None and list(metadata_classes) != cfg["classes"]:
            raise ValueError("Checkpoint class order differs from configured class order")
        self.pipeline = Compose(pipeline)
        self.box_type, self.box_mode = get_box_type("LiDAR")
        self.cfg, self.run_id = cfg, run_id

    def predict(self, frame, root, *, run_id=None):
        import torch
        from mmengine.dataset import pseudo_collate
        data = self.pipeline(model_input(frame, root, self.box_type, self.box_mode))
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            result = self.model.test_step(pseudo_collate([data]))[0].pred_instances_3d
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        boxes = result.bboxes_3d
        if not isinstance(boxes, self.box_type):
            raise ValueError("Expected LiDARInstance3DBoxes from CenterPoint")
        raw = {"tensor_bottom_center": boxes.tensor.detach().cpu().tolist(),
               "centers": boxes.gravity_center.detach().cpu().tolist(), "sizes": boxes.dims.detach().cpu().tolist(),
               "yaws": boxes.yaw.detach().cpu().tolist(), "labels": result.labels_3d.detach().cpu().tolist(),
               "scores": result.scores_3d.detach().cpu().tolist(), "forward_seconds": elapsed,
               "input_points": int(data["inputs"]["points"].shape[0]), "prior_sweeps": len(frame["sweeps"])}
        normalized = normalize_predictions(frame, raw, self.cfg["classes"], self.cfg.get("class_thresholds", {}),
                                           self.cfg["confidence"], "CenterPoint-MMDetection3D-1.4.0",
                                           self.cfg["checkpoint_sha256"], run_id or self.run_id)
        return raw, normalized


def load_predictions(run, manifest=None):
    run = Path(run)
    manifest = manifest or load_manifest(run)
    state = read_json(run / "inference_state.json")
    if state["manifest_signature"] != manifest["signature"]:
        raise ValueError("Predictions do not belong to this manifest")
    records, boxes = [], []
    for frame in manifest["frames"]:
        path = run / "predictions" / f"{frame['frame_id']:06d}.json"
        if not path.exists():
            raise ValueError(f"Inference incomplete at frame {frame['frame_id']}; resume infer before upload/evaluate")
        record = read_json(path)
        if record["inference_signature"] != state["signature"] or record["sample_token"] != frame["sample_token"]:
            raise ValueError("Prediction provenance mismatch")
        records.append(record)
        boxes.extend(Cuboid.from_dict(b) for b in record["boxes"])
    return boxes, records


def initialize_inference(cfg):
    """Validate inputs and establish the same provenance for CLI and UI inference."""
    run, root = Path(cfg["run_dir"]), dataset_root(cfg)
    manifest = load_manifest(run)
    validate_manifest(manifest, run, root, check_sweeps=True)
    d = dict(cfg["detector"])
    for key in ("config", "checkpoint"):
        if not Path(d[key]).is_file():
            raise ValueError(f"Missing {key}: {d[key]}. Run fetch-model.")
    d["checkpoint_sha256"] = file_digest(d["checkpoint"])
    from .assets import MODEL_PROFILES
    profile = MODEL_PROFILES.get(Path(d["config"]).name)
    if profile is None or not d["checkpoint_sha256"].startswith(profile["sha256_prefix"]):
        raise ValueError("Config/checkpoint is not an audited official profile; use fetch-model or extend and verify a separate adapter")
    # Hash the whole official config tree so inherited config changes invalidate resume.
    config_dir = Path(d["config"]).parent.parent
    tree_hash = digest({str(p.relative_to(config_dir)): file_digest(p) for p in sorted(config_dir.rglob("*.py"))})
    state = bind_inference_state(run, manifest, d, tree_hash)
    return manifest, root, d, state


def bind_inference_state(run, manifest, d, tree_hash):
    """Bind another verified manifest to the already validated model profile."""
    run = Path(run)
    signature = digest({"manifest": manifest["signature"], "detector": {k: v for k, v in d.items() if k not in ("config", "checkpoint")}, "config_tree": tree_hash})
    state_path = run / "inference_state.json"
    state = {"signature": signature, "run_id": signature[:16], "manifest_signature": manifest["signature"], "detector": d, "config_tree_sha256": tree_hash}
    if state_path.exists() and read_json(state_path)["signature"] != signature:
        raise ValueError("Detector configuration/input changed. Use a new run_dir.")
    write_json(state_path, state)
    return state


def infer(cfg):
    run = Path(cfg["run_dir"])
    manifest, root, d, state = initialize_inference(cfg)
    signature = state["signature"]
    pending = []
    for frame in manifest["frames"]:
        path = run / "predictions" / f"{frame['frame_id']:06d}.json"
        if path.exists():
            record = read_json(path)
            if record["inference_signature"] != signature or record["sample_token"] != frame["sample_token"]:
                raise ValueError("Stale prediction record; use a new run")
        else:
            pending.append(frame)
    if not pending:
        LOG.info("Inference already complete")
        return load_predictions(run, manifest)[0]
    adapter = MMDet3DCenterPoint(d, state["run_id"])
    write_json(run / "resolved_detector_config.json", {"config": adapter.resolved_config})
    for frame in pending:
        start = time.perf_counter()
        try:
            raw, boxes = adapter.predict(frame, root)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError("CUDA out of memory. Keep batch=1, use a larger GPU or a fresh run with the documented pillar profile; do not silently reduce sweeps.") from exc
            raise
        record = {"inference_signature": signature, "sample_token": frame["sample_token"], "frame_id": frame["frame_id"],
                  "raw": raw, "boxes": [b.to_dict() for b in boxes], "frame_seconds": time.perf_counter() - start, "source": "real_detector"}
        write_json(run / "predictions" / f"{frame['frame_id']:06d}.json", record)
        LOG.info("Inferred frame %d/%d: %d boxes, forward %.3fs", frame["frame_id"] + 1, len(manifest["frames"]), len(boxes), raw["forward_seconds"])
    return load_predictions(run, manifest)[0]
