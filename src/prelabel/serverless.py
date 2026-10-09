"""CenterPoint adapter for CVAT's existing Nuclio detector protocol.

CVAT 2.20 sends the original PCD bytes in the historical `image` field.
The PCD identifies a prepared sample; inference uses its original intensity
and calibrated sweeps, never the XYZ-only preview as a substitute.
"""
import base64
import binascii
from copy import deepcopy
import hashlib
from pathlib import Path
import time

from .dataset import validate_manifest, prepare_selection
from .detector import bind_inference_state, initialize_inference, MMDet3DCenterPoint
from .geometry import cvat_points
from .io import read_json, run_lock, write_json
from .schema import Cuboid


class CenterPointFunction:
    def __init__(self, cfg, adapter_factory=MMDet3DCenterPoint):
        self.cfg = cfg
        self.run = Path(cfg["run_dir"])
        with run_lock(self.run):
            self.manifest, self.root, self.detector, self.state = initialize_inference(cfg)
        self.by_pcd = {}
        for frame in self.manifest["frames"]:
            if frame["pcd_sha256"] in self.by_pcd:
                raise ValueError("Ambiguous identical PCD files in manifest; prepare an unambiguous selection")
            self.by_pcd[frame["pcd_sha256"]] = frame
        self.adapter_factory = adapter_factory
        self.adapter = None
        self.resolver = None
        self.uploads = {}

    def resolve_input(self, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get("image"), str):
            raise ValueError("Expected CVAT detector request containing a base64 PCD in 'image'")
        if len(payload["image"]) > 44_000_000:
            raise ValueError("PCD request exceeds 32 MiB")
        try:
            pcd = base64.b64decode(payload["image"], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid base64 PCD") from exc
        frame = self.by_pcd.get(hashlib.sha256(pcd).hexdigest())
        if frame is None:
            return self._resolve_upload(pcd)
        return self.run, frame, self.state

    def _resolve_upload(self, pcd):
        from .uploaded import NuScenesPCDResolver, parse_uploaded_pcd

        # Reject malformed requests before loading/indexing the dataset.
        _, sha = parse_uploaded_pcd(pcd)
        if sha in self.uploads:
            return self.uploads[sha]
        if self.resolver is None:
            self.resolver = NuScenesPCDResolver(self.cfg)
        scene, sample, split, sha = self.resolver.resolve(pcd)
        cfg = deepcopy(self.cfg)
        run = self.run / 'uploads' / sha
        cfg['run_dir'] = str(run)
        cfg['dataset'].update(split=split, scenes=[scene['name']], max_frames=1)
        with run_lock(run):
            path = run / 'pcd' / 'upload.pcd'
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_bytes() != pcd:
                    raise ValueError('Stored upload content changed')
            else:
                temporary = path.with_suffix('.pcd.tmp')
                temporary.write_bytes(pcd)
                temporary.replace(path)
            manifest = prepare_selection(cfg, self.resolver.nusc, [(scene, sample)],
                                         [(scene, sample, path, sha)])
            state = bind_inference_state(run, manifest, self.detector, self.state['config_tree_sha256'])
        result = run, manifest['frames'][0], state
        self.uploads[sha] = result
        return result

    def validate(self, payload):
        run, frame, state = self.resolve_input(payload)
        with run_lock(run):
            validate_manifest({'frames': [frame]}, run, self.root, check_sweeps=True)
        return {'ready': True, 'sample_token': frame['sample_token'], 'scene': frame.get('scene'),
                'sweeps': len(frame['sweeps']), 'run_id': state['run_id'], 'inference_started': False}

    def predict(self, payload):
        run, frame, state = self.resolve_input(payload)
        threshold = payload.get("threshold")
        if threshold is not None and (not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1):
            raise ValueError("Threshold must be in [0,1]")
        path = run / "predictions" / f"{frame['frame_id']:06d}.json"
        with run_lock(run):
            # Detect modified original inputs even when using a cached prediction.
            validate_manifest({"frames": [frame]}, run, self.root, check_sweeps=True)
            if path.exists():
                record = read_json(path)
                if (record["inference_signature"] != state["signature"] or
                        record["sample_token"] != frame["sample_token"]):
                    raise ValueError("Cached prediction provenance mismatch")
                boxes = [Cuboid.from_dict(b) for b in record["boxes"]]
            else:
                if self.adapter is None:
                    self.adapter = self.adapter_factory(self.detector, state["run_id"])
                write_json(run / "resolved_detector_config.json", {"config": self.adapter.resolved_config})
                start = time.perf_counter()
                try:
                    raw, boxes = self.adapter.predict(frame, self.root, run_id=state['run_id'])
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise RuntimeError("CenterPoint CUDA out of memory; GPU 4 GB may be insufficient. Use a GPU with enough VRAM or a separate pillar run.") from exc
                    raise
                write_json(path, {"inference_signature": state["signature"],
                    "sample_token": frame["sample_token"], "frame_id": frame["frame_id"],
                    "raw": raw, "boxes": [b.to_dict() for b in boxes],
                    "frame_seconds": time.perf_counter() - start, "source": "real_detector"})
        return [{"label": b.class_name, "type": "cuboid", "points": cvat_points(b),
                 "confidence": str(b.confidence), "attributes": [
                     {"name": "prediction_id", "value": b.prediction_id},
                     {"name": "confidence", "value": str(b.confidence)},
                     {"name": "model", "value": b.model},
                     {"name": "checkpoint", "value": b.checkpoint},
                     {"name": "run_id", "value": self.state["run_id"]},
                 ]} for b in boxes if threshold is None or b.confidence >= threshold]
