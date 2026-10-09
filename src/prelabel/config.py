import os
from pathlib import Path
import re
import yaml


def load_env(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid .env entry: {key}")
        os.environ.setdefault(key, value.strip().strip("\"'"))


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    root = (path.parent / raw.get("project_root", "..")).resolve()
    load_env(root / ".env")

    def expand(value):
        if isinstance(value, str):
            # Missing values remain placeholders, so offline commands can still run.
            return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m[1], m[0]), value)
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        return value

    cfg = expand(raw)
    cfg["project_root"] = str(root)
    for section, key in [(None, "run_dir"), ("dataset", "root"), ("detector", "config"), ("detector", "checkpoint")]:
        parent = cfg if section is None else cfg[section]
        if "${" not in str(parent[key]):
            p = Path(parent[key]).expanduser()
            parent[key] = str(p if p.is_absolute() else root / p)
    d = cfg["detector"]
    if d["backend"] != "mmdet3d_centerpoint":
        raise ValueError("Only mmdet3d_centerpoint is implemented; use its adapter contract for extensions.")
    if len(d["classes"]) != len(set(d["classes"])) or not d["classes"]:
        raise ValueError("Detector classes must be unique and nonempty")
    if set(d.get("class_thresholds", {})) - set(d["classes"]):
        raise ValueError("class_thresholds contains an unknown class")
    for threshold in [d["confidence"], *d.get("class_thresholds", {}).values(), cfg["evaluation"]["match_iou"]]:
        if not 0 <= threshold <= 1:
            raise ValueError("Thresholds must be in [0, 1]")
    if not 0 < cfg["evaluation"]["match_iou"] <= 1:
        raise ValueError("match_iou must be > 0")
    if d["sweeps"] != 9:
        raise ValueError("The supported CenterPoint checkpoints require 9 previous sweeps")
    if cfg["dataset"]["max_frames"] <= 0:
        raise ValueError("max_frames must be positive")
    e = cfg["evaluation"]
    if any(e[key] < 0 for key in ("center_tolerance_m", "size_tolerance_m", "rotation_tolerance_rad", "min_lidar_points")):
        raise ValueError("Evaluation tolerances and minimum point count must be nonnegative")
    roi = e["point_cloud_range"]
    if len(roi) != 6 or any(roi[i] >= roi[i + 3] for i in range(3)):
        raise ValueError("point_cloud_range must be [xmin,ymin,zmin,xmax,ymax,zmax]")
    if cfg["cvat"]["poll_seconds"] <= 0 or cfg["cvat"]["timeout_seconds"] <= 0:
        raise ValueError("CVAT poll/timeout must be positive")
    return cfg


def dataset_root(cfg):
    root = cfg["dataset"]["root"]
    if "${" in root or not Path(root).is_dir():
        raise ValueError("Set NUSCENES_ROOT to the extracted nuScenes directory containing v1.0-mini/, samples/, sweeps/.")
    return Path(root)
