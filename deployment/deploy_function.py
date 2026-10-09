"""Generate a Nuclio configuration for the prepared run; no credentials needed."""
import argparse
import json
from pathlib import Path
import yaml
from prelabel.config import load_config
from prelabel.cvat import label_spec
from prelabel.dataset import load_manifest


def generate(config, run_dir, mount_source=None, allow_unprepared=False):
    cfg = load_config(config)
    root = Path(cfg["project_root"])
    run = (root / run_dir).resolve()
    run_relative = run.relative_to(root).as_posix()
    config_relative = Path(config).resolve().relative_to(root).as_posix()
    if not allow_unprepared:
        load_manifest(run)
    for name in ("config", "checkpoint"):
        if not Path(cfg["detector"][name]).is_file():
            raise ValueError(f"Missing detector {name}; run fetch-model first")
        Path(cfg["detector"][name]).resolve().relative_to(root)
    function = yaml.safe_load((root / "deployment/nuclio/function-gpu.yaml").read_text())
    function["metadata"]["annotations"]["spec"] = json.dumps([label_spec(c) for c in cfg["detector"]["classes"]])
    function["spec"]["env"] = [
        {"name": "NUSCENES_ROOT", "value": "/workspace/data/nuscenes"},
        {"name": "PRELABEL_CONFIG", "value": f"/workspace/{config_relative}"},
        {"name": "PRELABEL_RUN_DIR", "value": f"/workspace/{run_relative}"},
        {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
        {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"},
    ]
    function["spec"]["volumes"] = [{
        "volume": {"name": "prelabel-workspace", "hostPath": {"path": mount_source or str(root)}},
        "volumeMount": {"name": "prelabel-workspace", "mountPath": "/workspace"},
    }]
    output = root / ".local/function-gpu.generated.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(function, sort_keys=False), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--run-dir", default="runs/ui-centerpoint")
    parser.add_argument("--mount-source")
    parser.add_argument("--allow-unprepared", action="store_true", help="Register the model while dataset preparation is pending")
    args = parser.parse_args()
    print(generate(args.config, args.run_dir, args.mount_source, args.allow_unprepared))
