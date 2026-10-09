import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from .io import write_json


def command_info(command):
    if not shutil.which(command[0]):
        return {"available": False}
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=15, errors="replace")
        return {"available": p.returncode == 0, "exit_code": p.returncode, "output": (p.stdout + p.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__}


def doctor(cfg, check_cvat=False):
    packages = {}
    for package in ("numpy", "scipy", "PyYAML", "requests", "nuscenes-devkit", "torch", "mmcv", "mmengine", "mmdet", "mmdet3d"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    root = cfg["dataset"]["root"]
    result = {"os": platform.platform(), "python": sys.version, "python_executable": sys.executable,
              "packages": packages, "gpu": command_info(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
              "cuda_toolkit": command_info(["nvcc", "--version"]), "docker": command_info(["docker", "version"]),
              "dataset_available": "${" not in root and (Path(root) / cfg["dataset"]["version"] / "sample.json").is_file(),
              "model_config_available": Path(cfg["detector"]["config"]).is_file(),
              "checkpoint_available": Path(cfg["detector"]["checkpoint"]).is_file(),
              "cvat_url_configured": bool(os.environ.get("CVAT_URL")), "cvat": {"status": "not_checked"}}
    if packages["torch"]:
        try:
            import torch
            result["torch_cuda"] = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda}
        except Exception as exc:
            result["torch_cuda"] = {"error": type(exc).__name__}
    if check_cvat:
        from .cvat import CVATClient
        client = CVATClient(cfg["cvat"])
        result["cvat"] = client.check_version()
    write_json(Path(cfg["run_dir"]) / "environment.json", result)
    return result
