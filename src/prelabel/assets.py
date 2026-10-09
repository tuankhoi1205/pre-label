"""Fetch official, versioned model assets; called explicitly by fetch-model."""
from pathlib import Path
import shutil
import urllib.request
import zipfile
from .io import file_digest, write_json

MODEL_PROFILES = {
    "centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py": {
        "url": "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth",
        "sha256_prefix": "9061688e", "reference_memory_gb": 5.2},
    "centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py": {
        "url": "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth",
        "sha256_prefix": "191a3822", "reference_memory_gb": 4.6},
}
REPO_URL = "https://codeload.github.com/open-mmlab/mmdetection3d/zip/refs/tags/v1.4.0"


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out)
    temporary.replace(destination)


def extract_repo(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            parts = Path(member.filename).parts
            if not parts or parts[0] != "mmdetection3d-1.4.0":
                raise ValueError("Unexpected archive root")
            target = destination.joinpath(*parts[1:]).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError("Unsafe archive entry")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are unsupported")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)


def fetch_model(cfg):
    config_path, checkpoint = Path(cfg["detector"]["config"]), Path(cfg["detector"]["checkpoint"])
    if config_path.name not in MODEL_PROFILES:
        raise ValueError("fetch-model only supports audited official CenterPoint profiles")
    profile = MODEL_PROFILES[config_path.name]
    repository = Path(cfg["project_root"]) / "models" / "mmdetection3d"
    expected_config = repository / "configs" / "centerpoint" / config_path.name
    if config_path.resolve() != expected_config.resolve():
        raise ValueError("Use the official config path under models/mmdetection3d/configs/centerpoint")
    assets = repository.parent
    archive = assets / "mmdetection3d-v1.4.0.zip"
    if not config_path.exists():
        if repository.exists():
            raise ValueError("Existing incomplete model repository: inspect it before fetching to avoid overwriting changes")
        if not archive.exists():
            download(REPO_URL, archive)
        extract_repo(archive, repository)
    if not checkpoint.exists():
        download(profile["url"], checkpoint)
    sha = file_digest(checkpoint)
    if not sha.startswith(profile["sha256_prefix"]):
        raise ValueError("Checkpoint SHA-256 does not match the hash prefix in the official filename")
    record = {"repository_tag": "v1.4.0", "repository_url": REPO_URL, "checkpoint_url": profile["url"],
              "config_path": str(config_path), "config_sha256": file_digest(config_path), "checkpoint_sha256": sha,
              "official_hash_prefix": profile["sha256_prefix"], "sweeps": "1 keyframe + up to 9 previous sweeps", **profile}
    write_json(checkpoint.with_suffix(".provenance.json"), record)
    return record
