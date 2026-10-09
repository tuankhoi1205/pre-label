"""Prepare nuScenes camera context for exactly the samples in a LiDAR run."""
from pathlib import Path
import shutil

from .io import digest, file_digest, read_json, write_json

CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")


def prepare_camera_context(manifest, dataset_root, run):
    root, run = Path(dataset_root).resolve(), Path(run)
    version = manifest["selection"]["version"]
    tables = root / version
    samples = {r["token"]: r for r in read_json(tables / "sample.json")}
    sample_data = {r["token"]: r for r in read_json(tables / "sample_data.json")}
    calibrations = {r["token"]: r for r in read_json(tables / "calibrated_sensor.json")}
    sensors = {r["token"]: r for r in read_json(tables / "sensor.json")}
    records = []
    # Validate every association before writing any output.
    for frame in manifest["frames"]:
        sample = samples[frame["sample_token"]]
        lidar = sample_data[frame["sample_data_token"]]
        if lidar["sample_token"] != sample["token"]:
            raise ValueError("LiDAR belongs to a different sample")
        camera_data = {sensors[calibrations[r["calibrated_sensor_token"]]["sensor_token"]]["channel"]: r
                       for r in sample_data.values() if r["sample_token"] == sample["token"] and r["is_key_frame"]}
        for channel in CAMERAS:
            camera = camera_data[channel]
            source = (root / camera["filename"]).resolve()
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError(f"Missing camera image for frame {frame['frame_id']}: {channel}")
            name = Path(frame["pcd_file"]).name.replace(".pcd", "_pcd")
            path = f"related_images/{name}/{channel}.jpg"
            records.append({"frame_id": frame["frame_id"], "pcd_name": Path(frame["pcd_file"]).name,
                            "sample_token": sample["token"], "camera": channel,
                            "sample_data_token": camera["token"], "source_file": camera["filename"],
                            "timestamp_us": camera["timestamp"],
                            "lidar_time_delta_us": camera["timestamp"] - frame["timestamp_us"],
                            "sha256": file_digest(source), "context_file": path})
    for record in records:
        target = run / "context" / record["context_file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_digest(target) != record["sha256"]:
                raise ValueError(f"Existing context image differs: {target}")
        else:
            shutil.copyfile(root / record["source_file"], target)
    result = {"manifest_signature": manifest["signature"], "cameras": list(CAMERAS),
              "source": "nuScenes camera keyframes from the same sample; no ground-truth annotations",
              "images": records}
    result["signature"] = digest(result)
    write_json(run / "context" / "manifest.json", result)
    return result
