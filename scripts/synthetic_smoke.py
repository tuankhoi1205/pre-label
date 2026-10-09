"""CPU-only geometry smoke example. This never calls a detector or CVAT server."""
from pathlib import Path
import numpy as np
from prelabel.cvat import label_maps, label_spec, shape_payload
from prelabel.dataset import write_pcd
from prelabel.evaluation import edit_metrics, quality_metrics, write_csv
from prelabel.geometry import box_from_cvat, yaw_quaternion
from prelabel.io import write_json
from prelabel.schema import Cuboid


def main():
    output = Path(__file__).resolve().parents[1] / "artifacts" / "synthetic_smoke"
    original = Cuboid("SYNTHETIC_ONLY", 0, "synthetic:0", "car", 0.8, [8, 2, 1], [4.6, 1.7, 1.5], yaw_quaternion(0.7), model="synthetic_geometry_fixture", checkpoint="none")
    rng = np.random.default_rng(7)
    points = np.zeros((200, 5), dtype="<f4")
    points[:, :3] = rng.normal(size=(200, 3)) * [2, 1, 0.5] + original.center_xyz
    points[:, 3] = 80
    write_pcd(output / "synthetic.pcd", points)
    spec = label_spec("car")
    spec.update(id=-1, attributes=[{**a, "id": -(i + 1)} for i, a in enumerate(spec["attributes"])])
    labels = label_maps([spec], ["car"], {})
    payload = shape_payload(original, {"0": 0}, labels, "synthetic")
    metadata = {k: v for k, v in original.to_dict().items() if k not in ("center_xyz", "size_lwh", "quaternion_wxyz")}
    restored = box_from_cvat(payload["points"], **metadata)
    reviewed = restored.updated(center_xyz=[8.1, 2, 1], identity_source="annotation_id", annotation_id=101)
    settings = {"center_tolerance_m": 0.02, "size_tolerance_m": 0.02, "rotation_tolerance_rad": 0.01}
    quality, pairs = quality_metrics([reviewed], [original], 0.5, 1)
    edits, rows = edit_metrics([original], [reviewed], settings)
    write_json(output / "payload.json", {"source": "synthetic_only", "placeholder_ids": True, "shape": payload})
    write_json(output / "metrics.json", {"source": "synthetic_only", "real_inference": False, "live_cvat": False,
                                         "before": original.to_dict(), "reviewed": reviewed.to_dict(), "quality": quality,
                                         "edits": edits, "annotation_time": "not_measured"})
    write_csv(output / "matched_pairs.csv", pairs, ["sample_token", "frame_id", "class_name", "prediction_id", "ground_truth_id", "iou3d"])
    write_csv(output / "edits.csv", rows, ["prediction_id", "status", "center_m", "size_m", "rotation_rad"])
    print(f"Synthetic-only geometry artifacts: {output}")


if __name__ == "__main__":
    main()
