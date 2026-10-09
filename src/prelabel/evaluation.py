from collections import defaultdict
import csv
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .dataset import ground_truth, load_manifest
from .detector import load_predictions
from .geometry import geometry_delta, iou3d
from .io import read_json, write_json
from .schema import Cuboid


def match_boxes(predictions, truth, threshold):
    if not 0 < threshold <= 1:
        raise ValueError("Matching IoU threshold must be in (0,1]")
    if any(b.coordinate_frame != "lidar_sensor" for b in [*predictions, *truth]):
        raise ValueError("Evaluation inputs must be transformed to keyframe lidar_sensor coordinates")
    groups_p, groups_t = defaultdict(list), defaultdict(list)
    for box in predictions:
        groups_p[(box.sample_token, box.class_name)].append(box)
    for box in truth:
        groups_t[(box.sample_token, box.class_name)].append(box)
    pairs, extra, missed = [], [], []
    for key in sorted(groups_p.keys() | groups_t.keys()):
        p, t = groups_p[key], groups_t[key]
        if not p:
            missed.extend(t)
            continue
        if not t:
            extra.extend(p)
            continue
        overlaps = np.array([[iou3d(a, b) for b in t] for a in p])
        # Maximum cardinality first, then maximum sum of IoUs among valid pairs.
        utility = np.where(overlaps >= threshold, min(len(p), len(t)) + 1 + overlaps, 0)
        rows, cols = linear_sum_assignment(utility, maximize=True)
        used_p, used_t = set(), set()
        for i, j in zip(rows, cols):
            if overlaps[i, j] >= threshold:
                pairs.append((p[i], t[j], float(overlaps[i, j])))
                used_p.add(int(i))
                used_t.add(int(j))
        extra.extend(b for i, b in enumerate(p) if i not in used_p)
        missed.extend(b for i, b in enumerate(t) if i not in used_t)
    return pairs, extra, missed


def quality_metrics(predictions, truth, threshold, frame_count):
    pairs, extra, missed = match_boxes(predictions, truth, threshold)

    def counts(ps, es, ms):
        tp, fp, fn = len(ps), len(es), len(ms)
        return {"matched_pairs": tp, "extra_boxes": fp, "missed_boxes": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "mean_matched_iou": float(np.mean([v for _, _, v in ps])) if ps else None}

    result = {"frames": frame_count, "prediction_boxes": len(predictions), "ground_truth_boxes": len(truth),
              "match_iou_threshold": threshold, "matching": "maximum_cardinality_then_maximum_total_iou_per_sample_and_class",
              **counts(pairs, extra, missed), "per_class": {}}
    result["extra_prediction_ids"] = [b.prediction_id for b in extra]
    result["missed_ground_truth_ids"] = [b.prediction_id for b in missed]
    for name in sorted({b.class_name for b in [*predictions, *truth]}):
        result["per_class"][name] = counts([pair for pair in pairs if pair[0].class_name == name],
                                            [b for b in extra if b.class_name == name], [b for b in missed if b.class_name == name])
    rows = [{"sample_token": p.sample_token, "frame_id": p.frame_id, "class_name": p.class_name,
             "prediction_id": p.prediction_id, "ground_truth_id": t.prediction_id, "iou3d": overlap} for p, t, overlap in pairs]
    return result, rows


def edit_metrics(baseline, reviewed, settings, identity_lost=False):
    old = {b.prediction_id: b for b in baseline}
    new = {b.prediction_id: b for b in reviewed}
    if len(old) != len(baseline) or len(new) != len(reviewed):
        raise ValueError("Duplicate box identity")
    unresolved = sum(b.identity_source == "unresolved_after_id_loss" for b in reviewed)
    if identity_lost or unresolved:
        return {"status": "unavailable_identity_loss", "initial_predictions": len(old),
                "unresolved_reviewed_boxes": unresolved, "unchanged": None, "edited": None,
                "deleted": None, "added": None,
                "reason": "Server IDs and prediction attributes were lost; geometry matching cannot prove edit identity."}, []
    rows, unchanged, edited = [], 0, 0
    for pid, box in old.items():
        if pid not in new:
            rows.append({"prediction_id": pid, "status": "deleted", "center_m": None, "size_m": None, "rotation_rad": None})
            continue
        other = new[pid]
        delta = geometry_delta(box, other)
        changed = box.class_name != other.class_name or delta["center_m"] > settings["center_tolerance_m"] or delta["size_m"] > settings["size_tolerance_m"] or delta["rotation_rad"] > settings["rotation_tolerance_rad"]
        edited += int(changed)
        unchanged += int(not changed)
        rows.append({"prediction_id": pid, "status": "edited" if changed else "unchanged", **delta})
    deleted, added = len(old.keys() - new.keys()), len(new.keys() - old.keys())
    n = len(old)
    return {"status": "measured", "initial_predictions": n, "unchanged": unchanged, "edited": edited,
            "deleted": deleted, "added": added, "unchanged_ratio": unchanged / n if n else None,
            "edited_ratio": edited / n if n else None, "deleted_ratio": deleted / n if n else None,
            "tolerances": {k: settings[k] for k in ("center_tolerance_m", "size_tolerance_m", "rotation_tolerance_rad")},
            "recovered_ids": sum(b.identity_source == "prediction_attribute_after_id_loss" for b in reviewed)}, rows


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(cfg):
    from .timing import time_metrics
    run = Path(cfg["run_dir"])
    manifest = load_manifest(run)
    before, inference_records = load_predictions(run, manifest)
    reviewed_doc = read_json(run / "reviewed.json")
    if reviewed_doc["manifest_signature"] != manifest["signature"]:
        raise ValueError("Reviewed annotation/manifest provenance mismatch")
    reviewed = [Cuboid.from_dict(b) for b in reviewed_doc["boxes"]]
    baseline_doc = read_json(run / "baseline.json")
    if baseline_doc["manifest_signature"] != manifest["signature"] or baseline_doc["boxes"] != [b.to_dict() for b in before]:
        raise ValueError("Initial prediction baseline was altered")
    frame_lookup = {f["sample_token"]: f["frame_id"] for f in manifest["frames"]}
    for b in [*before, *reviewed]:
        if frame_lookup.get(b.sample_token) != b.frame_id:
            raise ValueError("Annotation has an invalid sample_token/frame_id mapping")
    truth = ground_truth(cfg, manifest)
    threshold = cfg["evaluation"]["match_iou"]
    pre, pre_rows = quality_metrics(before, truth, threshold, len(manifest["frames"]))
    post, post_rows = quality_metrics(reviewed, truth, threshold, len(manifest["frames"]))
    edits, edit_rows = edit_metrics(before, reviewed, cfg["evaluation"], reviewed_doc.get("identity_mode") == "ids_and_attributes_lost")
    preparation = read_json(run / "prepare_timing.json") if (run / "prepare_timing.json").exists() else None
    upload_timing = read_json(run / "upload_timing.json") if (run / "upload_timing.json").exists() else None
    source = "real_nuscenes_and_detector" if all(r.get("source") == "real_detector" for r in inference_records) else "synthetic_or_imported_fixture"
    report = {"data_source": source, "official_nuscenes_map_nds": "not_computed",
              "manifest_signature": manifest["signature"], "ground_truth_policy": {
                  "classes": cfg["detector"]["classes"], "min_lidar_points": cfg["evaluation"]["min_lidar_points"],
                  "box_center_roi": cfg["evaluation"]["point_cloud_range"], "frame": "lidar_sensor"},
              "before": pre, "after": post, "edits": edits,
              "annotation_time": time_metrics(run), "pipeline_time": {
                  "preparation": preparation, "upload": upload_timing,
                  "inference_forward_seconds": sum(r["raw"]["forward_seconds"] for r in inference_records),
                  "inference_frame_seconds": sum(r["frame_seconds"] for r in inference_records),
                  "model_load_included": False}}
    report_dir = run / "reports"
    write_json(report_dir / "metrics.json", report)
    write_csv(report_dir / "matched_pairs.csv", [{"stage": stage, **row} for stage, rows in [("before", pre_rows), ("after", post_rows)] for row in rows],
              ["stage", "sample_token", "frame_id", "class_name", "prediction_id", "ground_truth_id", "iou3d"])
    write_csv(report_dir / "edits.csv", edit_rows, ["prediction_id", "status", "center_m", "size_m", "rotation_rad"])
    write_csv(report_dir / "quality_summary.csv", [{"stage": stage, **{k: metrics[k] for k in ("frames", "prediction_boxes", "ground_truth_boxes", "matched_pairs", "extra_boxes", "missed_boxes", "precision", "recall", "mean_matched_iou", "match_iou_threshold")}} for stage, metrics in [("before", pre), ("after", post)]],
              ["stage", "frames", "prediction_boxes", "ground_truth_boxes", "matched_pairs", "extra_boxes", "missed_boxes", "precision", "recall", "mean_matched_iou", "match_iou_threshold"])
    def display(v):
        return "chưa có dữ liệu" if v is None else f"{v:.4f}"
    text = "# Kết quả thực nghiệm\n\n| Giai đoạn | Frame | GT | Ghép | Thừa | Bỏ sót | Precision | Recall | Mean IoU |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for name, m in [("Trước chỉnh", pre), ("Sau chỉnh", post)]:
        text += f"| {name} | {m['frames']} | {m['ground_truth_boxes']} | {m['matched_pairs']} | {m['extra_boxes']} | {m['missed_boxes']} | {display(m['precision'])} | {display(m['recall'])} | {display(m['mean_matched_iou'])} |\n"
    text += f"\nNgưỡng ghép IoU: {threshold}. Mean IoU chỉ trên cặp ghép. mAP/NDS chính thức chưa tính.\n\n"
    text += "| Chỉnh sửa | Số box |\n|---|---:|\n"
    for name, key in [("Prediction ban đầu", "initial_predictions"), ("Giữ nguyên", "unchanged"), ("Được sửa", "edited"), ("Bị xóa", "deleted"), ("Thêm mới", "added")]:
        value = edits.get(key)
        text += f"| {name} | {value if value is not None else 'chưa xác định do mất identity'} |\n"
    annotation_time = report["annotation_time"]
    text += "\n| Workflow | Giây/frame | Giây/cuboid |\n|---|---:|---:|\n"
    for workflow in ("manual", "assisted"):
        timing = annotation_time.get(workflow) or {}
        text += f"| {workflow} | {display(timing.get('seconds_per_frame'))} | {display(timing.get('seconds_per_cuboid'))} |\n"
    text += f"\nTiết kiệm thời gian (%): {display(annotation_time.get('savings_percent'))}. Chi tiết policy/tolerance và từng nhóm: metrics.json.\n"
    (report_dir / "summary.md").write_text(text, encoding="utf-8")
    return report
