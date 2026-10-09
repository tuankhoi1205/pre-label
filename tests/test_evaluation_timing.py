import pytest
from prelabel.evaluation import edit_metrics, match_boxes, quality_metrics
from prelabel.geometry import yaw_quaternion
from prelabel.timing import active_seconds, time_metrics, timer


SETTINGS = {"center_tolerance_m": 0.02, "size_tolerance_m": 0.02, "rotation_tolerance_rad": 0.01}


def test_matching_is_one_to_one_same_sample_class(box):
    predictions = [box, box.updated(prediction_id="duplicate"), box.updated(prediction_id="bicycle", class_name="bicycle")]
    truth = [box.updated(prediction_id="gt"), box.updated(sample_token="another", frame_id=1, prediction_id="missed")]
    pairs, extra, missed = match_boxes(predictions, truth, 0.5)
    assert (len(pairs), len(extra), len(missed)) == (1, 2, 1)
    metrics, _ = quality_metrics(predictions, truth, 0.5, 2)
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["mean_matched_iou"] == pytest.approx(1)


def test_assignment_maximizes_cardinality_before_iou(box, monkeypatch):
    # Greedy highest IoU would give only one match; assignment must return two.
    values = {("p0", "g0"): 0.99, ("p0", "g1"): 0.51, ("p1", "g0"): 0.51, ("p1", "g1"): 0.1}
    monkeypatch.setattr("prelabel.evaluation.iou3d", lambda p, g: values[p.prediction_id, g.prediction_id])
    ps = [box.updated(prediction_id=f"p{i}") for i in range(2)]
    gs = [box.updated(prediction_id=f"g{i}") for i in range(2)]
    pairs, extra, missed = match_boxes(ps, gs, 0.5)
    assert len(pairs) == 2
    assert not extra and not missed


def test_no_matches_does_not_report_perfect_iou(box):
    result, rows = quality_metrics([box], [box.updated(center_xyz=[100, 100, 100])], 0.5, 1)
    assert result["mean_matched_iou"] is None
    assert result["precision"] == result["recall"] == 0
    empty, _ = quality_metrics([], [], 0.5, 1)
    assert empty["precision"] is None and empty["recall"] is None


def test_edit_ratios_added_deleted_tolerances(box):
    baseline = [box.updated(prediction_id=f"p{i}") for i in range(4)]
    reviewed = [baseline[0].updated(center_xyz=[4.001, 2, 1.5]),
                baseline[1].updated(class_name="truck"), baseline[2].updated(quaternion_wxyz=yaw_quaternion(0.02)),
                box.updated(prediction_id="manual:1", identity_source="new_annotation_id")]
    metrics, rows = edit_metrics(baseline, reviewed, SETTINGS)
    assert (metrics["unchanged"], metrics["edited"], metrics["deleted"], metrics["added"]) == (1, 2, 1, 1)
    assert metrics["unchanged_ratio"] + metrics["edited_ratio"] + metrics["deleted_ratio"] == pytest.approx(1)


def test_lost_identity_does_not_claim_deletions(box):
    result, _ = edit_metrics([box], [box.updated(prediction_id="unknown", identity_source="unresolved_after_id_loss")], SETTINGS)
    assert result["status"] == "unavailable_identity_loss"
    assert result["deleted"] is None
    assert result["added"] is None


def session_meta(workflow, frames):
    return {"workflow": workflow, "participant": "tester", "difficulty": "moderate", "quality_standard": "approved-cuboids", "frame_ids": frames, "cuboids": 4}


def test_timer_excludes_breaks_and_time_savings(tmp_path):
    timer(tmp_path, "start", "manual", session_meta("manual", [0, 1]), stamp="2026-10-08T01:00:00+00:00")
    timer(tmp_path, "pause", "manual", stamp="2026-10-08T01:01:00+00:00")
    timer(tmp_path, "resume", "manual", stamp="2026-10-08T01:10:00+00:00")
    stopped = timer(tmp_path, "stop", "manual", stamp="2026-10-08T01:11:00+00:00")
    assert active_seconds(stopped) == 120  # Nine-minute break excluded.
    timer(tmp_path, "start", "assisted", session_meta("assisted", [2, 3]), stamp="2026-10-08T02:00:00+00:00")
    timer(tmp_path, "stop", "assisted", stamp="2026-10-08T02:01:00+00:00")
    result = time_metrics(tmp_path)
    assert result["manual"]["seconds_per_frame"] == 60
    assert result["manual"]["seconds_per_cuboid"] == 30
    assert result["savings_percent"] == 50


def test_time_experiment_frame_overlap_is_not_reported_as_savings(tmp_path):
    for i, workflow in enumerate(("manual", "assisted")):
        timer(tmp_path, "start", workflow, session_meta(workflow, [0]), stamp=f"2026-10-08T0{i}:00:00+00:00")
        timer(tmp_path, "stop", workflow, stamp=f"2026-10-08T0{i}:01:00+00:00")
    assert time_metrics(tmp_path)["savings_percent"] is None


def test_no_annotation_time_is_explicitly_unmeasured(tmp_path):
    result = time_metrics(tmp_path)
    assert result["status"] == "not_measured"
    assert result["savings_percent"] is None


def test_no_two_active_timers(tmp_path):
    timer(tmp_path, "start", "a", session_meta("manual", [0]))
    with pytest.raises(ValueError, match="active"):
        timer(tmp_path, "start", "b", session_meta("assisted", [1]))


def test_frame_cannot_be_counted_twice_in_same_workflow(tmp_path):
    timer(tmp_path, "start", "a", session_meta("manual", [0]), stamp="2026-10-08T01:00:00+00:00")
    timer(tmp_path, "stop", "a", stamp="2026-10-08T01:01:00+00:00")
    with pytest.raises(ValueError, match="already timed"):
        timer(tmp_path, "start", "b", session_meta("manual", [0]))


def test_backward_timer_interval_rejected(tmp_path):
    timer(tmp_path, "start", "a", session_meta("manual", [0]), stamp="2026-10-08T01:00:00+00:00")
    with pytest.raises(ValueError, match="chronological"):
        timer(tmp_path, "stop", "a", stamp="2026-10-08T00:59:00+00:00")
