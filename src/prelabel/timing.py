from datetime import datetime, timezone
from pathlib import Path
from .io import read_json, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def active_seconds(session):
    seconds, last_end = 0.0, None
    for start, end in session["intervals"]:
        a, b = datetime.fromisoformat(start), datetime.fromisoformat(end)
        if a.tzinfo is None or b.tzinfo is None or b < a or (last_end is not None and a < last_end):
            raise ValueError("Timer intervals must have a timezone and be chronological")
        seconds += (b - a).total_seconds()
        last_end = b
    return seconds


def timer(run, action, session_id, metadata=None, stamp=None):
    path = Path(run) / "annotation_time.json"
    doc = read_json(path) if path.exists() else {"sessions": {}}
    sessions = doc["sessions"]
    stamp = stamp or now()
    if action == "start":
        if session_id in sessions:
            raise ValueError("Session ID already exists")
        if any(s["state"] == "active" for s in sessions.values()):
            raise ValueError("Pause/stop the active timer before starting another session")
        if not metadata or metadata["workflow"] not in ("manual", "assisted") or metadata["cuboids"] < 0 or not metadata["frame_ids"]:
            raise ValueError("Timer requires workflow, positive frame count and nonnegative cuboid count")
        for other in sessions.values():
            if (other["participant"], other["workflow"]) == (metadata["participant"], metadata["workflow"]):
                if set(other["frame_ids"]) & set(metadata["frame_ids"]):
                    raise ValueError("Frames already timed for this participant/workflow; pause/resume one session until review is complete")
        sessions[session_id] = {**metadata, "state": "active", "active_start": stamp, "intervals": []}
    else:
        if session_id not in sessions:
            raise ValueError("Unknown timer session")
        s = sessions[session_id]
        if action == "pause":
            if s["state"] != "active":
                raise ValueError("Only an active session can be paused")
            s["intervals"].append([s["active_start"], stamp])
            s.update(state="paused", active_start=None)
        elif action == "resume":
            if s["state"] != "paused" or any(v["state"] == "active" for v in sessions.values()):
                raise ValueError("Resume requires a paused session and no other active timer")
            s.update(state="active", active_start=stamp)
        elif action == "stop":
            if s["state"] == "stopped":
                raise ValueError("Session already stopped")
            if s["state"] == "active":
                s["intervals"].append([s["active_start"], stamp])
            s.update(state="stopped", active_start=None)
            if metadata and metadata.get("cuboids") is not None:
                s["cuboids"] = metadata["cuboids"]
        else:
            raise ValueError("Unknown timer action")
    if any(active_seconds(s) < 0 for s in sessions.values()):
        raise ValueError("Timer end predates start")
    write_json(path, doc)
    return sessions[session_id]


def time_metrics(run):
    from .evaluation import write_csv
    path = Path(run) / "annotation_time.json"
    if not path.exists():
        return {"status": "not_measured", "manual": None, "assisted": None, "savings_percent": None}
    sessions = read_json(path)["sessions"]
    rows = []
    for session_id, s in sessions.items():
        if s["state"] != "stopped":
            continue
        seconds = active_seconds(s)
        rows.append({"session_id": session_id, "workflow": s["workflow"], "participant": s["participant"],
                     "difficulty": s["difficulty"], "quality_standard": s["quality_standard"],
                     "frame_ids": ",".join(map(str, s["frame_ids"])), "frames": len(s["frame_ids"]), "cuboids": s["cuboids"],
                     "active_seconds": seconds, "seconds_per_frame": seconds / len(s["frame_ids"]),
                     "seconds_per_cuboid": seconds / s["cuboids"] if s["cuboids"] else None})
    write_csv(Path(run) / "reports" / "annotation_sessions.csv", rows, ["session_id", "workflow", "participant", "difficulty", "quality_standard", "frame_ids", "frames", "cuboids", "active_seconds", "seconds_per_frame", "seconds_per_cuboid"])
    groups = {}
    for workflow in ("manual", "assisted"):
        selected = [s for s in rows if s["workflow"] == workflow]
        seconds = sum(s["active_seconds"] for s in selected)
        frames = sum(s["frames"] for s in selected)
        cuboids = sum(s["cuboids"] for s in selected)
        groups[workflow] = {"sessions": len(selected), "frames": frames, "cuboids": cuboids, "active_seconds": seconds,
                            "seconds_per_frame": seconds / frames if frames else None,
                            "seconds_per_cuboid": seconds / cuboids if cuboids else None}
    # Only estimate savings within participant/difficulty/quality strata, on disjoint frames.
    strata = sorted({(s["participant"], s["difficulty"], s["quality_standard"]) for s in rows})
    comparisons = []
    for participant, difficulty, quality in strata:
        selected = [s for s in rows if (s["participant"], s["difficulty"], s["quality_standard"]) == (participant, difficulty, quality)]
        manual, assisted = [[s for s in selected if s["workflow"] == w] for w in ("manual", "assisted")]
        if not manual or not assisted:
            continue
        ids_m = {v for s in manual for v in s["frame_ids"].split(",")}
        ids_a = {v for s in assisted for v in s["frame_ids"].split(",")}
        m = sum(s["active_seconds"] for s in manual) / sum(s["frames"] for s in manual)
        a = sum(s["active_seconds"] for s in assisted) / sum(s["frames"] for s in assisted)
        overlap = bool(ids_m & ids_a)
        comparisons.append({"participant": participant, "difficulty": difficulty, "quality_standard": quality,
                            "manual_seconds_per_frame": m, "assisted_seconds_per_frame": a,
                            "overlapping_frames": overlap, "savings_percent": (m - a) / m * 100 if m > 0 and not overlap else None})
    return {"status": "measured" if rows else "not_measured", **groups, "comparisons": comparisons,
            "savings_percent": comparisons[0]["savings_percent"] if len(comparisons) == 1 else None,
            "formula": "100 * (manual_seconds_per_frame - assisted_seconds_per_frame) / manual_seconds_per_frame",
            "unfinished_sessions": sum(s["state"] != "stopped" for s in sessions.values()),
            "limits": "Difficulty and quality strata are supplied by the experimenter, not independently verified."}
