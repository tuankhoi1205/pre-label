import argparse
import json
import logging
from pathlib import Path
import sys
from .config import load_config
from .io import read_json, run_lock, write_json


def build_parser():
    parser = argparse.ArgumentParser(description="nuScenes -> CenterPoint -> CVAT 3D -> review -> evaluate")
    parser.add_argument("--config", default="configs/mini.yaml", help="YAML configuration path")
    parser.add_argument("--run-dir", help="Override output directory, relative to project_root")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect environment; credentials are never printed")
    doctor.add_argument("--check-cvat", action="store_true")
    commands.add_parser("fetch-model", help="Download the pinned official config repository and checkpoint")
    prep = commands.add_parser("prepare", help="Prepare CVAT PCD files and an immutable frame manifest")
    prep.add_argument("--limit", type=int, help="Use 1 for the initial smoke run; use a separate run_dir for batch")
    commands.add_parser("infer", help="Real CenterPoint inference with resumable per-frame records")
    commands.add_parser("create-task", help="Create a LiDAR/camera-context CVAT task; run inference from the Annotate button")
    up = commands.add_parser("upload", help="Create/reuse CVAT task and append editable pre-label cuboids")
    up.add_argument("--dry-run", action="store_true", help="Offline payload preview with placeholder IDs; no network access")
    up.add_argument("--check-server", action="store_true", help="For dry-run: check real task metadata/IDs using read-only API calls")
    exp = commands.add_parser("export", help="Read reviewed cuboids through REST and preserve sample mapping")
    exp.add_argument("--ui-journal", help="Local CVAT journal snapshot from deployment/export-ui.ps1")
    exp.add_argument("--ids-recreated", action="store_true", help="After export/import recreated server IDs: trust prediction attributes only")
    commands.add_parser("evaluate", help="Evaluate predictions and reviewed boxes against real nuScenes GT")
    commands.add_parser("report-time", help="Report active annotation sessions without requiring the dataset")
    timer = commands.add_parser("timer", help="Record active work; pause during breaks or waiting")
    timer.add_argument("action", choices=["start", "pause", "resume", "stop"])
    timer.add_argument("--session", required=True)
    timer.add_argument("--workflow", choices=["manual", "assisted"])
    timer.add_argument("--participant")
    timer.add_argument("--difficulty")
    timer.add_argument("--quality-standard")
    timer.add_argument("--frame-ids", help="Comma-separated manifest frame IDs")
    timer.add_argument("--cuboids", type=int, help="Number of final accepted cuboids; can be supplied at stop")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        cfg = load_config(args.config)
        if args.run_dir:
            override = Path(args.run_dir).expanduser()
            cfg["run_dir"] = str(override if override.is_absolute() else Path(cfg["project_root"]) / override)
        run = Path(cfg["run_dir"])
        with run_lock(run):
            if args.command == "doctor":
                from .doctor import doctor
                result = doctor(cfg, args.check_cvat)
            elif args.command == "fetch-model":
                from .assets import fetch_model
                result = fetch_model(cfg)
            elif args.command == "prepare":
                from .dataset import prepare
                manifest = prepare(cfg, args.limit)
                result = {"frames": len(manifest["frames"]), "manifest": str(run / "manifest.json")}
            elif args.command == "infer":
                from .detector import infer
                boxes = infer(cfg)
                result = {"boxes": len(boxes), "predictions": str(run / "predictions")}
            elif args.command == "create-task":
                from .ui_task import create_task
                state = create_task(cfg)
                result = {"task_url": state["task_url"], "task_id": state["task_id"],
                          "next": "Automatic annotation -> CenterPoint 3D -> Annotate"}
            elif args.command == "upload":
                from .cvat import CVATClient, dry_run, upload
                if args.check_server and not args.dry_run:
                    raise ValueError("--check-server applies only to --dry-run")
                if args.dry_run:
                    payload = dry_run(cfg, CVATClient(cfg["cvat"]) if args.check_server else None)
                    result = {"shapes": len(payload["shapes"]), "payload": str(run / "dry_run.json"), "server_checked": args.check_server}
                else:
                    state = upload(cfg)
                    result = {"task_url": state["task_url"], "initial_predictions": len(state["applied"])}
            elif args.command == "export":
                from .cvat import export_annotations
                if args.ui_journal:
                    from .ui_task import bind_ui_journal
                    bind_ui_journal(cfg, read_json(Path(args.ui_journal)))
                boxes = export_annotations(cfg, ids_recreated=args.ids_recreated)
                result = {"boxes": len(boxes), "reviewed": str(run / "reviewed.json")}
            elif args.command == "evaluate":
                from .evaluation import evaluate
                report = evaluate(cfg)
                result = {"before": report["before"], "after": report["after"], "edits": report["edits"], "report_dir": str(run / "reports")}
            elif args.command == "report-time":
                from .timing import time_metrics
                result = time_metrics(run)
                write_json(run / "reports" / "annotation_time_summary.json", result)
            elif args.command == "timer":
                from .timing import timer
                metadata = None
                if args.action == "start":
                    required = (args.workflow, args.participant, args.difficulty, args.quality_standard, args.frame_ids)
                    if not all(required):
                        raise ValueError("timer start requires --workflow, --participant, --difficulty, --quality-standard and --frame-ids")
                    ids = [int(v) for v in args.frame_ids.split(",")]
                    from .dataset import load_manifest
                    manifest = load_manifest(run)
                    if len(ids) != len(set(ids)) or any(v < 0 or v >= len(manifest["frames"]) for v in ids):
                        raise ValueError("Timer frame IDs must be unique entries in this run's manifest")
                    metadata = {"workflow": args.workflow, "participant": args.participant, "difficulty": args.difficulty,
                                "quality_standard": args.quality_standard, "frame_ids": ids, "cuboids": args.cuboids or 0}
                elif args.action == "stop" and args.cuboids is not None:
                    if args.cuboids < 0:
                        raise ValueError("Cuboid count cannot be negative")
                    metadata = {"cuboids": args.cuboids}
                result = timer(run, args.action, args.session, metadata)
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except KeyError as exc:
        print(f"ERROR: Missing required configuration/data field: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except (ValueError, RuntimeError, FileNotFoundError, ImportError, TimeoutError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
