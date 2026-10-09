"""Create a PCD-only task before inference from CVAT's Annotate button."""
import os
from pathlib import Path
from .cvat import CVATClient, frame_mapping, label_maps, label_spec
from .dataset import load_manifest, validate_manifest
from .io import digest, read_json, write_json


def bind_ui_journal(cfg, document, client=None):
    """Bind original CVAT IDs from the persistent UI journal for REST export.

    The journal contains the initial shapes even after human deletion/editing.
    No annotations are written and current shapes never replace the baseline.
    """
    from .cvat import attrs_by_name, shape_payload, validate_created
    from .detector import load_predictions
    run = Path(cfg['run_dir'])
    saved = read_json(run / 'ui_task.json')
    if not saved.get('task_id') or not saved.get('frame_mapping'):
        raise ValueError('Create the UI task and finish Annotate before exporting')
    client = client or CVATClient(cfg['cvat'])
    state = create_task(cfg, client)
    if document['task_id'] != state['task_id']:
        raise ValueError('UI journal belongs to a different task')
    journal = document['journal']
    if journal['version'] != 1 or journal.get('pending') is not None:
        raise ValueError('UI annotation journal is pending or unsupported; inspect before exporting')
    if set(journal['frames']) != {str(v) for v in state['frame_mapping'].values()}:
        raise ValueError('Finish Annotate for every frame before exporting this run')
    manifest = load_manifest(run)
    boxes, _ = load_predictions(run, manifest)
    inference = read_json(run / 'inference_state.json')
    labels = client.labels(state['task_id'])
    mapped = label_maps(labels, cfg['detector']['classes'], cfg['cvat'].get('label_mapping', {}))
    expected = {b.prediction_id: shape_payload(b, state['frame_mapping'], mapped, inference['run_id']) for b in boxes}
    baseline_shapes, applied = [], {}
    for frame, record in journal['frames'].items():
        frame_pids = []
        for shape in record['baseline']:
            pid = attrs_by_name(shape, labels).get('prediction_id')
            if pid not in expected or shape['frame'] != int(frame):
                raise ValueError('UI journal baseline does not match this inference/frame mapping')
            if attrs_by_name(shape, labels) != attrs_by_name(expected[pid], labels):
                raise ValueError('UI journal prediction metadata differs from this run')
            frame_pids.append(pid)
            baseline_shapes.append(shape)
        if set(frame_pids) != set(record['annotation_ids']):
            raise ValueError('UI journal is missing original annotation IDs')
        for pid, annotation_id in record['annotation_ids'].items():
            if pid in applied or not isinstance(annotation_id, int) or annotation_id <= 0:
                raise ValueError('UI journal contains invalid/duplicate original IDs')
            applied[pid] = {'annotation_id': annotation_id, 'recovered': False}
    if len({v['annotation_id'] for v in applied.values()}) != len(applied):
        raise ValueError('UI journal contains duplicate original annotation IDs')
    validate_created(baseline_shapes, expected, labels)
    baseline = {'manifest_signature': manifest['signature'], 'inference_signature': inference['signature'],
                'boxes': [b.to_dict() for b in boxes]}
    baseline_path = run / 'baseline.json'
    if baseline_path.exists() and read_json(baseline_path) != baseline:
        raise ValueError('Immutable prediction baseline differs from this inference')
    export_state = {'source': 'cvat_annotate_button', 'manifest_signature': manifest['signature'],
        'inference_signature': inference['signature'], 'server_url': client.url,
        'organization': state['organization'], 'task_id': state['task_id'], 'task_name': state['name'],
        'frame_mapping': state['frame_mapping'], 'task_url': state['task_url'],
        'applied': applied, 'pending': [], 'phase': 'complete'}
    state_path = run / 'cvat_state.json'
    if state_path.exists() and read_json(state_path) != export_state:
        raise ValueError('Saved export state belongs to a different journal/run')
    write_json(baseline_path, baseline)
    write_json(state_path, export_state)
    write_json(run / 'ui_journal_snapshot.json', document)
    return export_state


def create_task(cfg, client=None):
    run = Path(cfg["run_dir"])
    manifest = load_manifest(run)
    validate_manifest(manifest, run)
    client = client or CVATClient(cfg["cvat"])
    client.check_version()
    path = run / "ui_task.json"
    task_signature = digest({"manifest": manifest["signature"], "detector": cfg["detector"],
        "label_mapping": cfg["cvat"].get("label_mapping", {}), "run_dir": str(run.resolve())})
    state = read_json(path) if path.exists() else {
        "manifest_signature": manifest["signature"], "server_url": client.url,
        "task_signature": task_signature,
        "organization": os.environ.get("CVAT_ORG", ""), "task_id": None,
        "name": f"{cfg['cvat']['task_name']}-ui-{task_signature[:16]}"}
    if (state["manifest_signature"] != manifest["signature"] or state["server_url"] != client.url or
            state["organization"] != os.environ.get("CVAT_ORG", "") or
            state.get("task_signature") != task_signature):
        raise ValueError("UI task belongs to a different run/server/organization/model settings; use a new run_dir")
    write_json(path, state)
    if state["task_id"] is None:
        found = [t for t in client.all_pages("/api/tasks", search=state["name"]) if t["name"] == state["name"]]
        if len(found) > 1:
            raise ValueError("Multiple matching UI tasks; resolve the ambiguity before retrying")
        aliases = cfg["cvat"].get("label_mapping", {})
        task = found[0] if found else client.request("POST", "/api/tasks", json={
            "name": state["name"], "labels": [label_spec(aliases.get(c, c)) for c in cfg["detector"]["classes"]]})
        state["task_id"] = task["id"]
        write_json(path, state)
    task_id = state["task_id"]
    task = client.request("GET", f"/api/tasks/{task_id}")
    if not task.get("size", 0):
        if cfg.get("dataset") and "version" in manifest["selection"]:
            from .config import dataset_root
            from .context import prepare_camera_context
            prepare_camera_context(manifest, dataset_root(cfg), run)
        if not state.get("data_request"):
            pending = client.all_pages("/api/requests", task_id=task_id, action="create")
            if pending:
                state["data_request"] = pending[-1]["id"]
            elif state.get("data_attempted"):
                raise RuntimeError("UI task data upload is uncertain; inspect CVAT Requests before retrying")
            else:
                state["data_attempted"] = True
                write_json(path, state)
                state["data_request"] = client.upload_data(task_id, manifest, run)["rq_id"]
            write_json(path, state)
        client.wait_request(state["data_request"])
        task = client.request("GET", f"/api/tasks/{task_id}")
    if task["dimension"] != "3d":
        raise ValueError("CVAT must recognize the point-cloud task as 3D")
    mapping = frame_mapping(manifest, client.request("GET", f"/api/tasks/{task_id}/data/meta"))
    label_maps(client.labels(task_id), cfg["detector"]["classes"], cfg["cvat"].get("label_mapping", {}))
    if state.get("frame_mapping") and state["frame_mapping"] != mapping:
        raise ValueError("UI task frame mapping changed")
    state["frame_mapping"] = mapping
    state["task_url"] = f"http://localhost:8080/tasks/{task_id}" if client.url in ("http://cvat-server:8080", "http://cvat_server:8080") else f"{client.url}/tasks/{task_id}"
    write_json(path, state)
    return state
