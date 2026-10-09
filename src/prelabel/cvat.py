"""CVAT 2.20 REST adapter. Never replaces a task's complete annotation set."""
from collections import Counter
from contextlib import ExitStack
import logging
import os
from pathlib import Path
import time
import zipfile
from urllib.parse import quote, urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .dataset import load_manifest, validate_manifest
from .detector import load_predictions
from .geometry import box_from_cvat, cvat_points, geometry_delta
from .io import digest, file_digest, read_json, write_json
from .schema import Cuboid

LOG = logging.getLogger(__name__)
ATTRIBUTES = ("prediction_id", "confidence", "model", "checkpoint", "run_id")


def label_spec(name):
    return {"name": name, "type": "cuboid", "attributes": [
        {"name": a, "input_type": "text", "mutable": False, "default_value": "", "values": []} for a in ATTRIBUTES]}


class CVATClient:
    def __init__(self, cfg, session=None):
        self.cfg = cfg
        self.url = os.environ.get("CVAT_URL", "").rstrip("/")
        parsed = urlparse(self.url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("Set CVAT_URL to an http(s) server URL without embedded credentials")
        self.session = session or requests.Session()
        if session is None:
            retry = Retry(total=3, backoff_factor=0.5, allowed_methods=["GET"], status_forcelist=[502, 503, 504])
            self.session.mount("https://", HTTPAdapter(max_retries=retry))
            self.session.mount("http://", HTTPAdapter(max_retries=retry))
        org = os.environ.get("CVAT_ORG")
        if org:
            self.session.headers["X-Organization"] = org
        token = os.environ.get("CVAT_TOKEN")
        if token:
            scheme = os.environ.get("CVAT_AUTH_SCHEME", "Token")
            if scheme not in ("Token", "Bearer"):
                raise ValueError("CVAT_AUTH_SCHEME must be Token or Bearer")
            self.session.headers["Authorization"] = f"{scheme} {token}"
        else:
            username, password = os.environ.get("CVAT_USERNAME"), os.environ.get("CVAT_PASSWORD")
            if not username or not password:
                raise ValueError("Set CVAT_TOKEN or both CVAT_USERNAME and CVAT_PASSWORD")
            response = self.request("POST", "/api/auth/login", json={"username": username, "password": password})
            self.session.headers["Authorization"] = f"Token {response['key']}"

    def request(self, method, path, **kwargs):
        response = self.session.request(method, self.url + path, timeout=self.cfg["timeout_seconds"], **kwargs)
        if not 200 <= response.status_code < 300:
            # Deliberately avoid echoing request bodies, URLs with query strings or auth errors.
            raise RuntimeError(f"CVAT {method} {path.split('?')[0]} returned HTTP {response.status_code}")
        return response.json() if response.content else None

    def check_version(self):
        about = self.request("GET", "/api/server/about")
        version = about["version"]
        expected = self.cfg["expected_version"]
        if expected != "2.20.0" or version != expected:
            raise ValueError(f"CVAT server {version}, adapter audited for 2.20.0. Audit source and tests before extending supported versions.")
        return about

    def all_pages(self, path, **params):
        result, page = [], 1
        while True:
            data = self.request("GET", path, params={**params, "page": page, "page_size": 100})
            if isinstance(data, list):
                return data
            result.extend(data["results"])
            if not data.get("next"):
                return result
            page += 1

    def labels(self, task_id):
        return self.all_pages("/api/labels", task_id=task_id)

    def annotations(self, task_id):
        return self.request("GET", f"/api/tasks/{task_id}/annotations")

    def wait_request(self, rq_id):
        deadline = time.monotonic() + self.cfg["timeout_seconds"]
        while time.monotonic() < deadline:
            status = self.request("GET", f"/api/requests/{quote(rq_id, safe='')}")
            if status["status"] == "finished":
                return
            if status["status"] == "failed":
                raise RuntimeError(f"CVAT background request failed: {status.get('message', '')}")
            time.sleep(self.cfg["poll_seconds"])
        raise TimeoutError("CVAT data processing timed out; rerun to resume polling the saved request")

    def upload_data(self, task_id, manifest, run):
        with ExitStack() as stack:
            files = []
            context_manifest = Path(run) / "context" / "manifest.json"
            if context_manifest.exists():
                context = read_json(context_manifest)
                if context["manifest_signature"] != manifest["signature"]:
                    raise ValueError("Camera context belongs to a different LiDAR run")
                archive_path = Path(run) / "context" / "cvat-input.zip"
                with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
                    for frame in manifest["frames"]:
                        archive.write(Path(run) / frame["pcd_file"], f"pcd/{Path(frame['pcd_file']).name}")
                    for record in context["images"]:
                        path = Path(run) / "context" / record["context_file"]
                        if file_digest(path) != record["sha256"]:
                            raise ValueError("Camera context image changed since preparation")
                        archive.write(path, record["context_file"])
                files.append(("client_files[0]", ("cvat-input.zip", stack.enter_context(archive_path.open("rb")), "application/zip")))
            else:
                for i, frame in enumerate(manifest["frames"]):
                    path = Path(run) / frame["pcd_file"]
                    files.append((f"client_files[{i}]", (path.name, stack.enter_context(path.open("rb")), "application/octet-stream")))
            return self.request("POST", f"/api/tasks/{task_id}/data", data={"image_quality": 100,
                                "sorting_method": "lexicographical", "use_cache": "false"}, files=files)


def frame_mapping(manifest, metadata):
    """Use actual metadata names, independent of sorting and frame_step assumptions."""
    if metadata.get("deleted_frames"):
        raise ValueError("Task has deleted frames; restore them before mapping/export")
    start = metadata.get("start_frame", 0)
    if start != 0 or metadata.get("frame_filter", "") not in ("", None):
        raise ValueError("MVP requires a contiguous CVAT task beginning at frame 0")
    actual = metadata["frames"]
    if len(actual) != len(manifest["frames"]):
        raise ValueError("CVAT frame count differs from manifest")
    expected = {Path(f["pcd_file"]).name: f for f in manifest["frames"]}
    mapping = {}
    seen = set()
    for cvat_frame, record in enumerate(actual):
        name = record["name"].replace("\\", "/").split("/")[-1]
        if name in seen or name not in expected:
            raise ValueError(f"Unexpected/duplicate CVAT frame name: {name}")
        seen.add(name)
        mapping[str(expected[name]["frame_id"])] = cvat_frame
    if seen != set(expected):
        raise ValueError("CVAT dropped a point cloud")
    return mapping


def label_maps(labels, classes, aliases):
    names = [aliases.get(c, c) for c in classes]
    if len(names) != len(set(names)):
        raise ValueError("label_mapping must be one-to-one")
    by_name = {label["name"]: label for label in labels}
    mapped = {}
    for cls, name in zip(classes, names):
        if name not in by_name:
            raise ValueError(f"Task lacks label {name}; create it with the metadata attributes from the dry-run payload")
        label = by_name[name]
        if label.get("type") not in ("cuboid", "any"):
            raise ValueError(f"Label {name} cannot hold cuboids")
        attrs = {a["name"]: a["id"] for a in label["attributes"]}
        missing = set(ATTRIBUTES) - attrs.keys()
        if missing:
            raise ValueError(f"Label {name} lacks {sorted(missing)}; add text attributes or use a new task")
        mapped[cls] = {"label_id": label["id"], "attributes": attrs}
    return mapped


def shape_payload(box, mapping, labels, run_id):
    label = labels[box.class_name]
    values = {"prediction_id": box.prediction_id, "confidence": str(box.confidence), "model": box.model,
              "checkpoint": box.checkpoint, "run_id": run_id}
    return {"type": "cuboid", "frame": mapping[str(box.frame_id)], "label_id": label["label_id"],
            "points": cvat_points(box), "rotation": 0, "occluded": False, "outside": False,
            "source": "auto", "group": 0, "z_order": 0,
            "attributes": [{"spec_id": label["attributes"][k], "value": v} for k, v in values.items()]}


def attrs_by_name(shape, labels):
    spec_names = {a["id"]: a["name"] for l in labels for a in l["attributes"]}
    return {spec_names[a["spec_id"]]: a["value"] for a in shape.get("attributes", []) if a["spec_id"] in spec_names}


def validate_created(shapes, expected, labels):
    ids = set()
    for shape in shapes:
        attrs = attrs_by_name(shape, labels)
        pid = attrs.get("prediction_id")
        if pid not in expected:
            continue
        if pid in ids:
            raise ValueError(f"CVAT has duplicate prediction_id: {pid}")
        ids.add(pid)
        wanted = expected[pid]
        if shape["type"] != "cuboid" or shape["frame"] != wanted["frame"] or shape["label_id"] != wanted["label_id"]:
            raise ValueError("CVAT readback label/frame/type mismatch")
        meta = {"sample_token": "check", "frame_id": 0, "prediction_id": pid, "class_name": "check", "confidence": None}
        a, b = box_from_cvat(shape["points"], **meta), box_from_cvat(wanted["points"], **meta)
        if any(v > 1e-4 for v in geometry_delta(a, b).values()):
            raise ValueError("CVAT readback geometry mismatch; snapshot retained for inspection")
    if ids != set(expected):
        raise ValueError(f"CVAT readback missing {len(set(expected) - ids)} created shapes")


def dry_run(cfg, client=None):
    run = Path(cfg["run_dir"])
    manifest = load_manifest(run)
    validate_manifest(manifest, run)
    boxes, _ = load_predictions(run, manifest)
    run_id = read_json(run / "inference_state.json")["run_id"]
    aliases = cfg["cvat"].get("label_mapping", {})
    specs = [label_spec(aliases.get(cls, cls)) for cls in cfg["detector"]["classes"]]
    mapping = {str(f["frame_id"]): f["frame_id"] for f in manifest["frames"]}
    if client is not None:
        client.check_version()
        task_id = cfg["cvat"].get("task_id") or read_json(run / "cvat_state.json")["task_id"]
        task = client.request("GET", f"/api/tasks/{task_id}")
        if task["dimension"] != "3d":
            raise ValueError("Target task is not 3D")
        mapping = frame_mapping(manifest, client.request("GET", f"/api/tasks/{task_id}/data/meta"))
        labels = client.labels(task_id)
    else:
        # Negative IDs are explicit placeholders, never valid server IDs.
        labels = [{**s, "id": -(i + 1), "attributes": [{**a, "id": -(i * len(ATTRIBUTES) + j + 1)} for j, a in enumerate(s["attributes"])]} for i, s in enumerate(specs)]
    mapped = label_maps(labels, cfg["detector"]["classes"], aliases)
    payload = {"version": 0, "tags": [], "tracks": [], "shapes": [shape_payload(b, mapping, mapped, run_id) for b in boxes]}
    write_json(run / "dry_run.json", {"offline_placeholder_ids": client is None, "manifest_signature": manifest["signature"],
                                       "labels": specs, "frame_mapping": mapping, "annotations": payload})
    return payload


def upload(cfg, client=None):
    start = time.perf_counter()
    run = Path(cfg["run_dir"])
    manifest = load_manifest(run)
    validate_manifest(manifest, run)
    boxes, _ = load_predictions(run, manifest)
    inference = read_json(run / "inference_state.json")
    run_id = inference["run_id"]
    client = client or CVATClient(cfg["cvat"])
    client.check_version()
    state_path = run / "cvat_state.json"
    configured_task = cfg["cvat"].get("task_id")
    state = read_json(state_path) if state_path.exists() else {
        "manifest_signature": manifest["signature"], "inference_signature": inference["signature"],
        "server_url": client.url, "organization": os.environ.get("CVAT_ORG", ""),
        "task_name": f"{cfg['cvat']['task_name']}-{run_id}", "task_id": configured_task,
        "phase": "initial", "applied": {}, "pending": []}
    if state["manifest_signature"] != manifest["signature"] or state["inference_signature"] != inference["signature"]:
        raise ValueError("CVAT state belongs to another run")
    if state["server_url"] != client.url or state["organization"] != os.environ.get("CVAT_ORG", ""):
        raise ValueError("CVAT server/organization changed; use a new run")
    if configured_task is not None and state["task_id"] != configured_task:
        raise ValueError("task_id differs from saved upload state")
    write_json(state_path, state)
    if state["task_id"] is None:
        found = [t for t in client.all_pages("/api/tasks", search=state["task_name"]) if t["name"] == state["task_name"]]
        if len(found) > 1:
            raise ValueError("Multiple tasks share the run name; set task_id explicitly")
        if found:
            state["task_id"] = found[0]["id"]
        else:
            aliases = cfg["cvat"].get("label_mapping", {})
            task = client.request("POST", "/api/tasks", json={"name": state["task_name"],
                                  "labels": [label_spec(aliases.get(c, c)) for c in cfg["detector"]["classes"]]})
            state["task_id"] = task["id"]
        state["phase"] = "task_created"
        write_json(state_path, state)
    task_id = state["task_id"]
    task = client.request("GET", f"/api/tasks/{task_id}")
    if not task.get("size", 0):
        if not state.get("data_request"):
            # Recover after a lost HTTP response without submitting the files twice.
            pending = client.all_pages("/api/requests", task_id=task_id, action="create")
            if pending:
                state["data_request"] = pending[-1]["id"]
            elif state.get("data_attempted"):
                raise RuntimeError("Data upload outcome is uncertain. Inspect CVAT Requests/task before clearing data_attempted in the journal.")
            else:
                state["data_attempted"] = True
                write_json(state_path, state)
                result = client.upload_data(task_id, manifest, run)
                state["data_request"] = result["rq_id"]
            write_json(state_path, state)
        client.wait_request(state["data_request"])
        task = client.request("GET", f"/api/tasks/{task_id}")
    if task["dimension"] != "3d":
        raise ValueError("PCD task was not recognized as 3D")
    mapping = frame_mapping(manifest, client.request("GET", f"/api/tasks/{task_id}/data/meta"))
    if state.get("frame_mapping") and state["frame_mapping"] != mapping:
        raise ValueError("CVAT frame mapping changed since upload")
    state["frame_mapping"] = mapping
    labels = client.labels(task_id)
    mapped = label_maps(labels, cfg["detector"]["classes"], cfg["cvat"].get("label_mapping", {}))
    desired = {b.prediction_id: shape_payload(b, mapping, mapped, run_id) for b in boxes}
    baseline_path = run / "baseline.json"
    if not baseline_path.exists():
        write_json(baseline_path, {"manifest_signature": manifest["signature"], "inference_signature": inference["signature"],
                                   "boxes": [b.to_dict() for b in boxes]})
    elif read_json(baseline_path)["boxes"] != [b.to_dict() for b in boxes]:
        raise ValueError("Immutable prediction baseline differs from this inference")
    current = client.annotations(task_id)
    by_pid = {}
    for shape in current["shapes"]:
        pid = attrs_by_name(shape, labels).get("prediction_id")
        if pid in desired:
            if pid in by_pid:
                raise ValueError(f"Duplicate prediction_id already on task: {pid}")
            by_pid[pid] = shape
    # A pending transaction can have committed before the client saved its response.
    # If a pending shape disappeared, never recreate it: it might be a human deletion.
    uncertain = set(state.get("pending", [])) - by_pid.keys() - state["applied"].keys()
    if uncertain:
        write_json(run / "upload_uncertain.json", {"prediction_ids": sorted(uncertain)})
        raise RuntimeError("Annotation commit outcome is uncertain. Inspect upload_uncertain.json and CVAT; automatic retry would risk restoring human deletions.")
    for pid, shape in by_pid.items():
        state["applied"].setdefault(pid, {"annotation_id": shape["id"], "recovered": True})
    state["pending"] = []
    write_json(state_path, state)
    todo = [shape for pid, shape in desired.items() if pid not in state["applied"]]
    # Keep requests bounded; no NMS is performed here.
    for offset in range(0, len(todo), 200):
        batch = todo[offset:offset + 200]
        expected = {attrs_by_name(s, labels)["prediction_id"]: s for s in batch}
        state["pending"] = list(expected)
        write_json(state_path, state)
        response = client.request("PATCH", f"/api/tasks/{task_id}/annotations", params={"action": "create"},
                                  json={"version": current.get("version", 0), "tags": [], "tracks": [], "shapes": batch})
        # Save returned IDs before readback so an edited/deleted box is not recreated.
        for shape in response.get("shapes", []):
            pid = attrs_by_name(shape, labels).get("prediction_id")
            if pid in expected:
                state["applied"][pid] = {"annotation_id": shape["id"], "recovered": False}
        write_json(state_path, state)
        current = client.annotations(task_id)
        validate_created(current["shapes"], expected, labels)
        for shape in current["shapes"]:
            pid = attrs_by_name(shape, labels).get("prediction_id")
            if pid in expected:
                state["applied"][pid] = {"annotation_id": shape["id"], "recovered": False}
        state["pending"] = []
        write_json(state_path, state)
        LOG.info("Verified %d/%d pre-labels", len(state["applied"]), len(boxes))
    state["phase"] = "complete"
    state["task_url"] = f"{client.url}/tasks/{task_id}"
    write_json(state_path, state)
    timing_path = run / "upload_timing.json"
    timing = read_json(timing_path) if timing_path.exists() else {"attempts": []}
    timing["attempts"].append({"seconds": time.perf_counter() - start, "created_shapes": len(todo), "frames": len(mapping)})
    write_json(timing_path, timing)
    LOG.info("Edit and review cuboids: %s", state["task_url"])
    return state


def export_annotations(cfg, client=None, ids_recreated=False):
    run = Path(cfg["run_dir"])
    manifest = load_manifest(run)
    state = read_json(run / "cvat_state.json")
    client = client or CVATClient(cfg["cvat"])
    client.check_version()
    if state["manifest_signature"] != manifest["signature"] or state["server_url"] != client.url or state["organization"] != os.environ.get("CVAT_ORG", ""):
        raise ValueError("Task mapping provenance does not match this run/server")
    task_id = state["task_id"]
    mapping = frame_mapping(manifest, client.request("GET", f"/api/tasks/{task_id}/data/meta"))
    if mapping != state["frame_mapping"]:
        raise ValueError("CVAT mapping changed")
    labels = client.labels(task_id)
    label_names = {l["id"]: l["name"] for l in labels}
    aliases = {v: k for k, v in cfg["cvat"].get("label_mapping", {}).items()}
    actual = client.annotations(task_id)
    write_json(run / "cvat_export_raw.json", actual)
    if actual.get("tracks") or actual.get("tags"):
        raise ValueError("MVP export supports standalone cuboids only; tracks/tags are retained in raw export but require a separate adapter")
    inverse = {cvat_frame: manifest["frames"][int(frame)] for frame, cvat_frame in mapping.items()}
    id_to_pid = {value["annotation_id"]: pid for pid, value in state["applied"].items()}
    result, seen = [], set()
    for shape in actual["shapes"]:
        if shape["type"] != "cuboid" or shape.get("outside"):
            raise ValueError("Export contains an unsupported non-cuboid or outside shape")
        if shape["frame"] not in inverse:
            raise ValueError("Shape references unknown frame")
        frame = inverse[shape["frame"]]
        attrs = attrs_by_name(shape, labels)
        pid = None if ids_recreated else id_to_pid.get(shape["id"])
        source = "annotation_id" if pid else "new_annotation_id"
        if not pid and attrs.get("prediction_id") in state["applied"]:
            pid, source = attrs["prediction_id"], "prediction_attribute_after_id_loss"
        if not pid and ids_recreated:
            source = "unresolved_after_id_loss"
        pid = pid or f"manual:{task_id}:{shape['id']}"
        if pid in seen:
            raise ValueError("Duplicate restored prediction identity; cannot evaluate edits reliably")
        seen.add(pid)
        class_name = label_names[shape["label_id"]]
        class_name = aliases.get(class_name, class_name)
        # A new box may inherit metadata when copied. Stable original ID takes priority;
        # ambiguous copies are rejected above rather than silently counted as deletions.
        confidence = attrs.get("confidence", "")
        result.append(box_from_cvat(shape["points"], sample_token=frame["sample_token"], frame_id=frame["frame_id"],
                                    prediction_id=pid, class_name=class_name, confidence=float(confidence) if confidence else None,
                                    model=attrs.get("model", "manual"), checkpoint=attrs.get("checkpoint", ""),
                                    annotation_id=shape["id"], identity_source=source))
    write_json(run / "reviewed.json", {"manifest_signature": manifest["signature"], "task_id": task_id,
                                        "boxes": [b.to_dict() for b in result], "identity_counts": dict(Counter(b.identity_source for b in result)),
                                        "identity_mode": "server_ids_with_prediction_attribute_fallback"})
    return result
