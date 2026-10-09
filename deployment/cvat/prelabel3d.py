"""CVAT 2.20 task-level adapter; imports only stdlib until annotate() runs.

Journal is stored in CVAT's persistent data volume. Committed frames stay
committed when people edit/delete boxes. Uncertain commits stop for inspection.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path

FUNCTION_ID = "pth-prelabel-centerpoint"
ATTRIBUTES = ("prediction_id", "confidence", "model", "checkpoint", "run_id")


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"version": 1, "frames": {}, "pending": None}

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(self.data, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(self.path)

    @contextmanager
    def lock(self):
        # CVAT workers run on Linux. flock releases automatically after a crash.
        import fcntl
        with self.path.with_suffix(".lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            if self.path.exists():
                self.data = json.loads(self.path.read_text())
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    @staticmethod
    def identities(shapes, prediction_specs):
        result = {}
        for shape in shapes:
            pid = next((a["value"] for a in shape.get("attributes", [])
                        if a["spec_id"] in prediction_specs), None)
            if pid:
                if pid in result:
                    raise ValueError("Duplicate prediction_id in CVAT; inspect copied annotations")
                result[pid] = shape
        return result

    def recover(self, read_shapes, prediction_specs):
        pending = self.data["pending"]
        if not pending:
            return
        present = self.identities(read_shapes(), prediction_specs)
        if any(pid not in present for pid in pending["ids"]):
            raise ValueError("CenterPoint annotation commit is uncertain. Inspect the persistent prelabel3d journal and CVAT before retrying; missing boxes may be human deletions.")
        self._commit(pending, present)

    def _commit(self, pending, present):
        self.data["frames"][str(pending["frame"])] = {
            "annotation_ids": {pid: present[pid]["id"] for pid in pending["ids"]},
            "baseline": pending["baseline"],
        }
        self.data["pending"] = None
        self.save()

    def apply(self, frame, shapes, read_shapes, create_shapes, prediction_specs):
        self.recover(read_shapes, prediction_specs)
        if str(frame) in self.data["frames"]:
            return
        desired = self.identities(shapes, prediction_specs)
        if len(desired) != len(shapes):
            raise ValueError("Map the prediction_id attribute for every CenterPoint label")
        present = self.identities(read_shapes(), prediction_specs)
        todo = [shape for pid, shape in desired.items() if pid not in present]
        pending = {"frame": frame, "ids": list(desired), "baseline": shapes}
        self.data["pending"] = pending
        self.save()
        if todo:
            # The CVAT DB result contains assigned IDs; save these before readback.
            created = self.identities(create_shapes(todo), prediction_specs)
            present.update(created)
        if any(pid not in present for pid in desired):
            raise ValueError("CVAT did not return all created annotation IDs; retry only after inspecting the pending journal")
        self._commit(pending, present)


def parse_shapes(annotations, labels, frame):
    result = []
    for anno in annotations:
        label = labels.get(anno["label"])
        if not label:
            continue
        points = anno.get("points", [])
        if (anno.get("type") != "cuboid" or len(points) != 16 or
                not all(isinstance(v, (int, float)) and math.isfinite(v) for v in points) or
                min(points[6:9]) <= 0):
            raise ValueError("CenterPoint must return 16-value 3D cuboids with positive dimensions")
        attrs = [{"spec_id": label["attributes"][a["name"]], "value": a["value"]}
                 for a in anno.get("attributes", []) if a["name"] in label["attributes"]]
        if not all(any(a["spec_id"] == label["attributes"][name] and a["value"]
                       for a in attrs) for name in ATTRIBUTES):
            raise ValueError("Map all CenterPoint metadata attributes before annotating")
        result.append({"frame": frame, "label_id": label["id"], "source": "auto",
            "attributes": attrs, "group": 0, "type": "cuboid", "occluded": False,
            "outside": False, "points": points, "rotation": 0, "z_order": 0})
    return result


def ensure_metadata(labels, mapping, attribute_model):
    """Add reserved provenance attributes when a task was created in the UI."""
    for label in labels.values():
        if label['type'] not in ('cuboid', 'any'):
            continue
        for name in ATTRIBUTES:
            if name not in label['attributes']:
                attr, _ = attribute_model.objects.get_or_create(label_id=label['id'], name=name,
                    defaults={'input_type': 'text', 'mutable': False, 'default_value': '', 'values': ''})
                if attr.input_type != 'text' or attr.mutable:
                    raise ValueError(f'CenterPoint reserved attribute {name} must be immutable text')
                label['attributes'][name] = attr.id
    for entry in (mapping or {}).values():
        label = labels.get(entry['name'])
        if label and label['type'] in ('cuboid', 'any'):
            entry.setdefault('attributes', {}).update({name: name for name in ATTRIBUTES})


def annotate(function, db_task, labels, mapping, frame_set, update_progress, db_job=None):
    from cvat.apps.dataset_manager import task as dm_task
    from cvat.apps.engine.serializers import LabeledDataSerializer
    from cvat.apps.engine.models import AttributeSpec
    from django.db import transaction
    if db_task.dimension != "3d":
        raise ValueError("CenterPoint requires a 3D PCD task")
    with transaction.atomic():
        ensure_metadata(labels, mapping, AttributeSpec)
    for label in labels.values():
        if label["type"] not in ("cuboid", "any") or any(name not in label["attributes"] for name in ATTRIBUTES):
            raise ValueError("CenterPoint requires Cuboid/Any labels")
    specs = {label["attributes"]["prediction_id"] for label in labels.values()}
    journal = Journal(Path(db_task.data.get_data_dirname()) / "prelabel3d" / "centerpoint.json")

    def read_shapes():
        return dm_task.get_task_data(db_task.id)["shapes"]

    def create_shapes(shapes):
        serializer = LabeledDataSerializer(data={"version": 0, "tags": [], "tracks": [], "shapes": shapes})
        serializer.is_valid(raise_exception=True)
        return dm_task.patch_task_data(db_task.id, serializer.data, "create")["shapes"]

    frames = [f for f in frame_set if f not in db_task.data.deleted_frames]
    with journal.lock():
        journal.recover(read_shapes, specs)
        for index, frame in enumerate(frames):
            if str(frame) not in journal.data["frames"]:
                annotations = function.invoke(db_task, db_job=db_job, data={
                    "frame": frame, "quality": "original", "mapping": mapping})
                if not update_progress((index + 1) / len(frames)):
                    break
                journal.apply(frame, parse_shapes(annotations, labels, frame), read_shapes, create_shapes, specs)
            elif not update_progress((index + 1) / len(frames)):
                break
