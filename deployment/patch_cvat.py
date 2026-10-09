"""Apply small, version-checked changes to the downloaded CVAT 2.20 tree."""
import argparse
import hashlib
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("orientation_patch", Path(__file__).with_name("orientation_patch.py"))
_orientation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_orientation)
ORIENTATION_PATCHES = _orientation.ORIENTATION_PATCHES

UI_FILE = "cvat-ui/src/components/model-runner-modal/detector-runner.tsx"
VIEWS_FILE = "cvat/apps/lambda_manager/views.py"

PATCHES = {
    UI_FILE: [
        ("    const model = models.find((_model): boolean => _model.id === modelID);", """    // PRELABEL_3D: expose cuboid detectors through the existing annotation button.
    const availableModels = models.filter((candidate) => {
        const is3DModel = candidate.kind === ModelKind.DETECTOR && candidate.labels.length > 0 &&
            candidate.labels.every((label) => label.type === 'cuboid');
        return dimension === DimensionType.DIMENSION_3D ? is3DModel : !is3DModel;
    });
    const model = availableModels.find((_model): boolean => _model.id === modelID);"""),
        ("    const convertMasks2PolygonVisible = isDetector &&", "    const convertMasks2PolygonVisible = dimension === DimensionType.DIMENSION_2D && isDetector &&"),
        ("placeholder={dimension === DimensionType.DIMENSION_2D ? 'Select a model' : 'No models available'}", "placeholder={availableModels.length ? 'Select a model' : 'No models available'}"),
        ("disabled={dimension !== DimensionType.DIMENSION_2D}", "disabled={!availableModels.length}"),
        ("{models.map(", "{availableModels.map("),
        ("{isDetector && withCleanup && (", "{isDetector && withCleanup && dimension === DimensionType.DIMENSION_2D && ("),
    ],
    VIEWS_FILE: [
        ("        class Results:\n", """        # PRELABEL_3D: persist applied frames and preserve human edits/deletions.
        if function.id == 'pth-prelabel-centerpoint':
            from cvat.apps.lambda_manager.prelabel3d import annotate
            return annotate(function, db_task, labels, mapping,
                cls._get_frame_set(db_task, db_job), cls._update_progress, db_job=db_job)

        class Results:
"""),
        ("        if cleanup:\n", """        # Never clear human annotations when using this 3D pre-label function.
        if function.id == 'pth-prelabel-centerpoint' and cleanup:
            raise ValidationError('CenterPoint preserves existing annotations; use a fresh task for a new run')

        if cleanup:
"""),
    ],
}

# Apply independently so installations with the original 3D patch can upgrade.
ROUTING_PATCHES = {
    VIEWS_FILE: [
        ("    def invoke(self, func, payload):\n        invoke_method = {", """    def invoke(self, func, payload):
        # PRELABEL_3D_DNS: Docker restarts may change the processor's IP while
        # Nuclio 1.13 retains its old internalInvocationUrls. Resolve the local
        # processor by its Docker DNS name for this dedicated 3D integration.
        if func.id == 'pth-prelabel-centerpoint' and os.path.exists('/.dockerenv'):
            with make_requests_session() as session:
                reply = session.post(
                    'http://nuclio-nuclio-pth-prelabel-centerpoint:8080',
                    timeout=settings.NUCLIO['DEFAULT_TIMEOUT'], json=payload)
                reply.raise_for_status()
                return reply.json()
        invoke_method = {"""),
    ],
}


def patch(root):
    root = Path(root).resolve()
    if (root / "cvat/__init__.py").read_text().find("VERSION = (2, 20, 0, 'final', 0)") < 0:
        raise ValueError("Only the downloaded CVAT v2.20.0 source is supported")
    changes = []
    for name, replacements in PATCHES.items():
        path = root / name
        text = path.read_text(encoding="utf-8")
        original = text
        if "PRELABEL_3D" in text:
            if not all(new in text for _, new in replacements):
                raise ValueError(f"Partially patched or changed CVAT source: {name}")
        else:
            for old, new in replacements:
                if text.count(old) != 1:
                    raise ValueError(f"CVAT source differs from v2.20.0 at {name}; no files changed")
                text = text.replace(old, new)
        for old, new in ROUTING_PATCHES.get(name, []):
            if new in text:
                continue
            if text.count(old) != 1:
                raise ValueError(f"CVAT source differs from v2.20.0 at {name}; no files changed")
            text = text.replace(old, new)
        if text != original:
            changes.append((path, text))
    for name, replacements in ORIENTATION_PATCHES.items():
        path = root / name
        text = path.read_text(encoding="utf-8")
        original = text
        for old, new in replacements:
            if new in text:
                continue
            if text.count(old) != 1:
                raise ValueError(f"CVAT orientation source differs from v2.20.0 at {name}; no files changed")
            text = text.replace(old, new)
        if text != original:
            changes.append((path, text))
    for path, text in changes:
        path.write_text(text, encoding="utf-8", newline="\n")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in [*PATCHES, *ORIENTATION_PATCHES]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    print(patch(parser.parse_args().root))
