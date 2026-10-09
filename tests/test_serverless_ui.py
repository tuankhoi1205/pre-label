import base64
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import pytest
from prelabel.dataset import write_pcd
from prelabel.geometry import cvat_points
from prelabel.io import file_digest, read_json
from prelabel.serverless import CenterPointFunction
from prelabel.ui_task import bind_ui_journal, create_task

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backend = load_module('prelabel3d_test', ROOT / 'deployment/cvat/prelabel3d.py')
patcher = load_module('patch_cvat_test', ROOT / 'deployment/patch_cvat.py')
nuclio_handler = load_module('nuclio_handler_test', ROOT / 'deployment/nuclio/main.py')


def test_nuclio_registers_before_data_and_recovers_after_prepare(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(nuclio_handler, 'load_config', lambda path: {'run_dir': 'unused'})
    attempts = []

    class PreparedFunction:
        def __init__(self, cfg):
            attempts.append(cfg)
            if len(attempts) == 1:
                raise FileNotFoundError('manifest.json')

        def predict(self, payload):
            return [{'type': 'cuboid'}]

    monkeypatch.setattr(nuclio_handler, 'CenterPointFunction', PreparedFunction)
    context = SimpleNamespace(user_data=SimpleNamespace(), Response=lambda **kw: kw)
    nuclio_handler.init_context(context)
    assert attempts == [] and context.user_data.model is None
    event = SimpleNamespace(body=b'{"image": "fixture"}')
    response = nuclio_handler.handler(context, event)
    assert response['status_code'] == 400 and 'Prepare' in response['body']
    assert context.user_data.model is None
    assert nuclio_handler.handler(context, event)['status_code'] == 200
    assert nuclio_handler.handler(context, event)['status_code'] == 200
    assert len(attempts) == 2


@pytest.fixture
def function_fixture(run_fixture, monkeypatch):
    import numpy as np
    from prelabel import serverless
    cfg, manifest, boxes = run_fixture
    run = Path(cfg['run_dir'])
    write_pcd(run / manifest['frames'][1]['pcd_file'], np.array([[4, 5, 6, 100, 5]], float))
    manifest['frames'][1]['pcd_sha256'] = file_digest(run / manifest['frames'][1]['pcd_file'])
    state = {'run_id': 'synthetic-ui-run', 'signature': 'synthetic-ui-signature',
             'config_tree_sha256': 'synthetic-config-tree'}
    monkeypatch.setattr(serverless, 'initialize_inference', lambda c: (manifest, run, {}, state))
    monkeypatch.setattr(serverless, 'validate_manifest', lambda *a, **kw: None)
    loads, calls = [], []

    class SyntheticAdapter:
        resolved_config = 'synthetic test fixture'
        def __init__(self, d, rid):
            loads.append(rid)
        def predict(self, frame, root, *, run_id=None):
            calls.append(frame['frame_id'])
            prediction = replace(boxes[frame['frame_id']], sample_token=frame['sample_token'],
                                 prediction_id=f"{run_id}:{frame['sample_token']}:0")
            return {'forward_seconds': 0}, [prediction]

    for p in (run / 'predictions').glob('*.json'):
        p.unlink()
    function = CenterPointFunction(cfg, SyntheticAdapter)
    payloads = [{'image': base64.b64encode((run / f['pcd_file']).read_bytes()).decode()}
                for f in manifest['frames']]
    return function, payloads, loads, calls, boxes


def test_native_pcd_protocol_single_load_and_cached_frames(function_fixture):
    function, payloads, loads, calls, boxes = function_fixture
    result = function.predict(payloads[0])
    assert result[0]['type'] == 'cuboid'
    assert result[0]['points'] == pytest.approx(cvat_points(boxes[0]))
    assert {a['name'] for a in result[0]['attributes']} == set(backend.ATTRIBUTES)
    assert function.predict(payloads[0]) == result
    function.predict(payloads[1])
    assert calls == [0, 1] and loads == ['synthetic-ui-run']


@pytest.mark.parametrize('payload', [{}, {'image': 'not base64!'},
    {'image': base64.b64encode(b'JPEG or unregistered PCD').decode()}])
def test_reject_unprepared_data(function_fixture, payload):
    with pytest.raises(ValueError):
        function_fixture[0].predict(payload)
    assert not function_fixture[2]


def test_ui_threshold_filters_cached_shapes(function_fixture):
    function, payloads, loads, calls, _ = function_fixture
    assert len(function.predict(payloads[0])) == 1
    assert function.predict({**payloads[0], 'threshold': 0.95}) == []
    assert calls == [0]
    with pytest.raises(ValueError, match='Threshold'):
        function.predict({**payloads[0], 'threshold': float('nan')})


def test_cached_provenance_is_verified(function_fixture):
    function, payloads, _, _, _ = function_fixture
    function.predict(payloads[0])
    path = function.run / 'predictions/000000.json'
    from prelabel.io import write_json
    record = read_json(path)
    record['sample_token'] = 'tampered'
    write_json(path, record)
    with pytest.raises(ValueError, match='provenance'):
        function.predict(payloads[0])


def test_upload_auto_preparation_reuses_one_model_and_preserves_base_cache(function_fixture, tmp_path, monkeypatch):
    import sys
    import numpy as np
    from types import SimpleNamespace
    from prelabel import dataset

    function, payloads, loads, calls, _ = function_fixture
    root = tmp_path / 'raw'
    root.mkdir()
    points = np.array([[7, 8, 9, 100, 5]], dtype='<f4')
    points.tofile(root / 'key.bin')
    scene = {'token': 'scene', 'name': 'scene-upload'}
    sample = {'token': 'sample-upload', 'scene_token': 'scene', 'data': {'LIDAR_TOP': 'lidar'}}
    sd = {'token': 'lidar', 'filename': 'key.bin', 'is_key_frame': True, 'timestamp': 1000000,
          'prev': '', 'calibrated_sensor_token': 'cal', 'ego_pose_token': 'ego'}
    records = {('sample_data', 'lidar'): sd,
               ('calibrated_sensor', 'cal'): {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]},
               ('ego_pose', 'ego'): {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}}
    nusc = SimpleNamespace(scene=[scene], sample=[sample], get=lambda table, token: records[table, token])
    monkeypatch.setattr(dataset, 'open_nuscenes', lambda cfg: nusc)
    monkeypatch.setitem(sys.modules, 'nuscenes.utils.splits', SimpleNamespace(create_splits_scenes=lambda: {
        'mini_val': [], 'mini_train': ['scene-upload']}))
    function.cfg['dataset'] = {'root': str(root), 'version': 'v1.0-mini', 'split': 'mini_val'}
    function.cfg['detector']['sweeps'] = 9
    function.root = root
    upload = tmp_path / 'arbitrary-name.pcd'
    write_pcd(upload, points)
    payload = {'image': base64.b64encode(upload.read_bytes()).decode()}
    validated = function.validate(payload)
    assert validated['ready'] and validated['inference_started'] is False and not loads
    assert validated['sample_token'] == 'sample-upload'
    # An uploaded train scene is resolved even when the initial demo used mini_val.
    auto_run, frame, state = function.resolve_input(payload)
    assert read_json(auto_run / 'manifest.json')['selection']['split'] == 'mini_train'
    assert (auto_run / frame['pcd_file']).read_bytes() == upload.read_bytes()
    base = function.predict(payloads[0])
    result = function.predict(payload)
    assert len(loads) == 1 and calls == [0, 0]
    assert next(a['value'] for a in result[0]['attributes'] if a['name'] == 'prediction_id').startswith(
        state['run_id'] + ':sample-upload:')
    assert function.predict(payload) == result and function.predict(payloads[0]) == base
    assert calls == [0, 0]
    points[0, 0] += 1
    points.tofile(root / 'key.bin')
    monkeypatch.undo()
    # Restore the real validator after the fixture's mock, and reject stale raw inputs.
    from prelabel import serverless
    monkeypatch.setattr(serverless, 'validate_manifest', dataset.validate_manifest)
    with pytest.raises(ValueError, match='LiDAR input changed'):
        function.predict(payload)


def test_create_ui_task_does_not_infer_and_resume_preserves_annotations(run_fixture, fake_cvat):
    cfg, _, _ = run_fixture
    state = create_task(cfg, fake_cvat)
    assert fake_cvat.annotation_calls == 0
    fake_cvat.shapes = [{'id': 101, 'points': [1, 2, 3]}]
    assert create_task(cfg, fake_cvat) == state
    assert fake_cvat.shapes == [{'id': 101, 'points': [1, 2, 3]}]
    assert fake_cvat.create_calls == 1 and fake_cvat.data_calls == 1


def test_ui_model_settings_change_requires_a_new_run(run_fixture, fake_cvat):
    cfg, _, _ = run_fixture
    create_task(cfg, fake_cvat)
    cfg['detector']['confidence'] = 0.65
    with pytest.raises(ValueError, match='new run_dir'):
        create_task(cfg, fake_cvat)
    assert fake_cvat.create_calls == 1


def test_ui_new_run_does_not_reuse_task_for_same_manifest(run_fixture, fake_cvat, tmp_path):
    cfg, manifest, _ = run_fixture
    old_state = create_task(cfg, fake_cvat)
    from prelabel.io import write_json
    new_cfg = deepcopy(cfg)
    new_run = tmp_path / 'different-run'
    new_cfg['run_dir'] = str(new_run)
    write_json(new_run / 'manifest.json', manifest)
    for frame in manifest['frames']:
        path = new_run / frame['pcd_file']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((Path(cfg['run_dir']) / frame['pcd_file']).read_bytes())
    new_state = create_task(new_cfg, fake_cvat)
    assert new_state['name'] != old_state['name']
    assert fake_cvat.create_calls == 2


def ui_export_document(cfg, fake_cvat, boxes):
    from prelabel.cvat import label_maps, shape_payload
    state = create_task(cfg, fake_cvat)
    mapped = label_maps(fake_cvat.labels_data, cfg['detector']['classes'], {})
    frames = {}
    for box in boxes:
        shape = shape_payload(box, state['frame_mapping'], mapped, 'test-run')
        annotation_id = 501 + box.frame_id
        fake_cvat.shapes.append({**deepcopy(shape), 'id': annotation_id})
        frames[str(shape['frame'])] = {'baseline': [shape], 'annotation_ids': {box.prediction_id: annotation_id}}
    return {'task_id': state['task_id'], 'journal': {'version': 1, 'pending': None, 'frames': frames}}


def test_ui_export_preserves_identity_after_edit_delete_and_metadata_removal(run_fixture, fake_cvat):
    from prelabel.cvat import export_annotations
    from prelabel.evaluation import edit_metrics
    cfg, _, boxes = run_fixture
    document = ui_export_document(cfg, fake_cvat, boxes)
    fake_cvat.shapes.pop(0)
    fake_cvat.shapes[0]['points'][0] += 0.5
    fake_cvat.shapes[0]['attributes'] = []
    manual = {**deepcopy(fake_cvat.shapes[0]), 'id': 900, 'label_id': 2}
    fake_cvat.shapes.append(manual)
    assert bind_ui_journal(cfg, document, fake_cvat)['source'] == 'cvat_annotate_button'
    reviewed = export_annotations(cfg, fake_cvat)
    assert reviewed[0].prediction_id == boxes[1].prediction_id
    assert reviewed[0].identity_source == 'annotation_id'
    assert reviewed[0].sample_token == boxes[1].sample_token
    metrics, _ = edit_metrics(boxes, reviewed, cfg['evaluation'])
    assert (metrics['edited'], metrics['deleted'], metrics['added']) == (1, 1, 1)
    assert fake_cvat.annotation_calls == 0
    assert bind_ui_journal(cfg, document, fake_cvat)['applied'][boxes[0].prediction_id]['annotation_id'] == 501


@pytest.mark.parametrize('failure', ['pending', 'wrong_task', 'geometry', 'missing_frame'])
def test_ui_export_rejects_uncertain_or_mismatched_journal(run_fixture, fake_cvat, failure):
    cfg, _, boxes = run_fixture
    document = ui_export_document(cfg, fake_cvat, boxes)
    if failure == 'pending':
        document['journal']['pending'] = {'ids': ['unknown']}
    elif failure == 'wrong_task':
        document['task_id'] += 1
    elif failure == 'geometry':
        next(iter(document['journal']['frames'].values()))['baseline'][0]['points'][0] += 1
    else:
        document['journal']['frames'].pop('0')
    with pytest.raises(ValueError):
        bind_ui_journal(cfg, document, fake_cvat)
    assert not (Path(cfg['run_dir']) / 'baseline.json').exists()


@pytest.fixture
def journal_fixture(tmp_path):
    journal = backend.Journal(tmp_path / 'journal.json')
    shapes = [{'frame': 0, 'points': list(range(16)), 'attributes': [{'spec_id': 7, 'value': 'pid-0'}]}]
    stored = []
    def create(todo):
        stored.extend([{**deepcopy(s), 'id': 101 + i} for i, s in enumerate(todo)])
        return deepcopy(stored)
    return journal, shapes, stored, create


def test_native_journal_preserves_edits_and_deletions(journal_fixture):
    journal, shapes, stored, create = journal_fixture
    journal.apply(0, shapes, lambda: deepcopy(stored), create, {7})
    stored[0]['points'][0] = 99
    journal.apply(0, shapes, lambda: deepcopy(stored), create, {7})
    assert stored[0]['points'][0] == 99
    stored.clear()
    restarted = backend.Journal(journal.path)
    restarted.apply(0, shapes, lambda: deepcopy(stored), create, {7})
    assert not stored


def test_native_journal_recovers_response_loss(journal_fixture):
    journal, shapes, stored, create = journal_fixture
    def lost(todo):
        create(todo)
        raise ConnectionError('Synthetic response loss')
    with pytest.raises(ConnectionError):
        journal.apply(0, shapes, lambda: deepcopy(stored), lost, {7})
    restarted = backend.Journal(journal.path)
    restarted.apply(0, shapes, lambda: deepcopy(stored), create, {7})
    assert len(stored) == 1
    assert restarted.data['frames']['0']['annotation_ids'] == {'pid-0': 101}


def test_native_journal_stops_if_pending_box_might_be_deleted(journal_fixture):
    journal, shapes, stored, create = journal_fixture
    def lost(todo):
        create(todo)
        raise ConnectionError('Synthetic response loss')
    with pytest.raises(ConnectionError):
        journal.apply(0, shapes, lambda: deepcopy(stored), lost, {7})
    stored.clear()
    with pytest.raises(ValueError, match='uncertain'):
        backend.Journal(journal.path).recover(lambda: deepcopy(stored), {7})
    assert not stored


def test_native_journal_records_zero_predictions(tmp_path):
    journal = backend.Journal(tmp_path / 'journal.json')
    journal.apply(0, [], lambda: [], lambda shapes: pytest.fail('Should not create shapes'), {7})
    assert journal.data['frames']['0']['annotation_ids'] == {}


def test_version_checked_patch_is_repeatable_and_leaves_2d_models(tmp_path):
    (tmp_path / 'cvat').mkdir()
    (tmp_path / 'cvat/__init__.py').write_text("VERSION = (2, 20, 0, 'final', 0)")
    for name, replacements in patcher.PATCHES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(old for old, _ in replacements + patcher.ROUTING_PATCHES.get(name, [])))
    for name, replacements in patcher.ORIENTATION_PATCHES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(old for old, _ in replacements))
    first = patcher.patch(tmp_path)
    assert patcher.patch(tmp_path) == first
    text = (tmp_path / patcher.UI_FILE).read_text()
    assert "dimension === DimensionType.DIMENSION_3D ? is3DModel : !is3DModel" in text
    assert 'disabled={!availableModels.length}' in text


def test_partial_patch_rejected_before_other_file_writes(tmp_path):
    (tmp_path / 'cvat').mkdir()
    (tmp_path / 'cvat/__init__.py').write_text("VERSION = (2, 20, 0, 'final', 0)")
    for name, replacements in patcher.PATCHES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(old for old, _ in replacements + patcher.ROUTING_PATCHES.get(name, [])))
    for name, replacements in patcher.ORIENTATION_PATCHES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(old for old, _ in replacements))
    path = tmp_path / patcher.VIEWS_FILE
    path.write_text('PRELABEL_3D partial')
    original = (tmp_path / patcher.UI_FILE).read_text()
    with pytest.raises(ValueError, match='Partially'):
        patcher.patch(tmp_path)
    assert (tmp_path / patcher.UI_FILE).read_text() == original


def test_processor_input_error_reaches_cvat_instead_of_generic_400():
    from contextlib import nullcontext
    from types import SimpleNamespace
    import requests

    response = SimpleNamespace(status_code=400, json=lambda: {'error': 'PCD is not in the prepared manifest'})
    session = SimpleNamespace(post=lambda *args, **kwargs: response)
    source = patcher.ROUTING_PATCHES[patcher.VIEWS_FILE][0][1]
    for old, new in patcher.ERROR_PATCHES[patcher.VIEWS_FILE]:
        source = source.replace(old, new)
    source = source.split('        invoke_method = {')[0]
    namespace = {'os': SimpleNamespace(path=SimpleNamespace(exists=lambda path: True)),
                 'make_requests_session': lambda: nullcontext(session),
                 'settings': SimpleNamespace(NUCLIO={'DEFAULT_TIMEOUT': 600}), 'requests': requests}
    exec('class Gateway:\n' + source, namespace)
    with pytest.raises(requests.HTTPError, match='PCD is not in the prepared manifest') as caught:
        namespace['Gateway']().invoke(SimpleNamespace(id='pth-prelabel-centerpoint'), {})
    assert caught.value.response is response


def test_ui_task_without_metadata_gets_provenance_and_mapping():
    from types import SimpleNamespace
    created = {}

    def get_or_create(label_id, name, defaults):
        key = label_id, name
        if key not in created:
            created[key] = SimpleNamespace(id=len(created) + 100, **defaults)
        return created[key], True

    attributes = SimpleNamespace(objects=SimpleNamespace(get_or_create=get_or_create))
    labels = {'Car': {'id': 5, 'type': 'cuboid', 'attributes': {}}}
    mapping = {'car': {'name': 'Car', 'attributes': {}}}
    backend.ensure_metadata(labels, mapping, attributes)
    assert set(labels['Car']['attributes']) == set(backend.ATTRIBUTES)
    assert mapping['car']['attributes'] == {name: name for name in backend.ATTRIBUTES}
    backend.ensure_metadata(labels, mapping, attributes)
    assert len(created) == 5
