"""Attach cameras to an existing pinned local CVAT task without editing annotations.

CVAT 2.20 has no public endpoint for adding context after task creation. This
local maintenance command uses its native RelatedFile model and media cache.
It only adds verified camera files/relations and snapshots annotation readback.
"""
import argparse
import json
from pathlib import Path
import subprocess

from prelabel.context import prepare_camera_context
from prelabel.dataset import load_manifest
from prelabel.io import write_json


SERVER_CODE = '''
import hashlib, io, json, os, sys, zipfile
from pathlib import Path
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cvat.settings.production')
import django
django.setup()
from django.db import transaction
from cvat.apps.engine.models import Task, RelatedFile
from cvat.apps.engine.cache import MediaCache
from cvat.apps.dataset_manager.task import get_task_data
from cvat.apps.lambda_manager.views import LambdaQueue

payload = json.load(sys.stdin)
task = Task.objects.select_related('data').get(pk=payload['task_id'])
if task.dimension != '3d' or task.data.size != payload['frame_count']:
    raise ValueError('Expected the matching 3D task and frame count')
images = {image.frame: image for image in task.data.images.all()}
raw = Path(task.data.get_raw_data_dirname()).resolve()
for record in payload['images']:
    if Path(images[record['frame_id']].path).name != record['pcd_name']:
        raise ValueError('CVAT frame name differs from immutable LiDAR manifest')
    target = (raw / record['context_file']).resolve()
    if not target.is_relative_to(raw):
        raise ValueError('Context destination escapes the task raw directory')
    existing = list(images[record['frame_id']].related_files.all())
    desired = {str(raw / r['context_file']) for r in payload['images'] if r['frame_id'] == record['frame_id']}
    if any(str(r.path) not in desired for r in existing):
        raise ValueError('Task already has other context; nothing replaced')
before = get_task_data(task.id)
signature = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
if payload['mode'] == 'check':
    print(json.dumps({'raw_dir':str(raw), 'data_id':task.data_id,
                      'annotations':before, 'annotations_sha256':signature(before)}))
else:
    for record in payload['images']:
        target = raw / record['context_file']
        if hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Copied camera image hash differs from source')
    with transaction.atomic():
        Task.objects.select_for_update().get(pk=task.id)
        for record in payload['images']:
            RelatedFile.objects.get_or_create(data=task.data, primary_image=images[record['frame_id']],
                                              path=str(raw / record['context_file']))
    cache = MediaCache()
    verified = []
    for frame in sorted(images):
        cache._cache.delete(cache._make_context_image_preview_key(task.data, frame))
        buffer, mime = cache.prepare_context_images_chunk(task.data, frame)
        with zipfile.ZipFile(buffer) as archive:
            names = archive.namelist()
            if len(names) != 6 or any(not archive.read(name).startswith(b'\\xff\\xd8') for name in names):
                raise ValueError('Expected six JPEG context images in the native CVAT provider')
        verified.append({'frame':frame, 'images':names, 'mime':mime})
    after = get_task_data(task.id)
    print(json.dumps({'task_id':task.id, 'data_id':task.data_id,
                      'context_images':task.data.related_files.count(), 'verified_frames':verified,
                      'annotations_sha256_before':signature(before), 'annotations_sha256_after':signature(after),
                      'annotation_count':len(after['shapes']), 'annotations_unchanged':before == after,
                      'requests':[j.to_dict() for j in LambdaQueue().get_jobs()]}, default=str))
'''


def attach(docker, task_id, run, dataset_root):
    run = Path(run).resolve()
    manifest = load_manifest(run)
    context = prepare_camera_context(manifest, dataset_root, run)
    payload = {"task_id": task_id, "frame_count": len(manifest["frames"]), "images": context["images"], "mode": "check"}

    def server_call():
        reply = subprocess.run([docker, "exec", "-i", "cvat_server", "python", "-c", SERVER_CODE],
                               input=json.dumps(payload), text=True, capture_output=True, timeout=120)
        if reply.returncode:
            raise RuntimeError(reply.stderr[-2000:])
        return json.loads(reply.stdout)

    check = server_call()
    backup_path = run / "context" / f"task-{task_id}-annotations-before.json"
    if not backup_path.exists():
        write_json(backup_path, check)
    subprocess.run([docker, "cp", str(run / "context" / "related_images"),
                    f"cvat_server:{check['raw_dir']}/"], check=True, timeout=120)
    payload["mode"] = "attach"
    result = server_call()
    result["context_signature"] = context["signature"]
    result["manifest_signature"] = manifest["signature"]
    write_json(run / "context" / f"task-{task_id}-attachment.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", required=True)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--run-dir", default="runs/ui-centerpoint")
    parser.add_argument("--dataset-root", default="data/nuscenes")
    args = parser.parse_args()
    attach(args.docker, args.task_id, args.run_dir, args.dataset_root)
