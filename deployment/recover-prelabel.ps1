# Apply the internal Nuclio route after restarting a wedged Docker/WSL VM.
# Keeps CVAT volumes, point clouds and annotations; never starts inference.
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
$sourcePath = Join-Path $projectPath '.local\upstream\cvat-2.20.0'
$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerCommand) { $dockerPath = $dockerCommand.Source }
else {
    $dockerPath = @(
        "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe",
        "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe"
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $dockerPath) { throw 'Docker Desktop is not installed.' }
$dockerOS = & $dockerPath info --format '{{.OSType}}'
if ($LASTEXITCODE -ne 0 -or $dockerOS -ne 'linux') { throw 'Open Docker Desktop after reboot and wait for the Linux engine to start.' }
$env:CVAT_VERSION = 'v2.20.0'
$env:CVAT_HOST = 'localhost'
$composeArgs = @('compose', '--project-name', 'cvat', '--project-directory', $sourcePath,
    '-f', (Join-Path $sourcePath 'docker-compose.yml'),
    '-f', (Join-Path $sourcePath 'components\serverless\docker-compose.serverless.yml'),
    '-f', (Join-Path $PSScriptRoot 'compose.prelabel.yml'))
& $dockerPath @composeArgs config --quiet
if ($LASTEXITCODE -ne 0) { throw 'Invalid compose configuration; containers were not changed.' }
$pythonPath = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source -notmatch 'WindowsApps') { $pythonPath = $pythonCommand.Source }
}
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Python is needed to apply the CVAT routing patch.' }
& $pythonPath (Join-Path $PSScriptRoot 'patch_cvat.py') $sourcePath
if ($LASTEXITCODE -ne 0) { throw 'CVAT routing patch failed; containers were not changed.' }
& $dockerPath @composeArgs up -d
if ($LASTEXITCODE -ne 0) { throw 'CVAT recovery failed; check Docker Desktop.' }
# Mounted Python source is imported at startup. Recreate only the two services
# using the gateway so the stable processor DNS route takes effect.
& $dockerPath @composeArgs up -d --no-deps --force-recreate cvat_server cvat_worker_annotation
if ($LASTEXITCODE -ne 0) { throw 'CVAT gateway services could not be refreshed.' }
& $dockerPath start nuclio-nuclio-pth-prelabel-centerpoint
if ($LASTEXITCODE -ne 0) { throw 'The existing CenterPoint function could not be started.' }

# CVAT 2.20 treats canceled RQ records as conflicts. Remove only the canceled
# request from this incident, after the old worker has been replaced.
$cleanupCode = @'
import json
from django.conf import settings
from cvat.apps.lambda_manager.views import LambdaQueue
from rq.job import JobStatus
assert settings.NUCLIO['INVOKE_METHOD'] == 'dashboard'
queue = LambdaQueue()
request = queue._get_queue().fetch_job('autoannotate:task-2')
removed = False
if request is not None and request.get_status() == JobStatus.CANCELED:
    function = request.kwargs.get('function')
    if request.kwargs.get('task') != 2 or getattr(function, 'id', None) != 'pth-prelabel-centerpoint':
        raise RuntimeError('Unexpected request identity; no record was removed')
    request.delete()
    removed = True
print(json.dumps({'invoke_method': settings.NUCLIO['INVOKE_METHOD'], 'canceled_request_removed': removed, 'inference_started': False}))
'@
& $dockerPath exec cvat_worker_annotation python manage.py shell -c $cleanupCode
if ($LASTEXITCODE -ne 0) { throw 'Request recovery failed; do not click Annotate until this is resolved.' }
$checkCode = @'
import requests
from cvat.apps.lambda_manager.views import LambdaGateway
gateway = LambdaGateway()
function = gateway.get('pth-prelabel-centerpoint')
# An empty payload must reach the actual handler and fail before model load.
# A dashboard GET alone can report ready despite a stale invocation IP.
try:
    gateway.invoke(function, {})
    raise RuntimeError('Empty payload unexpectedly accepted')
except requests.HTTPError as exc:
    if exc.response.status_code != 400 or 'base64 PCD' not in exc.response.text:
        raise
print('CenterPoint gateway -> processor DNS -> handler: OK (no inference)')
'@
& $dockerPath exec cvat_worker_annotation timeout 45s python manage.py shell -c $checkCode
if ($LASTEXITCODE -ne 0) { throw 'Internal Nuclio connection is not ready; inspect the function status.' }
Write-Host 'Recovery completed. Open http://localhost:8080/tasks/2 and reload the page.'
Write-Host 'Actions -> Automatic annotation -> CenterPoint 3D (nuScenes) -> Annotate.'
Write-Host 'No inference has been started by this script.'
