param([string]$Config = 'configs/mini.yaml', [string]$RunDir = 'runs/ui-centerpoint', [switch]$IdsRecreated)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
$runPath = [IO.Path]::GetFullPath((Join-Path $projectPath $RunDir))
if (-not $runPath.StartsWith($projectPath + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'RunDir must be inside this local project.'
}
$state = Get-Content -LiteralPath (Join-Path $runPath 'ui_task.json') -Raw | ConvertFrom-Json
if ($state.server_url -notin @('http://cvat-server:8080', 'http://cvat_server:8080', 'http://localhost:8080')) {
    throw 'This script reads only the local CVAT Docker journal.'
}
$taskId = [int]$state.task_id
if ($taskId -le 0) { throw 'UI task has no task_id.' }
$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerCommand) { $dockerPath = $dockerCommand.Source } else {
    $dockerPath = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe"
    if (-not (Test-Path -LiteralPath $dockerPath)) { $dockerPath = "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe" }
}
$code = @'
import fcntl, json, os, sys
from pathlib import Path
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cvat.settings.production')
import django
django.setup()
from cvat.apps.engine.models import Task
task_id = int(sys.argv[1])
task = Task.objects.get(pk=task_id)
path = Path(task.data.get_data_dirname()) / 'prelabel3d' / 'centerpoint.json'
if not path.is_file():
    raise SystemExit('No CenterPoint UI journal: finish Annotate first')
with path.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_SH)
    print(json.dumps({'task_id': task_id, 'journal': json.loads(path.read_text())}))
'@
Push-Location -LiteralPath $projectPath
try {
    $snapshot = & $dockerPath exec cvat_server python3 -c $code $taskId
    if ($LASTEXITCODE -ne 0) { throw 'Could not read the persistent CVAT journal.' }
    $snapshot | ConvertFrom-Json | Out-Null
    $relative = ".local/ui-journal-task-$taskId.json"
    # Windows PowerShell 5 writes a BOM with Set-Content -Encoding utf8;
    # Python's JSON reader expects ordinary UTF-8.
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText((Join-Path $projectPath $relative), ($snapshot -join "`n"), $utf8)
    $exportArgs = @('--config', $Config, '--run-dir', $RunDir, 'export', '--ui-journal', $relative)
    if ($IdsRecreated) { $exportArgs += '--ids-recreated' }
    & (Join-Path $PSScriptRoot 'local.ps1') Tool @exportArgs
    if ($LASTEXITCODE -ne 0) { throw 'UI export failed.' }
} finally { Pop-Location }
