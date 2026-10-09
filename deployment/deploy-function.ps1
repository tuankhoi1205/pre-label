param([string]$Config = 'configs/mini.yaml', [string]$RunDir = 'runs/ui-centerpoint', [switch]$RegisterOnly)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
$dockerPath = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe"
if (-not (Test-Path -LiteralPath $dockerPath)) { $dockerPath = "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe" }
$env:PATH = "$(Split-Path -Parent $dockerPath);$env:PATH"
$driveLetter = [IO.Path]::GetPathRoot($projectPath).Substring(0, 1).ToLowerInvariant()
$daemonProjectPath = '/run/desktop/mnt/host/' + $driveLetter + '/' + $projectPath.Substring(3).Replace('\', '/')
Push-Location -LiteralPath $projectPath
try {
    # Generate the config inside the pinned tool image, then use the host bind path.
    & (Join-Path $PSScriptRoot 'local.ps1') Tool '--config' $Config '--run-dir' $RunDir doctor
    if ($LASTEXITCODE -ne 0) { throw 'Tool container check failed.' }
    $prepareArgs = @('python', 'deployment/deploy_function.py', '--config', $Config, '--run-dir', $RunDir, '--mount-source', $daemonProjectPath)
    if ($RegisterOnly) { $prepareArgs += '--allow-unprepared' }
    & $dockerPath run --rm -v "${projectPath}:/workspace" -w /workspace `
        -e NUSCENES_ROOT=/workspace/data/nuscenes prelabel/centerpoint:1.4.0-cu117 `
        @prepareArgs
    if ($LASTEXITCODE -ne 0) { throw 'Function preparation failed.' }
    # The upstream dashboard handles Linux Docker builds via its documented API.
    # No Windows nuctl or additional CLI download is needed.
    & $dockerPath run --rm --network cvat_cvat -v "${projectPath}:/workspace" -w /workspace `
        prelabel/centerpoint:1.4.0-cu117 python deployment/nuclio_api.py --url http://nuclio:8070
    if ($LASTEXITCODE -ne 0) { throw 'CenterPoint deployment failed; inspect Nuclio build/runtime logs.' }
} finally { Pop-Location }
