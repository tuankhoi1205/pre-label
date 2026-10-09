param(
    [ValidateSet('Start', 'Status', 'BuildTool', 'Tool', 'CreateUser', 'Stop')]
    [string]$Action = 'Start',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ToolArguments
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
$sourcePath = Join-Path $projectPath '.local\upstream\cvat-2.20.0'
$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerCommand) {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe",
        "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe"
    )
    $dockerPath = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
} else { $dockerPath = $dockerCommand.Source }
if (-not $dockerPath) { throw 'Install and open Docker Desktop first. Installer is in .local\downloads.' }
& $dockerPath info --format '{{.OSType}}'
if ($LASTEXITCODE -ne 0) { throw 'Docker engine is not ready. Open Docker Desktop with the WSL2 backend.' }
if (-not (Test-Path -LiteralPath $sourcePath)) { throw 'Missing CVAT source; see docs/CVAT_BUTTON.md.' }
$env:CVAT_VERSION = 'v2.20.0'
$env:CVAT_HOST = 'localhost'
$composeArgs = @('compose', '--project-name', 'cvat', '--project-directory', $sourcePath,
    '-f', (Join-Path $sourcePath 'docker-compose.yml'),
    '-f', (Join-Path $sourcePath 'components\serverless\docker-compose.serverless.yml'),
    '-f', (Join-Path $PSScriptRoot 'compose.prelabel.yml'))
switch ($Action) {
    'Start' {
        $pythonPath = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
        if (-not (Test-Path -LiteralPath $pythonPath)) {
            $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
            if ($pythonCommand -and $pythonCommand.Source -notmatch 'WindowsApps') { $pythonPath = $pythonCommand.Source }
        }
        if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Python is needed only to apply the CVAT source patch.' }
        & $pythonPath (Join-Path $PSScriptRoot 'patch_cvat.py') $sourcePath
        if ($LASTEXITCODE -ne 0) { throw 'CVAT source patch failed; containers were not changed.' }
        & $dockerPath @composeArgs config --quiet
        if ($LASTEXITCODE -ne 0) { throw 'Invalid compose configuration; containers were not changed.' }
        # Nuclio 1.13 hardcodes this retired mirror for its local storage helper.
        # Use the original official Alpine image under the expected local tag.
        & $dockerPath image inspect 'gcr.io/iguazio/alpine:3.17' *> $null
        if ($LASTEXITCODE -ne 0) {
            & $dockerPath pull 'alpine:3.17'
            if ($LASTEXITCODE -ne 0) { throw 'Could not pull the Nuclio storage helper.' }
            & $dockerPath tag 'alpine:3.17' 'gcr.io/iguazio/alpine:3.17'
            if ($LASTEXITCODE -ne 0) { throw 'Could not prepare the Nuclio storage helper.' }
        }
        & $dockerPath @composeArgs build cvat_ui
        if ($LASTEXITCODE -ne 0) { throw 'CVAT UI build failed.' }
        & $dockerPath @composeArgs up -d
    }
    'Status' { & $dockerPath @composeArgs ps }
    'BuildTool' {
        $pythonPath = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
        if (-not (Test-Path -LiteralPath $pythonPath)) {
            $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
            if ($pythonCommand -and $pythonCommand.Source -notmatch 'WindowsApps') { $pythonPath = $pythonCommand.Source }
        }
        if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Python is needed to fetch and verify the official runtime wheels.' }
        & $pythonPath (Join-Path $PSScriptRoot 'runtime_wheels.py')
        if ($LASTEXITCODE -ne 0) { throw 'Official CUDA wheel download or hash verification failed.' }
        & $dockerPath @composeArgs --profile tools build prelabel
    }
    'Tool' {
        if (-not $ToolArguments) { throw 'Supply CLI arguments after Tool, for example Tool doctor.' }
        & $dockerPath @composeArgs --profile tools run --rm prelabel python -m prelabel @ToolArguments
    }
    'CreateUser' { & $dockerPath @composeArgs exec cvat_server python3 manage.py createsuperuser }
    'Stop' { & $dockerPath @composeArgs stop }
}
if ($LASTEXITCODE -ne 0) { throw "Docker action failed: $Action" }
