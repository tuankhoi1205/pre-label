# This enables Windows components; run with Administrator privileges.
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script in PowerShell as Administrator to install WSL and enable VirtualMachinePlatform.'
}
$installerPath = Join-Path $projectPath '.local\downloads\wsl.3.0.1.0.x64.msi'
$signature = Get-AuthenticodeSignature -LiteralPath $installerPath
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Microsoft Corporation') {
    throw 'WSL installer signature is not a valid Microsoft signature.'
}
$logPath = Join-Path $projectPath '.local\downloads\wsl-install.log'
$installProcess = Start-Process -FilePath 'msiexec.exe' -ArgumentList "/i `"$installerPath`" /qn /norestart /L*v `"$logPath`"" -Wait -PassThru -WindowStyle Hidden
if ($installProcess.ExitCode -notin @(0, 3010)) { throw "WSL MSI failed with code $($installProcess.ExitCode); inspect $logPath" }
$featureResults = @()
foreach ($featureName in @('Microsoft-Windows-Subsystem-Linux', 'VirtualMachinePlatform')) {
    $feature = Get-WindowsOptionalFeature -Online -FeatureName $featureName
    if ($feature.State -ne 'Enabled') {
        $result = Enable-WindowsOptionalFeature -Online -FeatureName $featureName -All -NoRestart
        $featureResults += @{feature=$featureName;restart_needed=$result.RestartNeeded}
    }
}
@{wsl_msi_exit_code=$installProcess.ExitCode;features=$featureResults;automatic_restart=$false} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $projectPath '.local\downloads\wsl-setup-result.json')
Write-Output 'WSL installed/enabled. Restart Windows when needed, then open Docker Desktop.'
