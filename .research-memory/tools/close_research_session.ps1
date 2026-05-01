param(
    [Parameter(Mandatory = $true)][string]$Title,
    [Parameter(Mandatory = $true)][string]$Summary,
    [string[]]$Finding = @(),
    [string[]]$Decision = @(),
    [string[]]$Bottleneck = @(),
    [string[]]$NextStep = @(),
    [string[]]$Artifact = @(),
    [string[]]$Tag = @(),
    [int]$Importance = 1,
    [switch]$Pinned,
    [string]$ProjectBriefFile = "",
    [string]$ActiveContextFile = "",
    [string]$NextStepsFile = "",
    [string]$Write = ""
)

$pluginRoot = "C:\Users\21107\Documents\Codex\2026-04-29\codex-256k-token-token-token-codex\plugins\research-context"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$scriptPath = Join-Path $pluginRoot "scripts\session_workflow.py"

$commandArgs = @(
    $scriptPath,
    "close",
    "--project-root", $projectRoot,
    "--title", $Title,
    "--summary", $Summary,
    "--importance", $Importance
)

foreach ($item in $Finding) {
    $commandArgs += @("--finding", $item)
}
foreach ($item in $Decision) {
    $commandArgs += @("--decision", $item)
}
foreach ($item in $Bottleneck) {
    $commandArgs += @("--bottleneck", $item)
}
foreach ($item in $NextStep) {
    $commandArgs += @("--next-step", $item)
}
foreach ($item in $Artifact) {
    $commandArgs += @("--artifact", $item)
}
foreach ($item in $Tag) {
    $commandArgs += @("--tag", $item)
}

if ($Pinned) {
    $commandArgs += "--pinned"
}
if (-not [string]::IsNullOrWhiteSpace($ProjectBriefFile)) {
    $commandArgs += @("--project-brief-file", $ProjectBriefFile)
}
if (-not [string]::IsNullOrWhiteSpace($ActiveContextFile)) {
    $commandArgs += @("--active-context-file", $ActiveContextFile)
}
if (-not [string]::IsNullOrWhiteSpace($NextStepsFile)) {
    $commandArgs += @("--next-steps-file", $NextStepsFile)
}
if (-not [string]::IsNullOrWhiteSpace($Write)) {
    $commandArgs += @("--write", $Write)
}

& python @commandArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
