param(
    [string]$Task = "继续 period_ms_tree_query 主线",
    [string]$Query = "",
    [string]$ThreadLink = "codex://threads/019dce40-34ba-7cb3-b1b9-6fde3d6a6ea7",
    [string]$Write = ""
)

$pluginRoot = "C:\Users\21107\Documents\Codex\2026-04-29\codex-256k-token-token-token-codex\plugins\research-context"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$scriptPath = Join-Path $pluginRoot "scripts\session_workflow.py"
$resolvedQuery = if ([string]::IsNullOrWhiteSpace($Query)) { $Task } else { $Query }

$commandArgs = @(
    $scriptPath,
    "start",
    "--project-root", $projectRoot,
    "--task", $Task,
    "--query", $resolvedQuery,
    "--thread-link", $ThreadLink,
    "--language", "zh"
)

if (-not [string]::IsNullOrWhiteSpace($Write)) {
    $commandArgs += @("--write", $Write)
}

& python @commandArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
