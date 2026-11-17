param([Parameter(Position=0, ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
$ErrorActionPreference = 'Stop'
$environmentPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $environmentPython)) {
    throw 'Run ./setup.ps1 first.'
}
if (-not $CommandArgs -or $CommandArgs.Count -eq 0) { $CommandArgs = @('--help') }
& $environmentPython -X utf8 (Join-Path $PSScriptRoot 'phase1.py') @CommandArgs
exit $LASTEXITCODE
