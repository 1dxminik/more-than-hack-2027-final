param([Parameter(Position=0, ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
$ErrorActionPreference = 'Stop'
$environmentPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $environmentPython)) { throw 'Najpierw uruchom ./setup.ps1.' }
if (-not $CommandArgs -or $CommandArgs.Count -eq 0) { $CommandArgs = @('--help') }
Push-Location -LiteralPath $PSScriptRoot
try {
    & $environmentPython -s -X utf8 (Join-Path $PSScriptRoot 'phase1.py') @CommandArgs
    if ($LASTEXITCODE -ne 0) { throw "C5: polecenie zakonczylo sie kodem $LASTEXITCODE." }
} finally { Pop-Location }
