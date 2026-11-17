param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$environmentPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath $environmentPython)) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Nie udalo sie utworzyc venv.' }
    }
    & $environmentPython -m pip install --disable-pip-version-check -r requirements-nn.txt
    if ($LASTEXITCODE -ne 0) { throw 'Instalacja zaleznosci nie powiodla sie.' }
    & $environmentPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Niespojne zaleznosci.' }
    Write-Host 'Srodowisko gotowe. Zainstaluj lokalny pakiet wag wedlug README.'
} finally { Pop-Location }
