param([string]$Python = '')
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$environmentPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'

if (-not (Test-Path -LiteralPath $environmentPython)) {
    $reuseBundledPackages = $false
    if (-not $Python) {
        if (Test-Path -LiteralPath $bundledPython) {
            $Python = $bundledPython
            $reuseBundledPackages = $true
        } else {
            $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
            if ($launcher) {
                $Python = & $launcher.Source -3 -c 'import sys; print(sys.executable)'
                if ($LASTEXITCODE -ne 0) { throw 'Python launcher failed. Use ./setup.ps1 -Python C:\path\python.exe' }
            } else {
                $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
                if (-not $pythonCommand) { throw 'Install Python 3.11+ or pass -Python C:\path\python.exe' }
                $Python = $pythonCommand.Source
            }
        }
    }
    & $Python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11+ is required.' }
    if ($reuseBundledPackages) {
        Write-Host 'Creating .venv and reusing installed bundled packages (no download needed if versions match).'
        & $Python -m venv --system-site-packages (Join-Path $projectRoot '.venv')
    } else {
        & $Python -m venv (Join-Path $projectRoot '.venv')
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv.' }
}

& $environmentPython (Join-Path $projectRoot 'scripts\check_environment.py')
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Installing pinned dependencies from requirements.txt.'
    & $environmentPython -m pip install -r (Join-Path $projectRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $environmentPython (Join-Path $projectRoot 'scripts\check_environment.py')
    if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed.' }
}
Write-Host 'Ready. Next: ./run.ps1 demo'
