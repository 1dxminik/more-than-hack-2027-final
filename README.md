# Ctrl Alt Elite — More Than a Hack 2027, Phase 1

Frame-by-frame ball possession (`H` / `A` / `D`) from the provided ATD tracking data.
A bidirectional temporal convolutional network (MS-TCN style) predicts possession at 5 fps; a Viterbi
decoder with a fixed cost per label change removes short flickers. The submission averages 6 models
trained on the public IDSSE dataset (Bundesliga 2022/23, CC BY 4.0) after degrading it to match the
competition tracking.

Full description, training data, possession rules and limitations: **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)**.

No match data is included in this repository. The README files under `official/`
are unchanged upstream documentation: their examples refer to the full upstream
repository, including public GT and baseline CSVs intentionally absent here.
Use the commands below for this submission; those upstream examples are not the
reproduction instructions for this repository.

## Install

Run all commands from the repository root in a fresh checkout without an existing
`.venv`. The reported reference run used Windows 11, Python 3.12.10,
PyTorch 2.8.0+cpu and 4 inference threads. Use that Python version when checking
the reference CSV hash; the scripts' broader Python minimum is not a claim of
byte-identical output on every environment.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-nn.txt --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip check
```

Check that `py -3.12` selects Python 3.12.10 if reproducing the reference
environment. These commands create an isolated environment without inheriting
Codex packages. `setup.ps1` remains available for the base dependencies, but does
not install PyTorch.

Linux:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -r requirements-nn.txt --index-url https://download.pytorch.org/whl/cpu
./.venv/bin/python -m pip check
```

On macOS, create the same environment and install `requirements-nn.txt` from
the default package index, without the Linux/Windows CPU index option.
The Linux/macOS commands describe installation and execution; byte identity
with the reference Windows output must be checked, not assumed.

## Reproduce the submission

Only two input files are needed: the **tactical** ATD metadata XML and raw TXT.
They remain outside this public repository. In the commands below, the data
directory is the **root of the organiser's unpacked bundle**, which contains
`tactical/`. `scripts.predict_tcn` reads the explicit paths passed to it;
it does not use `MTAH_DATA_DIR`.

Use the six published `model_*.pt` files in `weights/tcn_v1/`, without adding or
replacing models. `SHA256SUMS` lists their hashes. The script loads architecture,
normalisation and decoder penalty from each checkpoint; `config.json` documents
training and is not an inference configuration file.

Windows PowerShell (replace the two directory values):

```powershell
$ErrorActionPreference = 'Stop'
$dataDir = 'D:\path\to\organiser-bundle'
$resultDir = 'D:\path\to\private-submission'

$expectedModels = @()
Get-Content -LiteralPath 'weights/tcn_v1/SHA256SUMS' | ForEach-Object {
    $hash, $name = $_ -split '\s+', 2
    $name = $name.TrimStart('*')
    $expectedModels += $name
    $path = Join-Path 'weights/tcn_v1' $name
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hash) {
        throw "Weight hash mismatch: $name"
    }
}
$actualModels = @(Get-ChildItem -LiteralPath 'weights/tcn_v1' -Filter 'model_*.pt' -File | Select-Object -ExpandProperty Name)
if (Compare-Object $expectedModels $actualModels) { throw 'Unexpected model set' }

.\.venv\Scripts\python.exe -X utf8 -m scripts.predict_tcn `
    --atd-xml "$dataDir\tactical\BAR-ATH_tactical_atd.xml" `
    --atd-raw "$dataDir\tactical\BAR-ATH_tactical_atd.txt" `
    --out "$resultDir\Ctrl_Alt_Elite.csv" --threads 4
if ($LASTEXITCODE -ne 0) { throw 'Prediction or format validation failed' }

$expectedCsv = 'c5d46c8af5513e597564700ce02cd22b8c8b93ee2d8f51a2b37d8e608efae576'
$actualCsv = (Get-FileHash -LiteralPath "$resultDir\Ctrl_Alt_Elite.csv" -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualCsv -ne $expectedCsv) { throw 'CSV differs from the reference submission' }
```

Linux/macOS equivalent for inference (replace the directory values; first verify
the weights against `SHA256SUMS` and keep exactly the six published model files):

```bash
data_dir='/path/to/organiser-bundle'
result_dir='/path/to/private-submission'
./.venv/bin/python -X utf8 -m scripts.predict_tcn \
    --atd-xml "$data_dir/tactical/BAR-ATH_tactical_atd.xml" \
    --atd-raw "$data_dir/tactical/BAR-ATH_tactical_atd.txt" \
    --out "$result_dir/Ctrl_Alt_Elite.csv" --threads 4
```

The command defaults to 199,319 frames and finishes with the official format
checks. It writes UTF-8 `frame,label` CSV, frames 0–199318, with H/A/D labels
and LF line endings. The input reader expects ATD records numbered 1–199319;
the writer converts their row order to the zero-based CSV frame index.
No ground truth, video, Smart Tagging or homography is needed.

The author's reported runtime is about 30 s on CPU, with two byte-identical
runs on the reference environment. The expected CSV SHA-256 is
`c5d46c8af5513e597564700ce02cd22b8c8b93ee2d8f51a2b37d8e608efae576`.
On Linux use `sha256sum` and on macOS `shasum -a 256` to compare the output
against this value. A successful format check alone does not establish identity.

Record `git rev-parse HEAD`, the runtime versions and the local verification
result for the exact public version used. Keep the CSV and private execution
records outside public Git and AI services.

## Retrain the models (optional, about 1 h on CPU)

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\fetch_idsse.py
foreach ($m in 'J03WMX','J03WOH','J03WOY','J03WPY','J03WQQ','J03WR9') { foreach ($s in 0..4) {
  .\.venv\Scripts\python.exe -X utf8 -m scripts.degrade_idsse --match $m --seed $s } }
.\.venv\Scripts\python.exe -X utf8 -m scripts.train_tcn cv --run v1
```

Models are written to `outputs/tcn/v1/`. Retraining is not bit-identical to the published weights.
The submission command continues to use `weights/tcn_v1/`; it does not automatically
switch to newly trained models. The downloader resolves the current IDSSE revision
and records it locally, so this optional recipe does not pin the historical training
input. No retraining or IDSSE download is required to reproduce the submission.

## Tests

```powershell
.\run.ps1 check
```

Verifies the unchanged organiser files in `official/` and runs the unit tests on synthetic data.
Install PyTorch first to include the network test. The current GitHub Actions workflow
installs only the base requirements, so the network test can be skipped there.
These checks do not execute the six published models through the final exporter
and do not prove identity with the reference CSV.

## Layout

```text
scripts/predict_tcn.py   submission command
scripts/train_tcn.py     leave-one-match-out training on IDSSE
scripts/degrade_idsse.py IDSSE -> competition ATD format with simulated tracking errors
solution/                features, network, tracking loader, earlier heuristic methods
weights/tcn_v1/          published model weights and cross-validation results
official/                organiser code (unchanged, MIT license)
tests/                   unit tests
```

## Credits

IDSSE: Bassek et al., Scientific Data 2025, doi:10.1038/s41597-025-04505-y, CC BY 4.0, data owner DFL.
Organiser code: Metrica Sports, MIT license. Details in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
