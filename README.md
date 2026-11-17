# Ctrl Alt Elite — More Than a Hack 2027, Phase 1

Frame-by-frame ball possession (`H` / `A` / `D`) from the provided ATD tracking data.
A bidirectional temporal convolutional network (MS-TCN style) predicts possession at 5 fps; a Viterbi
decoder with a fixed cost per label change removes short flickers. The submission averages 6 models
trained on the public IDSSE dataset (Bundesliga 2022/23, CC BY 4.0) after degrading it to match the
competition tracking.

Full description, training data, possession rules and limitations: **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)**.

No match data is included in this repository.

## Install (Windows, Python 3.12)

```powershell
./setup.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements-nn.txt
```

On other systems: `python -m venv .venv`, then `pip install -r requirements.txt -r requirements-nn.txt`.

## Reproduce the submission

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.predict_tcn `
    --atd-xml <data>\BAR-ATH_tactical_atd.xml `
    --atd-raw <data>\BAR-ATH_tactical_atd.txt `
    --out Ctrl_Alt_Elite.csv --threads 4
```

Runs in about 30 s on CPU, uses the published weights in `weights/tcn_v1/` (hashes in `SHA256SUMS`)
and ends with the official format validation. Same machine and thread count give a byte-identical file.
SHA-256 of the submitted file: `c5d46c8af5513e597564700ce02cd22b8c8b93ee2d8f51a2b37d8e608efae576`.

## Retrain the models (optional, about 1 h on CPU)

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\fetch_idsse.py
foreach ($m in 'J03WMX','J03WOH','J03WOY','J03WPY','J03WQQ','J03WR9') { foreach ($s in 0..4) {
  .\.venv\Scripts\python.exe -X utf8 -m scripts.degrade_idsse --match $m --seed $s } }
.\.venv\Scripts\python.exe -X utf8 -m scripts.train_tcn cv --run v1
```

Models are written to `outputs/tcn/v1/`. Retraining is not bit-identical to the published weights.

## Tests

```powershell
.\run.ps1 check
```

Verifies the unchanged organiser files in `official/` and runs the unit tests on synthetic data.

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
