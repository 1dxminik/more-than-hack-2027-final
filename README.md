# C5-alpha-0.1 — Ctrl Alt Elite alternative method

TCN + full N14 + C5 decoder. This is an additional method developed by the team;
the CSV submitted through the competition form comes from our teammate's model.
This branch does not replace that submission. It contains no match data or
competition CSVs. Model weights remain local and outside Git.

## Installation

Reference environment: Windows, Python 3.12.14, PyTorch 2.8.0+cpu,
two CPU threads. Run from the repository root using a Python 3.12 interpreter:

```powershell
.\setup.ps1 -Python python
.\.venv\Scripts\python.exe -s -X utf8 -m scripts.install_assets `
    --from-package 'D:\path\to\code-and-weights.zip'
```

The package is the local C5 code-and-weights archive or its extracted `code/`
directory. The installer reads only the seven model assets listed in
`configs/assets.json` and verifies their sizes and SHA-256 hashes. It does not
copy data, CSVs, logs or caches. The six checkpoints in our teammate's
`weights/tcn_v1/` belong to a different architecture and cannot be used here.
Cloning this branch alone requires the local model package to be supplied separately.

`setup.ps1` creates an isolated venv without inherited packages, installs pinned
CPU dependencies and runs `pip check`. On Linux, use a Python 3.12 venv and
install `requirements-nn.txt`. Identical output across operating systems must
be verified rather than assumed.

## Synthetic checks

```powershell
.\run.ps1 check
.\run.ps1 demo
.\run.ps1 improve
```

`check` verifies `official/`, dependencies and model assets, then tests actual
inference on invented trajectories. Missing PyTorch or weights cause a failure,
not a skipped test. `demo` generates 6,000 frames; `improve` generates 199,319
frames and 46 slots with at most 22 visible players. Both run two fresh processes,
apply the official format checks and compare CSVs byte for byte. They force a
synthetic `MTAH_DATA_DIR`. New inputs go to `data/synthetic-c5/`; outputs go to
`outputs/synthetic-c5/`. These are software checks, not competition scores.
Previous results are preserved.

## Export from explicit ATD files

Command for an authorised operator. It was not run on competition data while
preparing this branch:

```powershell
$dataDir = 'D:\path\to\organiser-bundle'
$resultDir = 'D:\path\to\private-output'
.\.venv\Scripts\python.exe -s -X utf8 -m scripts.predict_c5 `
    --atd-xml "$dataDir\tactical\BAR-ATH_tactical_atd.xml" `
    --atd-raw "$dataDir\tactical\BAR-ATH_tactical_atd.txt" `
    --out "$resultDir\C5.csv"
if ($LASTEXITCODE -ne 0) { throw 'Inference or validation failed' }
```

`dataDir` is the root of the full bundle containing `tactical/`. The exporter
reads only the specified XML/TXT files, performs no data discovery and does not
use `MTAH_DATA_DIR`. The CSV has the `frame,label` header, 199,319 data rows,
frames 0–199318 and H/A/D/X labels. ATD records 1–199319 become zero-based CSV
frames. Processing periods and resets match the source and are explicit in
`configs/c5.json`; frames outside these periods remain X.

The exact model set, its hashes and the output format are checked. Existing CSVs
are not overwritten. No ground truth, video or retraining is needed. Identity of
a particular output requires a separate byte comparison and SHA-256 check of
two fresh exports by the operator. Our teammate's CSV hash is not the C5 reference.
Private inputs, outputs and execution logs remain outside Git and AI services.

## Method and provenance

37 features at 25 fps feed five frozen TCN B+CE models. Their H/A/D probabilities
are averaged with team symmetrisation preserved. Full N14 provides state and
evidence of ball control, stoppages and restarts. C5 combines the TCN and
meta-model probabilities log-linearly at alpha=0.1; Viterbi and fixed rules
align boundaries and protect D plateaus. Parameters are not selected at runtime.

Source commit: `d99253c24e739735510f76be40f1adaaafe39344`.
`SOURCE_MANIFEST.json` records source hashes and migration details. Inference
lives in `solution/`; frozen N14 is in `solution/n14/`. There is no runtime
AST/exec, function replacement, `sys.path` modification or dependency on historical
`outputs/`. The TCN and meta-model originate from work on public IDSSE data;
Linear N14 was trained on synthetic data. This branch contains inference,
without training or large research campaigns. Stoppage errors, ambiguous contacts
and tracking gaps remain; development results do not establish performance on
an unseen match.

[Sources and licences](THIRD_PARTY_NOTICES.md) ·
[Model assets](weights/c5-alpha-0.1/README.md).

## Migration verification — 17 November 2026

11 tests passed without skips in an isolated venv; 17 upstream files passed
integrity checks. Both the demo and the full 199,319-frame run produced two
byte-identical CSVs with valid formats. The full synthetic CSV is byte-identical
to the frozen original C5 output. On two additional inputs, all 20 diagnostic
arrays match exactly, including features, both models' probabilities and the
N14 trace. Reports remain local.
