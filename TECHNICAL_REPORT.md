# Ball possession from tracking: TCN + Viterbi — technical report

Team **Ctrl Alt Elite** · More Than a Hack 2027, Phase 1

## 1. Approach

We label every frame `H`/`A`/`D` with a small neural sequence model followed by a change-penalised decoder.

1. **Features (25 fps → 5 fps).** From the ATD tracking only, per frame: ball position, speed,
   acceleration, whether it is observed or interpolated (gaps ≤ 2 s), time since / until it was last
   seen; for each team the **3 nearest tracklets to the ball, sorted by distance** (distance, offset,
   speed, speed relative to the ball, validity), counts within 2 m and 5 m, team centroid, spread, mean
   speed and "everyone slow". Tracklets are anonymous, so sorting by distance makes the features
   independent of slot order and robust to identity switches. 62 features, averaged over 5-frame bins.
2. **Model.** Bidirectional (non-causal) multi-stage temporal convolutional network in the style of
   MS-TCN (Farha & Gall, CVPR 2019): 2 stages × 10 dilated residual layers, 64 channels, ~0.3 M
   parameters, CPU only. Dilations 1…512 give each stage a receptive field of ±1023 steps (±3.4 min).
   Loss per stage: class-weighted cross-entropy plus 0.15 × the MS-TCN truncated-MSE smoothing term.
3. **Decoder.** Viterbi over H/A/D with one fixed log-cost for every label change. It removes short
   excursions that the evidence does not pay for, which matters because the ranking metric (spellF1)
   punishes fragmented possessions.
4. **Ensemble.** The submission averages the probabilities of the 6 cross-validation fold models
   (`weights/tcn_v1/`) and decodes with the median selected penalty (1.0). Bins are repeated back to
   25 fps. The model never outputs `X`.

**Why this.** Rules on nearest player plus smoothing (official baseline) cannot tell a pass from a
loss, cannot bridge long ball gaps and confuse stoppages with slow play. A learned model with long
bidirectional context can use what happens *after* a gap (who has the ball next, whether the restart
looks like a throw-in) and learns the dead-ball pattern from data. A TCN trains in minutes on CPU and is
easy to inspect; the decoder gives direct control over spell fragmentation.

## 2. Assets used

| Asset | Used | How |
|---|---|---|
| ATD tactical tracking (`*_tactical_atd.xml` + `.txt`) | **yes** | only input of the model; team of each tracklet from the XML |
| Video, homography, Smart Tagging | no | not needed for the chosen approach; no time left to add |
| Official baseline (`official/starter/baseline.py`) | as reference | compared on the same data in every fold |
| Official scorer / validator | yes | all scores and the submission check |
| Public 5-minute ground truth | **no** | not used for training, tuning or model selection |
| IDSSE (public, see §3) | yes | only labelled training / validation data |

**Tried and dropped** (all evaluated on the same degraded IDSSE matches, never on the provided game):

- *Step 1 heuristic*: nearest player + possession memory through ball gaps + explicit stoppage rules.
- *Segment decoder*: hand-made team evidence (distance to ball, player activity) decoded into
  whole H/A/D segments with a minimum duration. It beat smooth50 on every IDSSE match, but under an
  earlier, simpler degradation it reached only spellF1 ≈ 0.27–0.39.
- *TCN without Viterbi*: consistently ~0.015 spellF1 below TCN + Viterbi (table in §5).

## 3. Training data

**IDSSE** — Bassek et al., *An integrated dataset of spatiotemporal and event data in elite soccer*,
Scientific Data 2025, [doi:10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y),
CC BY 4.0, data owner DFL. 7 Bundesliga / 2. Bundesliga 2022/23 matches with official 25 fps
positional data and per-frame `BallPossession` and `BallStatus` attributes supplied by DFL. **We did
not label anything ourselves.** Labels: `BallStatus = 0` → `D`; otherwise `BallPossession` 1/2 → `H`/`A`.

- 6 matches used for training/validation (≈ 6 × 90 min ≈ 0.8 M labelled frames); the 7th (J03WN1,
  chosen with `random.Random(2027)`) is held out and was never used.
- **Domain matching (`scripts/degrade_idsse.py`).** IDSSE tracking is far cleaner than the provided ATD,
  so each match is converted to the exact ATD file format and degraded: anonymous 46-slot tracklets,
  broken tracks, identity swaps near contacts, duplicates, positional noise, single-camera ball
  projection, false balls and ball occlusion that is more likely in crowds, in the air and in fast
  passes. Missing-ball rate and mean object count are calibrated to the public figures in
  `official/DATA.md` (40.7 % / 20.5; we reach 40.67 % / 20.49). IDSSE freezes the ball during stoppages,
  which would leak the label, so the ball is removed there except near the restart. Each match is
  generated with 5 random seeds; training samples a seed per crop.
- Augmentation: team swap (H↔A) and pitch mirroring in x and y.
- **No labels of any kind from the provided game**, and no human-produced data about it, were used.
  Synthetic data generated by our own code was used only to test software plumbing.

## 4. Possession rules

We did not hand-code rules; the model inherits them from the IDSSE labels, with these consequences:

- **Loose balls / passes:** IDSSE keeps the passer's team in possession until the next player touches
  the ball, so an intercepted pass flips possession at the interception, not at the kick.
- **Challenges:** IDSSE switches at contact (≈ 4.4 spells / min, 26–39 % of switches come back within 3 s).
  The competition requires control, not a touch. The Viterbi penalty suppresses short flips: on IDSSE we
  predict ≈ 271 spells per match against ≈ 427 in the IDSSE labels, which we expect to be closer to the
  competition definition. We did not tune this on the provided game.
- **Restarts and stoppages:** `D` is learned from DFL's `BallStatus` (ball out of play), so a stoppage
  runs from the ball going dead to the restart; possession after the restart follows IDSSE.
- Frames outside play (before kick-off, half-time, after the final whistle) still receive H/A/D; they
  are not graded.

## 5. Validation

Leave-one-match-out over the 6 non-held-out IDSSE matches: in each fold, 4 matches train, 1 selects
the epoch and the Viterbi penalty (by spellF1), and 1 is tested. Mean of degradation seeds 0 and 1.

| method | spellF1 | macroF1 | F1 H | F1 A | F1 D |
|---|---|---|---|---|---|
| **TCN + Viterbi** | **0.552 ± 0.042** | 0.905 ± 0.009 | 0.870 | 0.860 | 0.983 |
| TCN, per-step argmax | 0.536 ± 0.041 | 0.905 ± 0.009 | 0.871 | 0.861 | 0.983 |
| official baseline smooth50 | 0.267 ± 0.021 | 0.593 ± 0.021 | 0.663 | 0.618 | 0.496 |

TCN + Viterbi beats the baseline in every fold (per-fold results: `weights/tcn_v1/WYNIKI.md`). For
reference, smooth50 scores 0.243 spellF1 on the provided game according to the organisers, close to
its 0.267 here, which suggests the degradation is roughly as hard as the real data. **These are scores
on public IDSSE with simulated errors, not competition performance.**

## 6. Limitations and next steps

- **Domain gap.** The degradation is calibrated on two public numbers; the true error structure of the
  provided ATD is unknown. Out of play, the real ball is visible ~48 % of frames; we only partly
  simulate that, so `D` may be over-confident on real data (0.98 F1 here is optimistic).
- **Label definition.** Trained on touch-based IDSSE possession, while the competition counts control.
  With more time: merge brief spells in the training labels (`--merge-seconds`) and choose the threshold
  on IDSSE.
- **Unused assets.** No video, homography or Smart Tagging. Ball-in-air and set-piece cues from video are
  the most promising additions.
- **Resolution.** 5 fps bins limit boundary precision to ±0.2 s.
- **Single league / camera** in training (Bundesliga, optical multi-camera tracking).
- The held-out IDSSE match was not evaluated before the deadline.

## 7. Reproduction

Tested on Windows 11, Python 3.12.10, PyTorch 2.8.0+cpu (all versions pinned in `requirements.txt` and
`requirements-nn.txt`).

```powershell
./setup.ps1                                            # creates .venv with requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-nn.txt
.\.venv\Scripts\python.exe -X utf8 -m scripts.predict_tcn `
    --atd-xml <data>\BAR-ATH_tactical_atd.xml --atd-raw <data>\BAR-ATH_tactical_atd.txt `
    --out Ctrl_Alt_Elite.csv --threads 4
```

The script uses the published weights in `weights/tcn_v1/` (SHA-256 in `SHA256SUMS`), runs in about
30 s on CPU, is deterministic on a fixed machine and thread count (two runs give byte-identical
files) and finishes with the official validation. SHA-256 of the submitted CSV (`Ctrl_Alt_Elite.csv`): `c5d46c8af5513e597564700ce02cd22b8c8b93ee2d8f51a2b37d8e608efae576`.

Retraining from scratch (≈ 1 h on CPU, not bit-identical to the published weights):

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\fetch_idsse.py
foreach ($m in 'J03WMX','J03WOH','J03WOY','J03WPY','J03WQQ','J03WR9') { foreach ($s in 0..4) {
  .\.venv\Scripts\python.exe -X utf8 -m scripts.degrade_idsse --match $m --seed $s } }
.\.venv\Scripts\python.exe -X utf8 -m scripts.train_tcn cv --run v1
```
