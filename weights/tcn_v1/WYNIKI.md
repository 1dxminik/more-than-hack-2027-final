# Leave-one-match-out: v1

Data: degraded IDSSE (public Bundesliga matches, CC BY 4.0). **These are not competition scores.**
Each result is the mean over seeds [0, 1] of the test match. Detailed scores
(`wyniki.csv`, per seed), epoch records (`epochs.csv`) and logs (`log.txt`) are
local training artifacts and are not included in this public repository.

| fold | test | validation | epoch | penalty | spellF1 TCN+Viterbi | spellF1 TCN without Viterbi | spellF1 baseline50 | macroF1 TCN | F1 D TCN | possessions pred/GT |
|---|---|---|---|---|---|---|---|---|---|---|
| cv1_J03WMX | J03WMX | J03WOH | 18 | 1 | 0.4972 | 0.4879 | 0.2498 | 0.9064 | 0.9834 | 271/511 |
| cv2_J03WOH | J03WOH | J03WOY | 15 | 1 | 0.5189 | 0.5053 | 0.2760 | 0.8967 | 0.9813 | 246/402 |
| cv3_J03WOY | J03WOY | J03WPY | 18 | 1 | 0.5165 | 0.4943 | 0.2335 | 0.8890 | 0.9818 | 287/456 |
| cv4_J03WPY | J03WPY | J03WQQ | 20 | 1 | 0.5800 | 0.5769 | 0.2751 | 0.9100 | 0.9860 | 268/421 |
| cv5_J03WQQ | J03WQQ | J03WR9 | 20 | 1 | 0.6011 | 0.5783 | 0.2991 | 0.9111 | 0.9820 | 290/401 |
| cv6_J03WR9 | J03WR9 | J03WMX | 18 | 1 | 0.5963 | 0.5742 | 0.2667 | 0.9143 | 0.9843 | 266/370 |

## Mean over 6 folds (± standard deviation across folds)

| method | spell_f1_mean | macro_f1 | f1_H | f1_A | f1_D | n_pred_spells | n_gt_spells |
|---|---|---|---|---|---|---|---|
| tcn_viterbi | 0.5517 ± 0.0419 | 0.9046 ± 0.0089 | 0.8702 ± 0.0260 | 0.8604 ± 0.0198 | 0.9831 ± 0.0016 | 271.1 | 426.8 |
| tcn_argmax | 0.5362 ± 0.0406 | 0.9052 ± 0.0093 | 0.8710 ± 0.0267 | 0.8614 ± 0.0188 | 0.9831 ± 0.0017 | 344.3 | 426.8 |
| baseline_smooth50 | 0.2667 ± 0.0208 | 0.5925 ± 0.0214 | 0.6633 ± 0.0481 | 0.6178 ± 0.0616 | 0.4964 ± 0.0183 | 562.2 | 426.8 |
