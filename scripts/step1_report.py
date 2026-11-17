"""Post-prediction diagnostics. These labels never enter the inference algorithm."""

import json

import numpy as np
import pandas as pd

from scripts.make_synthetic import GROUND_TRUTH, local_path
from scripts.make_step1_synthetic import CASE_FILE


def compare_cases(synthetic, out, variants, own_labels, reasons):
    cases = json.loads(local_path(synthetic, CASE_FILE).read_text(encoding="utf-8"))
    gt = pd.read_csv(local_path(synthetic, GROUND_TRUTH))["label"].to_numpy()
    predictions = {v: pd.read_csv(out / "submissions" / (v + ".csv"))["label"].to_numpy() for v in variants}
    by_case = {}
    for case in cases:
        start, stop, name = case["start"], case["stop"], case["name"]
        entry = by_case.setdefault(name, {"n_frames": 0, "variants": {}})
        entry["n_frames"] += stop - start
        truth = gt[start:stop]
        graded = truth != "X"
        for variant, prediction in predictions.items():
            frame = prediction[start:stop]
            values = entry["variants"].setdefault(variant, {"graded": 0, "correct": 0, "false_D": 0, "X": 0})
            values["graded"] += int(graded.sum())
            values["correct"] += int(((frame == truth) & graded).sum())
            values["false_D"] += int(((frame == "D") & (truth != "D") & graded).sum())
            values["X"] += int(((frame == "X") & graded).sum())
    labels, counts = np.unique(own_labels, return_counts=True)
    reason_names, reason_counts = np.unique(reasons, return_counts=True)
    report = {
        "kind": "wholly_synthetic", "not_for_submission": True,
        "note": "Frame-level diagnostic counts. X is penalized in macroF1 and filled by the official temporal scorer.",
        "limitations": ["Stationary players plus missing ball can also be live play.",
                        "A turnover entirely hidden during an occlusion is not observable.",
                        "A dead ball near a stationary player can be mistaken for control."],
        "labels": dict(zip(labels.tolist(), counts.tolist())),
        "reasons": dict(zip(reason_names.tolist(), reason_counts.tolist())),
        "cases": by_case,
    }
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report
