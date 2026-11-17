"""Explicit ATD XML/TXT to C5 CSV; no input discovery and no fitting."""
from pathlib import Path
import argparse

from solution.assets import DEFAULT_WEIGHTS


def validate_csv(path, rows):
    from official.evaluation import possession_metrics as pm
    table = pm.read_label_csv(path)
    if list(table.columns) != ["frame", "label"]:
        raise ValueError("Expected exactly frame,label")
    reason = pm.validate_submission(table, rows)
    if reason:
        raise ValueError(reason)


def export(xml, raw, output, weights_dir=DEFAULT_WEIGHTS, *, frames=199319, periods=None):
    from solution.runtime import Runtime
    from solution.tracking import load_tracking
    from solution.submission import write_submission
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output filename; existing files are preserved")
    runtime = Runtime(weights_dir)
    players, ball, teams = load_tracking(xml, raw, frames)
    labels = runtime.predict_match(players, ball, teams, periods)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(output, labels, frames)
    validate_csv(output, frames)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atd-xml", required=True, type=Path)
    parser.add_argument("--atd-raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args(argv)
    export(args.atd_xml, args.atd_raw, args.out, args.weights_dir)
    print("C5_EXPORT_PASS")


if __name__ == "__main__":
    main()
