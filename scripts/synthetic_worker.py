"""One fresh synthetic export process, never accepts an external input path."""
from pathlib import Path
import argparse
import json

from solution.assets import ROOT, digest
from .predict_c5 import export
from .synthetic import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--pass-number", choices=("1", "2"), required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or "/" in args.run or "\\" in args.run:
        raise ValueError("Expected a generated run identifier")
    folder = ROOT / "data/synthetic-c5" / args.run
    meta = verify(folder)
    frames = meta["frames"]
    periods = None if frames == 199319 else [(0, frames // 2), (frames // 2 + 1, frames)]
    output = ROOT / "outputs/synthetic-c5" / args.run / ("pass-" + args.pass_number + ".csv")
    export(folder / "tactical/synthetic_atd.xml", folder / "tactical/synthetic_atd.txt", output,
           frames=frames, periods=periods)
    verify(folder)
    receipt = dict(kind="wholly_synthetic", frames=frames, periods=periods,
                   csv_sha256=digest(output), format_pass=True)
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("C5_SYNTHETIC_WORKER_PASS")


if __name__ == "__main__":
    main()
