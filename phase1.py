"""Small Phase 1 command line: synthetic demo, official validation, project checks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from importlib.metadata import version

from scripts.make_synthetic import GROUND_TRUTH, generate, local_path, sha256, verify_bundle


ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT / "official"
N_FRAMES = 199319
VARIANTS = ("baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125")


def verify_official():
    """Verify exact upstream Git blob hashes (not hashes of reserialized text)."""
    manifest = json.loads((OFFICIAL / "UPSTREAM.json").read_text(encoding="utf-8"))
    for relative, info in manifest["files"].items():
        path = local_path(OFFICIAL, relative)
        data = path.read_bytes()
        blob = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
        if hashlib.sha1(blob).hexdigest() != info["git_blob_sha1"]:
            raise ValueError("Official source differs from pinned version: " + relative)
    print(f"Official source OK: {manifest['commit'][:12]} ({len(manifest['files'])} files)", flush=True)
    return manifest


def metrics():
    # The upstream library does not load any match data when imported.
    evaluation = str(OFFICIAL / "evaluation")
    if evaluation not in sys.path:
        sys.path.insert(0, evaluation)
    import possession_metrics
    return possession_metrics


def validate_csv(path, expected=N_FRAMES):
    """Official checks plus the documented two-column contract. No GT is read."""
    pm = metrics()
    frame = pm.read_label_csv(path)
    if list(frame.columns) != ["frame", "label"]:
        raise ValueError("Expected exactly two columns in order: frame,label")
    reason = pm.validate_submission(frame, expected)
    if reason:
        raise ValueError(reason)
    return len(frame)


def synthetic_environment(data_dir=None):
    env = os.environ.copy()
    # Override any user's pre-existing path; never print or inspect its value.
    allowed = [local_path(ROOT, name) for name in ("data/synthetic", "data/synthetic-step1")]
    selected = Path(data_dir).absolute() if data_dir is not None else allowed[0]
    if selected not in allowed:
        raise ValueError("Only the two generated synthetic datasets are allowed")
    env["MTAH_DATA_DIR"] = str(selected)
    env["PYTHONUTF8"] = "1"
    return env


def run_python(*args, log_path=None, data_dir=None):
    command = [sys.executable, "-X", "utf8", *map(str, args)]
    if log_path is None:
        return subprocess.run(command, cwd=ROOT, env=synthetic_environment(data_dir), check=True)
    result = subprocess.run(command, cwd=ROOT, env=synthetic_environment(data_dir),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8", newline="\n")
    if result.returncode:
        raise ValueError(f"Official command failed; see {log_path}")
    return result


def prepare_outputs(relative="outputs/synthetic", variants=VARIANTS, extra_outputs=()):
    out = local_path(ROOT, relative)
    marker = local_path(out, "SYNTHETIC_ONLY.json")
    if out.exists() and not marker.exists() and any(out.iterdir()):
        raise ValueError("Output folder has unrecognized files; refusing to overwrite")
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")).get("kind") != "wholly_synthetic":
            raise ValueError("Output folder is not marked synthetic")
    predictions = local_path(out, "submissions")
    expected = {name + ".csv" for name in variants}
    if predictions.exists():
        # The official scorer scans CSVs; don't let an unrelated file enter it.
        for path in predictions.iterdir():
            if path.name not in expected or not path.is_file():
                raise ValueError("Unexpected entry in synthetic submissions: " + path.name)
            local_path(predictions, path.name)
    known_outputs = ["run_manifest.json", "evaluation/scoreboard.csv", "logs/baseline.log", "logs/scorer.log"]
    known_outputs.extend(extra_outputs)
    for name in variants:
        known_outputs.extend([f"submissions/{name}.csv", f"evaluation/reports/{name}.json",
                              f"evaluation/reports/{name}_transitions.csv"])
    for relative in known_outputs:
        local_path(out, relative)
    predictions.mkdir(parents=True, exist_ok=True)
    marker.write_text('{"kind": "wholly_synthetic", "not_for_submission": true}\n',
                      encoding="utf-8", newline="\n")
    return out


def demo():
    print("SYNTHETIC DEMO ONLY - these results are not a competition score.\n", flush=True)
    source = verify_official()
    run_python(ROOT / "scripts/check_environment.py")
    synthetic = local_path(ROOT, "data/synthetic")
    data_manifest = generate(synthetic, N_FRAMES)
    verify_bundle(synthetic)
    out = prepare_outputs()
    print(f"\n1/3 Synthetic input ready: {data_manifest['n_frames']} frames.", flush=True)
    run_python(OFFICIAL / "starter/baseline.py", "--out", out / "submissions",
               log_path=out / "logs/baseline.log")
    print("\n2/3 Validating all four full-length CSV files.", flush=True)
    for name in VARIANTS:
        rows = validate_csv(out / "submissions" / (name + ".csv"))
        print(f"OK: {name}.csv - {rows} rows", flush=True)
    print("\n3/3 Official scorer, synthetic ground truth only.", flush=True)
    run_python(OFFICIAL / "evaluation/run_evaluation.py",
               "--gt", local_path(synthetic, GROUND_TRUTH),
               "--results", out / "submissions", "--out", out / "evaluation",
               "--bootstrap", "0", "--no-detail", log_path=out / "logs/scorer.log")
    print("\nSYNTHETIC SOFTWARE CHECK (not competition performance)", flush=True)
    print(f"{'variant':<22} {'macroF1':>8} {'spellF1':>8}", flush=True)
    for name in VARIANTS:
        report = json.loads((out / "evaluation/reports" / (name + ".json")).read_text(encoding="utf-8"))
        print(f"{name:<22} {report['macro_f1']:>8.4f} {report['spell_f1_mean']:>8.4f}", flush=True)
    manifest = {
        "kind": "wholly_synthetic",
        "not_for_submission": True,
        "warning": "Scores check plumbing, not football accuracy. Do not select a competition model using this demo.",
        "upstream_commit": source["commit"],
        "python": sys.version.split()[0],
        "packages": {p: version(p) for p in ("numpy", "pandas", "python-dateutil", "six", "tzdata")},
        "n_frames": N_FRAMES,
        "synthetic_input_sha256": data_manifest["sha256"],
        "submission_sha256": {name + ".csv": sha256(out / "submissions" / (name + ".csv")) for name in VARIANTS},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8", newline="\n")
    print("\nDone: outputs/synthetic/evaluation/scoreboard.csv", flush=True)
    print("Full official command output: outputs/synthetic/logs/", flush=True)
    print("SYNTHETIC ONLY: none of these files should be submitted to the competition.", flush=True)


def improve():
    from dataclasses import asdict
    from scripts.make_step1_synthetic import generate_step1, graded_blocks, verify_step1
    from scripts.step1_report import compare_cases
    from scripts.make_synthetic import METADATA, TRACKING
    from solution.decoder import DecoderConfig, decode
    from solution.possession import Config, fill_abstentions, predict
    from solution.tracking import load_tracking

    print("KROK 1: brakujaca pilka i przerwy. Tylko dane syntetyczne.\n", flush=True)
    source = verify_official()
    run_python(ROOT / "scripts/check_environment.py")
    synthetic = local_path(ROOT, "data/synthetic-step1")
    data_manifest = generate_step1(synthetic, N_FRAMES)
    verify_step1(synthetic)
    name, filled_name, decoder_name = "step1_memory_dead", "step1_filled", "segment_decoder"
    variants = (*VARIANTS, name, filled_name, decoder_name)
    out = prepare_outputs("outputs/synthetic-step1", variants, ("comparison.json",))
    print("1/4 Oficjalne baseline'y na nowym zestawie sztucznych scenariuszy.", flush=True)
    run_python(OFFICIAL / "starter/baseline.py", "--out", out / "submissions",
               log_path=out / "logs/baseline.log", data_dir=synthetic)
    print("2/4 Nasza metoda: obserwacje -> pamiec posiadania -> potwierdzanie przerw.", flush=True)
    players, ball, teams = load_tracking(local_path(synthetic, METADATA), local_path(synthetic, TRACKING), N_FRAMES)
    config = Config()
    # Inference has no access to GT or scenario definitions.
    labels, reasons = predict(players, ball, teams, config=config)
    decoder_config = DecoderConfig()
    # Only window boundaries, as in the competition where the half boundaries are published.
    blocks = graded_blocks(synthetic)
    own = {name: labels, filled_name: fill_abstentions(labels, blocks),
           decoder_name: decode(players, ball, teams, blocks=blocks, config=decoder_config)}
    for variant, values in own.items():
        with (out / "submissions" / (variant + ".csv")).open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("frame,label\n")
            stream.writelines(f"{i},{label}\n" for i, label in enumerate(values))
    print("3/4 Walidacja wszystkich plikow.", flush=True)
    for variant in variants:
        rows = validate_csv(out / "submissions" / (variant + ".csv"))
        print(f"OK: {variant}.csv - {rows} wierszy", flush=True)
    print("4/4 Ocena oficjalnym scorerem. Referencja wylacznie syntetyczna.", flush=True)
    run_python(OFFICIAL / "evaluation/run_evaluation.py", "--gt", local_path(synthetic, GROUND_TRUTH),
               "--results", out / "submissions", "--out", out / "evaluation",
               "--bootstrap", "0", "--no-detail", log_path=out / "logs/scorer.log", data_dir=synthetic)
    print(f"\n{'wariant':<24} {'macroF1':>8} {'spellF1':>8} {'F1 dla D':>9} {'X oceniane':>11}", flush=True)
    for variant in variants:
        result = json.loads((out / "evaluation/reports" / (variant + ".json")).read_text(encoding="utf-8"))
        print(f"{variant:<24} {result['macro_f1']:>8.4f} {result['spell_f1_mean']:>8.4f} "
              f"{result['per_class']['D']['f1']:>9.4f} {result['n_abstain_on_graded']:>11}", flush=True)
    compare_cases(synthetic, out, variants, labels, reasons)
    source_files = ("phase1.py", "solution/decoder.py", "solution/possession.py", "solution/tracking.py",
                    "scripts/make_step1_synthetic.py", "scripts/step1_report.py")
    manifest = {"kind": "wholly_synthetic", "not_for_submission": True,
                "upstream_commit": source["commit"], "python": sys.version.split()[0],
                "packages": {p: version(p) for p in ("numpy", "pandas", "python-dateutil", "six", "tzdata")},
                "config": asdict(config), "decoder_config": asdict(decoder_config), "n_frames": N_FRAMES,
                "warning": "Engineering checks only; thresholds not validated on real matches.",
                "our_source_sha256": {p: sha256(ROOT / p) for p in source_files},
                "synthetic_input_sha256": data_manifest["sha256"],
                "submission_sha256": {v + ".csv": sha256(out / "submissions" / (v + ".csv")) for v in variants}}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("\nGotowe: outputs/synthetic-step1/comparison.json i evaluation/scoreboard.csv", flush=True)
    print("To nie wynik konkursowy. Sprawdz tez trudne przypadki opisane w README.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: public code and synthetic data only.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Generate synthetic tracking, run official baseline, validate and score")
    commands.add_parser("improve", help="Compare Step 1 missing-ball/dead-ball logic on synthetic scenarios")
    commands.add_parser("check", help="Check upstream integrity, environment and synthetic contract tests")
    validate = commands.add_parser("validate", help="Check one explicitly selected CSV, without ground truth")
    validate.add_argument("csv", type=Path, help="CSV path (relative to your current directory, or absolute)")
    args = parser.parse_args()
    try:
        if args.command == "demo":
            demo()
        elif args.command == "improve":
            improve()
        elif args.command == "check":
            verify_official()
            run_python(ROOT / "scripts/check_environment.py")
            run_python("-m", "unittest", "discover", "-s", ROOT / "tests", "-v")
        elif args.command == "validate":
            verify_official()
            rows = validate_csv(args.csv)
            print(f"OK: {rows} rows, frame 0..{rows - 1}, labels H/A/D/X, correct columns.")
            print("File format is valid. This does not measure prediction quality.")
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
