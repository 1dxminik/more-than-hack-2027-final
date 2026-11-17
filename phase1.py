"""C5 checks and end-to-end runs on newly generated independent synthetics only."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import filecmp
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid

from solution.assets import ROOT, digest, verify as verify_assets


def verify_official():
    manifest = json.loads((ROOT / "official/UPSTREAM.json").read_text(encoding="utf-8"))
    for name, row in manifest["files"].items():
        data = (ROOT / "official" / name).read_bytes()
        blob = b"blob " + str(len(data)).encode() + b"\0" + data
        if hashlib.sha1(blob).hexdigest() != row["git_blob_sha1"]:
            raise ValueError("Pinned official source changed: " + name)
    return len(manifest["files"])


def check():
    from importlib.metadata import version
    for filename in ("requirements.txt", "requirements-nn.txt"):
        for line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith(("#", "-")):
                continue
            package, expected = line.strip().split("==")
            if version(package) != expected:
                raise ValueError("Dependency version differs: " + package)
    count = verify_official()
    provenance = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, row in provenance["files"].items():
        if digest(ROOT / name) != row["sha256"]:
            raise ValueError("Migrated source differs from its manifest: " + name)
    verify_assets()
    subprocess.run([sys.executable, "-s", "-X", "utf8", "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, check=True)
    print("C5_CHECK_PASS", count, "official files")


def synthetic_run(frames):
    from scripts.synthetic import generate, verify
    verify_official()
    before = {name: digest(path) for name, path in verify_assets().items()}
    run = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    inputs = ROOT / "data/synthetic-c5" / run
    outputs = ROOT / "outputs/synthetic-c5" / run
    outputs.mkdir(parents=True, exist_ok=False)
    generate(inputs, frames)
    env = os.environ.copy()
    env["MTAH_DATA_DIR"] = str(inputs)
    env["PYTHONUTF8"] = "1"
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    elapsed = []
    for index in (1, 2):
        start = time.perf_counter()
        subprocess.run([sys.executable, "-s", "-X", "utf8", "-m", "scripts.synthetic_worker",
                        "--run", run, "--pass-number", str(index)], cwd=ROOT, env=env, check=True)
        elapsed.append(time.perf_counter() - start)
        print(f"SYNTHETIC_PASS_{index}: {elapsed[-1]:.2f}s", flush=True)
    a, b = outputs / "pass-1.csv", outputs / "pass-2.csv"
    if not filecmp.cmp(a, b, shallow=False):
        raise ValueError("Fresh synthetic exports differ")
    if before != {name: digest(path) for name, path in verify_assets().items()}:
        raise ValueError("Model assets changed during synthetic execution")
    verify(inputs)
    report = dict(kind="wholly_synthetic", competition_data=False, frames=frames,
                  slots=46, processes=2, byte_identical=True, format_pass=True,
                  csv_sha256=digest(a), seconds=elapsed)
    (outputs / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("C5_SYNTHETIC_PASS", run, frames, "frames; byte-identical exports", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    commands.add_parser("demo")
    commands.add_parser("improve")
    args = parser.parse_args()
    if args.command == "check":
        check()
    else:
        synthetic_run(6000 if args.command == "demo" else 199319)


if __name__ == "__main__":
    main()
