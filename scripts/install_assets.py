"""Install only the seven allowlisted assets from an explicit local C5 code package."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import zipfile

from solution.assets import DEFAULT_WEIGHTS, manifest, verify


def install(source, target=DEFAULT_WEIGHTS):
    source, target = Path(source), Path(target)
    rows = manifest()["files"]
    payloads = {}
    archive = zipfile.ZipFile(source) if source.is_file() else None
    try:
        for row in rows:
            member = row["package_member"]
            if PurePosixPath(member).is_absolute() or ".." in PurePosixPath(member).parts:
                raise ValueError("Invalid allowlisted package member")
            data = archive.read(member) if archive else (source / member).read_bytes()
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("Source asset hash mismatch: " + row["file"])
            payloads[row["file"]] = data
    finally:
        if archive:
            archive.close()
    target.mkdir(parents=True, exist_ok=True)
    for filename, data in payloads.items():
        path = target / filename
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("Existing model asset differs; refusing to overwrite it")
        else:
            with path.open("xb") as stream:
                stream.write(data)
    verify(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-package", type=Path, required=True, help="Local code-and-weights.zip or extracted code directory")
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    install(args.from_package, args.weights_dir)
    print("C5_ASSETS_PASS")


if __name__ == "__main__":
    main()
