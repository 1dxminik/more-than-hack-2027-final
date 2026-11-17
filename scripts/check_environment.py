"""Check the small pinned environment without installing or downloading anything."""

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def main():
    problems = []
    if sys.version_info < (3, 11):
        problems.append("Python 3.11+ is required")
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        package, wanted = line.split("==")
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = "missing"
        if actual != wanted:
            problems.append(f"{package}: {actual}; required: {wanted}")
    if problems:
        print("\n".join(problems))
        return 1
    import numpy  # noqa: F401 - also check that imports actually work
    import pandas  # noqa: F401
    print("Environment OK: Python " + sys.version.split()[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
