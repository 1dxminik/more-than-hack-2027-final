"""Explicit seven-file model bundle, verified before any model is loaded."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/c5.json"
MANIFEST = ROOT / "configs/assets.json"
DEFAULT_WEIGHTS = ROOT / "weights/c5-alpha-0.1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest():
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = value["files"]
    names = [r["file"] for r in rows]
    if value.get("variant") != "C5-alpha-0.1" or len(rows) != 7 or len(set(names)) != 7:
        raise ValueError("Expected the frozen seven-file C5 asset manifest")
    if any(Path(n).name != n or "/" in n or "\\" in n for n in names):
        raise ValueError("Model asset names must be plain filenames")
    if sorted(r["role"] for r in rows) != ["linear", "meta", "tcn", "tcn", "tcn", "tcn", "tcn"]:
        raise ValueError("Unexpected C5 asset roles")
    return value


def verify(weights_dir=DEFAULT_WEIGHTS):
    root = Path(weights_dir).resolve()
    value = manifest()
    expected = {r["file"] for r in value["files"]}
    if not root.is_dir():
        raise ValueError("C5 model assets are missing; install the local model bundle first")
    actual = {p.name for p in root.iterdir() if p.name != "README.md"}
    if actual != expected:
        raise ValueError("Missing or unexpected files in the C5 weights directory")
    result = {}
    for row in value["files"]:
        path = root / row["file"]
        if not path.is_file() or path.is_symlink() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("C5 model asset failed its manifest check: " + row["file"])
        result[row["file"]] = path
    return result


def configuration():
    if digest(CONFIG) != manifest()["config_sha256"]:
        raise ValueError("Frozen C5 configuration changed")
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value["variant"]["name"] != "C5-alpha-0.1" or value["variant"]["alpha"] != 0.1:
        raise ValueError("Expected the frozen C5-alpha-0.1 configuration")
    return value
