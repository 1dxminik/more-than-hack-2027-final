"""Download the public IDSSE dataset at a pinned revision. It never touches competition data.

    python scripts/fetch_idsse.py

Bassek et al. 2025, https://doi.org/10.1038/s41597-025-04505-y, CC BY 4.0, data owner DFL.
Files land in the gitignored datasets/idsse/. Existing files are kept when their hash matches.
"""

import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request


ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "datasets" / "idsse"
REPOSITORY = "https://huggingface.co/api/datasets/pysport/idsse-data"
DOWNLOAD = "https://huggingface.co/datasets/pysport/idsse-data/resolve/{revision}/{name}"
ALLOWED = re.compile(r"DFL_(02_01_matchinformation|03_02_events_raw|04_03_positions_raw_observed)"
                     r"_DFL-COM-\d{6}_DFL-MAT-[A-Z0-9]{6}\.xml")
HEADERS = {"User-Agent": "mtah-idsse-fetch"}
MANIFEST = "manifest.json"


def read_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
        return json.load(response)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path, expected):
    partial = path.with_name(path.name + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as response, \
            partial.open("wb") as stream:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            digest.update(chunk)
            stream.write(chunk)
    if expected and digest.hexdigest() != expected:
        partial.unlink()
        raise ValueError("Hash mismatch: " + path.name)
    partial.replace(path)
    return digest.hexdigest()


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    revision = read_json(REPOSITORY)["sha"]
    tree = read_json(f"{REPOSITORY}/tree/{revision}?recursive=true")
    files = [entry for entry in tree if ALLOWED.fullmatch(entry["path"])]
    if len(files) != 21:
        raise ValueError(f"Expected 21 files for 7 matches, found {len(files)}")
    hashes = {}
    for entry in files:
        name = entry["path"]
        path = TARGET / name
        expected = (entry.get("lfs") or {}).get("oid")
        if path.exists() and path.stat().st_size == entry["size"] and (not expected or file_sha256(path) == expected):
            hashes[name] = file_sha256(path)
            print("OK    " + name, flush=True)
            continue
        print("FETCH " + name, flush=True)
        hashes[name] = download(DOWNLOAD.format(revision=revision, name=name), path, expected)
    (TARGET / MANIFEST).write_text(json.dumps({
        "source": REPOSITORY, "revision": revision, "license": "CC-BY-4.0",
        "paper": "https://doi.org/10.1038/s41597-025-04505-y", "sha256": hashes,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Done: {len(hashes)} files, revision {revision}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
