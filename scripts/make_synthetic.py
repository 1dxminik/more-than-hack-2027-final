"""Generate a deterministic interface fixture, never data from a real match."""

import hashlib
import json
import math
from pathlib import Path


TRACKING = "tactical/BAR-ATH_tactical_atd.txt"
METADATA = "tactical/BAR-ATH_tactical_atd.xml"
GROUND_TRUTH = "synthetic_gt.csv"
SCORE_METADATA = "synthetic_gt_meta.json"
MARKER = "synthetic_manifest.json"
FILES = (TRACKING, METADATA, GROUND_TRUTH, SCORE_METADATA)
GENERATOR_VERSION = 2


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(root, relative):
    """Reject links/junctions and paths that escape the owned synthetic directory."""
    root = Path(root).absolute()
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Synthetic path must stay inside its directory")
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("Links and junctions are not allowed in synthetic paths")
    return path


def verify_bundle(root):
    marker_path = local_path(root, MARKER)
    metadata = json.loads(marker_path.read_text(encoding="utf-8"))
    if metadata.get("kind") != "wholly_synthetic" or metadata.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("Unrecognized synthetic manifest; refusing to read inputs")
    for relative in FILES:
        if sha256(local_path(root, relative)) != metadata["sha256"][relative]:
            raise ValueError("Synthetic input changed: " + relative)
    return metadata


def generate(root, n_frames=199319):
    """Create fixed synthetic coordinates and 1-based raw ATD frame numbers.

    Only the minimal XML consumed by the official baseline is generated. It is
    intentionally not a full EPTS fixture for kloppy. Names follow games.py's
    interface; no values are obtained from a real BAR-ATH match or public GT.
    """
    if n_frames < 1000:
        raise ValueError("Use at least 1000 frames for this fixture")
    root = Path(root)
    marker_path = local_path(root, MARKER)
    if marker_path.exists():
        metadata = verify_bundle(root)
        if metadata["n_frames"] != n_frames:
            raise ValueError("Existing synthetic bundle has a different frame count")
        return metadata
    for relative in FILES:
        if local_path(root, relative).exists():
            raise ValueError("Refusing to overwrite unrecognized input: " + relative)

    tracking_path = local_path(root, TRACKING)
    tracking_path.parent.mkdir(parents=True, exist_ok=True)
    local_path(root, METADATA).write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<SyntheticBaselineFixture>\n'
        '  <Team id="synthetic-home"><Name>SYNTHETIC HOME</Name></Team>\n'
        '  <Team id="synthetic-away"><Name>SYNTHETIC AWAY</Name></Team>\n'
        '  <Player id="1" teamId="synthetic-home" />\n'
        '  <Player id="2" teamId="synthetic-home" />\n'
        '  <Player id="3" teamId="synthetic-away" />\n'
        '  <Player id="4" teamId="synthetic-away" />\n'
        '</SyntheticBaselineFixture>\n', encoding="utf-8", newline="\n"
    )
    players = "0.30,0.50,0;0.42,0.75,0;0.70,0.50,0;0.58,0.25,0"
    counts = dict.fromkeys("HADX", 0)
    missing_live = 0
    n_segments = 0
    previous = "X"
    with tracking_path.open("w", encoding="utf-8", newline="\n") as atd, \
            local_path(root, GROUND_TRUTH).open("w", encoding="utf-8", newline="\n") as gt:
        gt.write("frame,label\n")
        for frame in range(n_frames):
            # Arbitrary synthetic gaps and states, independent of match timing.
            ungraded = frame < 100 or frame >= n_frames - 100 or n_frames // 2 <= frame < n_frames // 2 + 100
            state = "X" if ungraded else ("H", "H", "A", "D", "H", "A")[(frame // 400) % 6]
            n_segments += int(state != "X" and state != previous)
            previous = state
            counts[state] += 1
            missing = state in "DX" or 90 <= frame % 233 < 115
            if missing:
                ball = "NaN,NaN,NaN"
                missing_live += int(state in "HA")
            else:
                observed = state
                # Brief misleading observations create a reason for smoothing.
                if frame % 187 < 3:
                    observed = "A" if state == "H" else "H"
                x = (0.302 if observed == "H" else 0.698) + 0.001 * math.sin(frame / 23)
                y = 0.5 + 0.002 * math.sin(frame / 19)
                ball = f"{x:.6f},{y:.6f},0"
            atd.write(f"{frame + 1}:{players}:{ball}\n")
            gt.write(f"{frame},{state}\n")

    score_metadata = {"kind": "wholly_synthetic", "n_video_frames": n_frames,
                      "n_graded_frames": n_frames - counts["X"], "n_segments": n_segments}
    local_path(root, SCORE_METADATA).write_text(json.dumps(score_metadata, indent=2) + "\n",
                                               encoding="utf-8", newline="\n")
    metadata = {
        "kind": "wholly_synthetic",
        "generator_version": GENERATOR_VERSION,
        "n_frames": n_frames,
        "fps": 25,
        "labels": counts,
        "missing_ball_on_live_frames": missing_live,
        "warning": "Interface test only. Not competition data or a competition submission.",
        "sha256": {relative: sha256(local_path(root, relative)) for relative in FILES},
    }
    marker_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    return metadata
