"""Synthetic edge cases for missing-ball and dead-ball reasoning, including failures.

The inference code never receives CASES, labels, durations, or the case report.
All trajectories and timings below are invented. No real match or public GT used.
"""

import itertools
import json
import math
from pathlib import Path

from scripts.make_synthetic import (
    FILES, GENERATOR_VERSION, GROUND_TRUTH, MARKER, METADATA, SCORE_METADATA,
    TRACKING, local_path, sha256, verify_bundle,
)


SUITE = "missing_ball_and_stoppages_v1"
CASE_FILE = "synthetic_cases.json"
# name, truth, duration (frames at 25 fps), moving, ball observation, players visible
CASES = (
    ("live_home", "H", 200, True, "owner", True),
    ("short_occlusion", "H", 25, True, "missing", True),
    ("live_home_recovered", "H", 100, True, "owner", True),
    ("long_occlusion", "H", 150, True, "missing", True),
    ("live_away", "A", 200, True, "owner", True),
    ("dead_missing", "D", 125, False, "missing", True),
    ("restart_away", "A", 150, True, "owner", True),
    ("dead_visible_far", "D", 125, False, "far", True),
    ("restart_home", "H", 150, True, "owner", True),
    ("live_stationary_visible", "H", 100, False, "owner", True),
    ("all_tracking_missing", "H", 100, True, "missing", False),
    ("recovered_home", "H", 150, True, "owner", True),
    # Intentionally ambiguous counterexample: same cues as a stoppage, still live.
    ("ambiguous_live_missing", "H", 100, False, "missing", True),
    ("recovered_after_ambiguity", "H", 150, True, "owner", True),
    ("touch_opponent", "H", 3, True, "away", True),
    ("continued_home", "H", 147, True, "owner", True),
    # A turnover hidden from tracking cannot be discovered by possession memory.
    ("unobserved_change", "A", 100, True, "missing", True),
    ("recovered_away", "A", 150, True, "owner", True),
    ("ungraded_gap", "X", 150, False, "missing", False),
)


def graded_blocks(root):
    """Inclusive graded windows, the synthetic twin of the published half boundaries. No labels."""
    cases = json.loads(local_path(root, CASE_FILE).read_text(encoding="utf-8"))
    blocks = []
    for case in cases:
        if case["truth"] == "X" or case["stop"] <= case["start"]:
            continue
        if blocks and blocks[-1][1] + 1 == case["start"]:
            blocks[-1][1] = case["stop"] - 1
        else:
            blocks.append([case["start"], case["stop"] - 1])
    return [tuple(block) for block in blocks]


def verify_step1(root):
    manifest = verify_bundle(root)
    if manifest.get("suite") != SUITE:
        raise ValueError("Unrecognized Step 1 synthetic suite")
    if sha256(local_path(root, CASE_FILE)) != manifest["sha256"][CASE_FILE]:
        raise ValueError("Synthetic case report changed")
    return manifest


def generate_step1(root, n_frames=199319):
    if n_frames < 3000:
        raise ValueError("Use at least 3000 frames to include all edge cases")
    root = Path(root)
    marker = local_path(root, MARKER)
    if marker.exists():
        manifest = verify_step1(root)
        if manifest["n_frames"] != n_frames:
            raise ValueError("Existing synthetic suite has a different length")
        return manifest
    for relative in (*FILES, CASE_FILE):
        if local_path(root, relative).exists():
            raise ValueError("Refusing to overwrite unrecognized input: " + relative)
    local_path(root, TRACKING).parent.mkdir(parents=True, exist_ok=True)
    local_path(root, METADATA).write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<SyntheticBaselineFixture>\n'
        '  <Team id="home"><Name>SYNTHETIC HOME</Name></Team>\n'
        '  <Team id="away"><Name>SYNTHETIC AWAY</Name></Team>\n'
        '  <Player id="1" teamId="home" /><Player id="2" teamId="home" />\n'
        '  <Player id="3" teamId="away" /><Player id="4" teamId="away" />\n'
        '</SyntheticBaselineFixture>\n', encoding="utf-8", newline="\n")
    bases = ((0.30, 0.50), (0.42, 0.75), (0.70, 0.50), (0.58, 0.25))
    frame = 0
    motion_time = 0.0
    previous_positions = None
    last_label = "X"
    counts = dict.fromkeys("HADX", 0)
    n_segments = 0
    ranges = []
    with local_path(root, TRACKING).open("w", encoding="utf-8", newline="\n") as raw, \
            local_path(root, GROUND_TRUTH).open("w", encoding="utf-8", newline="\n") as gt:
        gt.write("frame,label\n")
        for name, label, duration, moving, ball_mode, visible in itertools.cycle(CASES):
            start = frame
            stop = min(n_frames, frame + duration)
            ranges.append({"name": name, "start": start, "stop": stop, "truth": label})
            while frame < stop:
                motion_time += 0.04 if moving else 0.0
                positions = [(x + 0.018 * math.sin(1.9 * motion_time + slot),
                              y + 0.012 * math.cos(1.3 * motion_time + slot))
                             for slot, (x, y) in enumerate(bases)]
                if visible:
                    groups = []
                    for slot, (x, y) in enumerate(positions):
                        speed = (math.hypot((x - previous_positions[slot][0]) * 105.3,
                                            (y - previous_positions[slot][1]) * 68) * 25
                                 if previous_positions is not None else float("nan"))
                        groups.append(f"{x:.7f},{y:.7f},{speed:.4f}")
                else:
                    groups = ["NaN,NaN,NaN"] * 4
                if ball_mode == "missing":
                    ball = "NaN,NaN,NaN"
                elif ball_mode == "far":
                    ball = "0.95,0.95,0"
                else:
                    carrier = 2 if label == "A" or ball_mode == "away" else 0
                    x, y = positions[carrier]
                    ball = f"{x + 0.005:.7f},{y + 0.002:.7f},NaN"
                raw.write(f"{frame + 1}:{';'.join(groups)}:{ball}\n")
                gt.write(f"{frame},{label}\n")
                counts[label] += 1
                n_segments += int(label != "X" and label != last_label)
                last_label = label
                previous_positions = positions if visible else None
                frame += 1
            if frame == n_frames:
                break
    score_meta = {"kind": "wholly_synthetic", "n_video_frames": n_frames,
                  "n_graded_frames": n_frames - counts["X"], "n_segments": n_segments}
    local_path(root, SCORE_METADATA).write_text(json.dumps(score_meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    local_path(root, CASE_FILE).write_text(json.dumps(ranges, indent=2) + "\n", encoding="utf-8", newline="\n")
    manifest = {"kind": "wholly_synthetic", "generator_version": GENERATOR_VERSION,
                "suite": SUITE, "n_frames": n_frames, "fps": 25, "labels": counts,
                "warning": "Synthetic edge cases, not a real-world benchmark or training dataset.",
                "sha256": {p: sha256(local_path(root, p)) for p in (*FILES, CASE_FILE)}}
    marker.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest
