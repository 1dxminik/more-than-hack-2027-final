"""Deterministic CSV export; callers select the explicit output path."""

from pathlib import Path

import numpy as np


def write_submission(path, labels, expected_frames):
    """Validate before opening the file; write UTF-8 without BOM and LF lines."""
    if isinstance(expected_frames, bool) or not isinstance(expected_frames, (int, np.integer)) or expected_frames <= 0:
        raise ValueError("Expected a positive integer frame count")
    labels = np.asarray(labels)
    if labels.shape != (expected_frames,) or not np.isin(labels, ["H", "A", "D", "X"]).all():
        raise ValueError("Expected one H/A/D/X label per frame")
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("frame,label\n")
        stream.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
    return path
