"""Actual-time support for bounded intermittent observations; no filling or IO."""
import math
import numpy as np
from .possession import Config

def limits(fps=25.0, config=None):
    c = config or Config()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Positive fps required')
    gap = 2 * math.ceil(0.12 * fps)
    window = max(2, math.ceil(c.dead_confirm_seconds * fps))
    return dict(max_sample_gap_frames=gap, D_window_frames=window, D_min_points=math.ceil(window / 2), HA_half_window_frames=math.ceil(0.12 * fps), HA_min_points_per_side=2)

def fit(path, fps=25.0, minimum=2, max_gap=None):
    values = np.asarray(path, float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError('XY samples required')
    ids = np.flatnonzero(np.isfinite(values).all(axis=1))
    count = len(ids)
    span = int(ids[-1] - ids[0]) if count else 0
    gap = int(np.diff(ids).max()) if count > 1 else 0
    age = len(values) - 1 - int(ids[-1]) if count else len(values)
    quality = dict(count=count, span_frames=span, max_gap_frames=gap, freshness_frames=age, indices=ids)
    if count < minimum or span == 0 or (max_gap is not None and gap > max_gap):
        return (None, quality)
    time = (ids - ids.mean()) / fps
    points = values[ids]
    center = points.mean(axis=0)
    velocity = time @ (points - center) / (time @ time)
    residual = np.linalg.norm(points - center - time[:, None] * velocity, axis=1)
    return (dict(center=center, velocity=velocity, residual_rms=float(np.sqrt(np.mean(residual ** 2)))), quality)

def continuous_slot(path, fps=25.0, config=None):
    c = config or Config()
    p = np.asarray(path, float)
    return bool(np.isfinite(p).all() and (np.linalg.norm(np.diff(p, axis=0), axis=1) * fps <= c.max_player_speed_mps).all())
