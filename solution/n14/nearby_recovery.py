"""Existing sustained-nearby team proof after a recent ball observation gap."""
import math
import numpy as np
from .motion import WINDOW
from .possession import Config

def channel(players, ball, teams, fps=25.0, pitch=(105.3, 68.0), config=None, coherent=False):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Positive fps required')
    c = config or Config()
    confirm = math.ceil(c.control_confirm_seconds * fps)
    p = np.asarray(players, float) * pitch
    b = np.asarray(ball, float) * pitch
    t = np.asarray(teams)
    if p.ndim != 3 or b.shape != (len(p), 2) or t.shape != (p.shape[1],):
        raise ValueError('Compatible tracking required')
    near = np.isfinite(p).all(axis=2) & (np.linalg.norm(p - b[:, None], axis=2) <= c.control_radius_m)
    distance = np.linalg.norm(p - b[:, None], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    nearest = np.argmin(distance, axis=1)
    envelope = c.still_ball_speed_mps * c.dead_confirm_seconds
    sides = np.any(near[:, t == 0], axis=1).astype(np.uint8) + 2 * np.any(near[:, t == 1], axis=1).astype(np.uint8)
    result = np.full(len(b), -1, int)
    starts = np.full(len(b), -1, int)
    pending = 0
    count = 0
    missing = -WINDOW - 1
    for i, side in enumerate(sides):
        if not np.isfinite(b[i]).all():
            missing = i
        if side in (1, 2):
            count = count + 1 if side == pending else 1
            pending = int(side)
        else:
            pending = 0
            count = 0
        if count >= confirm and i - missing <= WINDOW:
            first = i - confirm + 1
            if coherent:
                ids = np.arange(first, i + 1)
                relative = b[ids] - p[ids, nearest[ids]]
                if (np.linalg.norm(relative - relative.mean(axis=0), axis=1) > envelope).any():
                    continue
            result[i] = 0 if side == 1 else 1
            starts[i] = first
    return (result, starts)
