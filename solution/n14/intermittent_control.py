"""Actual-time co-motion certificates in the existing 11-velocity window."""
import math
import numpy as np
from .observed_sample_contract import fit, limits, continuous_slot
from .dead_restart import rolling_sum
from .dribble_evidence import RELATIVE_SPEED_MPS
from .motion import WINDOW
from .possession import Config

def support(players, ball, teams, fps=25.0, pitch=(105.3, 68.0), config=None):
    c = config or Config()
    policy = limits(fps, c)
    width = WINDOW + 1
    minimum = math.ceil(c.control_confirm_seconds * fps) + 1
    p = np.asarray(players, float) * pitch
    b = np.asarray(ball, float) * pitch
    t = np.asarray(teams)
    if p.ndim != 3 or p.shape[-1] != 2 or b.shape != (len(p), 2) or (t.shape != (p.shape[1],)):
        raise ValueError('Compatible tracking required')
    result = np.full(p.shape[:2], -1, np.int32)
    finite = np.isfinite(b).all(axis=1)
    count = rolling_sum(finite, width)
    near = np.isfinite(p).all(axis=2) & (np.linalg.norm(p - b[:, None], axis=2) <= c.control_radius_m)
    near_count = rolling_sum(near, width)
    envelope = c.still_ball_speed_mps * c.dead_confirm_seconds
    for i in np.flatnonzero(finite & (count >= minimum) & (count < width) & (np.arange(len(b)) >= WINDOW)):
        start = i - WINDOW
        path = b[start:i + 1]
        for j in np.flatnonzero(near_count[i] == count[i]):
            player = p[start:i + 1, j]
            if not continuous_slot(player, fps, c):
                continue
            value, q = fit(path - player, fps, minimum, policy['max_sample_gap_frames'])
            if value is None or q['freshness_frames'] or q['span_frames'] < minimum - 1:
                continue
            if np.linalg.norm(value['velocity']) < RELATIVE_SPEED_MPS and value['residual_rms'] <= envelope:
                result[i, j] = start + int(q['indices'][0])
    return result
