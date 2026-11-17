"""Bound retrospective D by actual in-envelope support, never by evaluation GT."""
import numpy as np
from .observed_sample_contract import limits, fit
from .possession import Config

def bounded_support_starts(ball, confirmed, fps=25.0, pitch=(105.3, 68.0), config=None):
    c = config or Config()
    policy = limits(fps, c)
    width = policy['D_window_frames']
    b = np.asarray(ball, float) * pitch
    confirmed = np.asarray(confirmed)
    if b.ndim != 2 or b.shape[1] != 2 or confirmed.shape != (len(b),) or (confirmed.dtype != np.bool_):
        raise ValueError('XY ball and one confirmation boolean per frame required')
    result = np.full(len(b), -1, np.int64)
    envelope = c.still_ball_speed_mps * c.dead_confirm_seconds
    for i in np.flatnonzero(confirmed):
        start = i - width + 1
        if start < 0:
            raise ValueError('Complete certificate window required')
        value, q = fit(b[start:i + 1], fps, policy['D_min_points'], policy['max_sample_gap_frames'])
        if value is None or q['freshness_frames']:
            raise ValueError('Current actual certificate samples required')
        ids = q['indices']
        residual = np.linalg.norm(b[start + ids] - value['center'], axis=1)
        bad = ids[residual > envelope]
        result[i] = min(i, start + int(bad[-1]) + 1) if len(bad) else start
    return result
