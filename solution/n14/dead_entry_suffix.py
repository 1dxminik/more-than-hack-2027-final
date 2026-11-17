"""Select only the observed stationary suffix of an already confirmed D window."""
import math
import numpy as np
from . import dead_anchor

def supported_suffix_start(residual, envelope):
    """Skip one outlier only with two actual in-envelope samples per side."""
    good = np.isfinite(residual) & (residual <= envelope)
    barrier = ~good.copy()
    for j in range(2, len(residual) - 2):
        if np.isfinite(residual[j]) and (not good[j]) and good[j - 2:j].all() and good[j + 1:j + 3].all():
            barrier[j] = False
    bad = np.flatnonzero(barrier)
    return int(bad[-1]) + 1 if len(bad) else 0

def state_masks(evidence, enabled=True, robust=False):
    masks = dead_anchor.state_masks(evidence)
    n = len(masks['dead'])
    original = masks['dead'].copy()
    available = np.full(n, -1, np.int32)
    if enabled:
        c = evidence['config']
        width = max(2, math.ceil(c.dead_confirm_seconds * evidence['fps']))
        envelope = c.still_ball_speed_mps * c.dead_confirm_seconds
        for i in np.flatnonzero(masks['entered'] & evidence['stopped']):
            start = max(0, int(i) - width + 1)
            points = evidence['ball_m'][start:i + 1]
            residual = np.linalg.norm(points - evidence['mean_m'][i], axis=1)
            if robust:
                start += supported_suffix_start(residual, envelope)
            else:
                bad = np.flatnonzero(~np.isfinite(residual) | (residual > envelope))
                if len(bad):
                    start += int(bad[-1]) + 1
            masks['dead'][start:i + 1] = True
            available[start:i + 1] = i
    added = masks['dead'] & ~original
    masks['owner'][added] = -1
    masks['entry_backdated'] = added
    masks['entry_available_at'] = available
    return masks
