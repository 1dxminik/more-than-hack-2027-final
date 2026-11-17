"""One D confirmation from bounded actual samples; no missingness-only entry."""
import numpy as np
from . import dead_anchor
from .dead_restart import rolling_sum
from .observed_sample_contract import limits, fit

def augment(players, ball, teams, evidence, fps=25.0, pitch=(105.3, 68.0), config=None):
    c = config or evidence['config']
    policy = limits(fps, c)
    width = policy['D_window_frames']
    b = np.asarray(ball, float) * pitch
    p = np.asarray(players, float) * pitch
    n = len(b)
    finite = np.isfinite(b).all(axis=1)
    count = rolling_sum(finite, width)
    distance = np.linalg.norm(p - b[:, None], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    isolated = finite & (distance.min(axis=1) > 2 * c.control_radius_m) & (evidence['present'] >= c.min_motion_tracks)
    enough = (count >= policy['D_min_points']) & (count < width) & (rolling_sum(isolated, width) == count) & evidence['usable']
    enough[:width - 1] = False
    result = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in evidence.items()}
    confirmed = np.zeros(n, bool)
    quality = np.zeros((n, 4), np.int16)
    for i in np.flatnonzero(enough):
        value, q = fit(b[i - width + 1:i + 1], fps, policy['D_min_points'], policy['max_sample_gap_frames'])
        quality[i] = (q['count'], q['span_frames'], q['max_gap_frames'], q['freshness_frames'])
        if value is None or q['freshness_frames'] or q['span_frames'] < width - policy['max_sample_gap_frames']:
            continue
        speed = np.linalg.norm(value['velocity'])
        if speed < c.still_ball_speed_mps and value['residual_rms'] <= c.still_ball_speed_mps * c.dead_confirm_seconds:
            confirmed[i] = True
            result['stopped'][i] = True
            result['mean_m'][i] = value['center']
            result['fit_speed'][i] = speed
            result['residual_rms'][i] = value['residual_rms']
    return (result, dict(confirmed=confirmed, quality=quality))

def combine(base, evidence, players, ball, teams, fps=25.0, pitch=(105.3, 68.0)):
    augmented, detail = augment(players, ball, teams, evidence, fps, pitch)
    observed = dead_anchor.state_masks(augmented)
    masks = {k: v.copy() for k, v in base.items()}
    added = observed['dead'] & ~base['dead']
    masks['dead'] |= observed['dead']
    masks['owner'][added] = -1
    masks['observed_D_added'] = added
    detail['available_at'] = np.where(detail['confirmed'], np.arange(len(ball)), -1)
    return (masks, detail)
