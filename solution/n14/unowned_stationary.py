"""Opt-in observed rest without sustained player control; no labels or IO."""
import numpy as np
from .dead_restart import rolling_sum
from .observed_sample_contract import limits, fit
from .dribble_evidence import RELATIVE_SPEED_MPS

def combine(base, evidence, players, ball, teams, fps=25.0, pitch=(105.3, 68.0)):
    c = evidence['config']
    policy = limits(fps, c)
    width = policy['D_window_frames']
    p = np.asarray(players, float) * pitch
    b = np.asarray(ball, float) * pitch
    n = len(b)
    last_seen = np.maximum.accumulate(np.where(np.isfinite(p).all(axis=2), np.arange(n)[:, None], -1), axis=0)
    finite = np.isfinite(b).all(axis=1)
    count = rolling_sum(finite, width)
    candidates = (count >= policy['D_min_points']) & finite & evidence['usable']
    candidates[:width - 1] = False
    confirmed = np.zeros(n, bool)
    support = np.zeros(n, bool)
    available = np.full(n, -1, np.int32)
    for i in np.flatnonzero(candidates):
        start = i - width + 1
        path = b[start:i + 1]
        value, q = fit(path, fps, policy['D_min_points'], policy['max_sample_gap_frames'])
        if value is None or q['freshness_frames'] or q['span_frames'] < width - policy['max_sample_gap_frames']:
            continue
        if np.linalg.norm(value['velocity']) >= c.still_ball_speed_mps or value['residual_rms'] > c.still_ball_speed_mps * c.dead_confirm_seconds:
            continue
        ids = q['indices']
        players_window = p[start:i + 1]
        valid = np.isfinite(players_window).all(axis=(0, 2))
        speed = np.linalg.norm(np.diff(players_window, axis=0), axis=2) * fps
        valid &= np.all(speed <= c.max_player_speed_mps, axis=0)
        if valid.sum() < c.min_motion_tracks or valid.sum() < 0.75 * evidence['present'][i]:
            continue
        latest = p[np.maximum(last_seen[i], 0), np.arange(p.shape[1])]
        known_near = (last_seen[i] >= 0) & (np.linalg.norm(latest - b[i], axis=1) <= 2 * c.control_radius_m)
        if np.any(known_near & ~valid):
            continue
        relative = players_window[ids] - path[ids, None]
        distance = np.linalg.norm(relative, axis=2)
        nearby = np.any(np.isfinite(distance) & (distance <= 2 * c.control_radius_m), axis=0)
        if not nearby.any() or np.any(nearby & ~valid):
            continue
        relative_speed = np.linalg.norm(np.diff(relative, axis=0), axis=2) * fps / np.diff(ids)[:, None]
        controlled = valid & np.all(distance <= c.control_radius_m, axis=0) & np.all(relative_speed <= RELATIVE_SPEED_MPS, axis=0)
        if controlled.any():
            continue
        confirmed[i] = True
        support[start:i + 1] = True
        missing = available[start:i + 1] < 0
        available[start:i + 1][missing] = i
    masks = {k: v.copy() for k, v in base.items()}
    added = support & ~base['dead']
    masks['dead'] |= support
    masks['owner'][added] = -1
    masks['unowned_added'] = added
    masks['unowned_available_at'] = available
    masks['unowned_confirmed'] = confirmed
    return (masks, dict(confirmed=confirmed, added=added, available_at=available))
