"""Delayed current-control evidence after an acquisition NONE; no GT or IO."""
import numpy as np
from .observed_sample_contract import limits, continuous_slot
from .dribble_evidence import RELATIVE_SPEED_MPS
from .possession import Config

def choices(players, ball, teams, events, fps=25.0, pitch=(105.3, 68.0), config=None):
    c = config or Config()
    policy = limits(fps, c)
    budget = policy['max_sample_gap_frames']
    p = np.asarray(players, float) * pitch
    b = np.asarray(ball, float) * pitch
    t = np.asarray(teams)
    observed = np.isfinite(p).any(axis=2)
    first_seen = np.where(observed.any(axis=0), observed.argmax(axis=0), len(p))
    result = []
    for event in events:
        if event.get('source') != 'observed_return_impulse' or event['class'] != 0:
            continue
        e = int(event['frame'])
        end = e + budget
        if end >= len(b) or not np.isfinite(b[end]).all():
            continue
        samples = b[e:end + 1]
        ids = np.flatnonzero(np.isfinite(samples).all(axis=1))
        if len(ids) < 3 or ids[0] != 0 or np.diff(ids).max() > budget:
            continue
        sides = set()
        witnesses = []
        for j in event['support']['continuous_slots']:
            path = p[e:end + 1, j]
            if not continuous_slot(path, fps, c):
                continue
            rel = samples[ids] - path[ids]
            speeds = np.linalg.norm(np.diff(rel, axis=0), axis=1) * fps / np.diff(ids)
            if not ((np.linalg.norm(rel, axis=1) <= c.control_radius_m).all() and (speeds <= RELATIVE_SPEED_MPS).all()):
                continue
            opponents = p[e:end + 1][ids][:, (t != t[j]) & (first_seen <= end)]
            if not np.isfinite(opponents).all():
                continue
            if (np.linalg.norm(opponents - samples[ids, None], axis=2) <= c.control_radius_m).any():
                continue
            sides.add(int(t[j]))
            witnesses.append(int(j))
        if len(sides) != 1:
            continue
        result.append(dict(frame=end, available_at=end, proposal_frame=e, source='observed_recovery', **{'class': next(iter(sides)) + 1}, support=dict(indices=(e + ids).tolist(), continuous_slots=witnesses)))
    return result

def channel(events, n):
    result = np.full(n, -1, int)
    conflict = set()
    for event in events:
        i = event['frame']
        side = event['class'] - 1
        if result[i] >= 0 and result[i] != side:
            conflict.add(i)
        result[i] = side
    for i in conflict:
        result[i] = -1
    return result
