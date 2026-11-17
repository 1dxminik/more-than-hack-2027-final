"""Incremental C3-based fusion. No fitting or competition-specific rules."""
import numpy as np
from .c3_structural import Config, mean_window
from . import c2_structural as c2

def postprocess(path, data, config=Config()):
    speed = np.linalg.norm(data['x'][:, 4:6], axis=1) * 20
    jump = speed - mean_window(speed, 10, 0)
    out = path.copy()
    n = len(out)
    counts = dict(short_D_removed=0, one_touch_removed=0, HA_aligned=0, restart_aligned=0)
    anchor = (data['raw'] == 2).copy()
    for before in (0, 1):
        anchor[data['anchors'][before, 2]] = True
        anchor[data['anchors'][2, before]] = True
    segments = c2.runs(out)
    for i, (a, z, k) in enumerate(segments):
        if k == 2 and z - a < config.min_D_len and (not anchor[a:z].any()):
            left = segments[i - 1][2] if i else 3
            right = segments[i + 1][2] if i + 1 < len(segments) else 3
            if left < 2 and right < 2:
                out[a:z] = left if left == right else int(data['cond'][a:z].mean(0).argmax())
                counts['short_D_removed'] += 1
    segments = c2.runs(out)
    for i, (a, z, k) in enumerate(segments):
        if k < 2 and z - a <= 25 and (0 < i < len(segments) - 1):
            left, right = (segments[i - 1][2], segments[i + 1][2])
            if left == right and left < 2 and (left != k):
                flank = max(float(data['cond'][max(0, a - 25):a, k].mean()), float(data['cond'][z:min(n, z + 25), k].mean()))
                prominent = float(data['cond'][a:z, k].max()) - flank >= config.prominence
                confirmed = bool((data['physical'][a:z] & (data['raw'][a:z] == k) & data['contact'][a:z, k]).any())
                if not (prominent and confirmed):
                    out[a:z] = left
                    counts['one_touch_removed'] += 1
    segments = c2.runs(out)
    bounds = [s[0] for s in segments] + [n]
    shift = [0]
    for j in range(1, len(segments)):
        t = bounds[j]
        old, new = (segments[j - 1][2], segments[j][2])
        target = t
        if old < 2 and new < 2 and (not data['contact'][t, new]):
            lo = max(shift[-1] + 1, t - 8)
            hi = min(bounds[j + 1] - 1, t + 45)
            ids = np.arange(lo, hi + 1)
            confirmed = data['contact'][ids, new] & data['physical'][ids] & (data['raw'][ids] == new)
            valid = ids[confirmed]
            if len(valid):
                target = int(valid[0])
            else:
                distances = data['x'][ids, 14 + 12 * new] * 30
                if len(ids) and np.min(distances) <= config.contact_m and (data['x'][ids[np.argmin(distances)], 2] > 0.5):
                    target = int(ids[np.argmin(distances)])
            counts['HA_aligned'] += int(target != t)
        elif old == 2 and new < 2:
            lo = max(shift[-1] + 1, t - 15)
            hi = min(bounds[j + 1] - 1, t + 30)
            ids = np.arange(lo, hi + 1)
            valid = ids[(jump[ids] > 1.0) & (data['x'][ids, 7] > 0.5)]
            if len(valid):
                target = int(valid[np.argmax(jump[valid])])
                counts['restart_aligned'] += int(target != t)
        shift.append(target)
    shift.append(n)
    result = np.empty_like(out)
    for i, (_, _, k) in enumerate(segments):
        result[shift[i]:shift[i + 1]] = k
    return (result, counts)
from . import c3_structural as c3, c4_structural as c4

def context(trace, prob, x, players, pitch):
    data = c3.context(trace, prob, x)
    confirmed = data['physical'] & data['contact'].any(1) & (data['raw'] < 2)
    for a, z, state in c3.c2.runs(data['plateau'].astype(np.uint8)):
        if state and confirmed[a:z].all():
            data['plateau'][a:z] = False
    tier = c4.context(trace, prob, x, players, pitch)
    data.update(strong_events=tier['strong_events'], weak_events=tier['weak_events'])
    return data

def emissions(data, meta, cfg):
    alpha = cfg['alpha']
    e = np.log(np.maximum(data['prob'], 1e-08)).astype(float)
    if alpha:
        e = (1 - alpha) * e + alpha * np.log(np.maximum(meta, 1e-08))
    plateau = data['plateau']
    raw = data['raw']
    for k in (0, 1):
        e[(raw == k) & data['active'] & ~plateau, k] += 1.0
    e[data['physical'] & data['inside'] & ~plateau, 2] -= 8.0
    e[data['support'] | plateau, 2] += 4.0
    if cfg['tier']:
        costs = np.where(data['strong_events'], cfg['strong'], np.where(data['weak_events'], cfg['weak'], 14.0))
    else:
        costs = np.where(data['events'], 6.0, 14.0)
    return (e, costs)

def viterbi(e, costs):
    n = len(e)
    back = np.zeros((n, 3), np.uint8)
    value = e[0].copy()
    for i in range(1, n):
        matrix = np.array([[0.0, costs[i, 1], 8.0], [costs[i, 0], 0.0, 8.0], [8.0, 8.0, 0.0]])
        candidates = value[:, None] - matrix
        back[i] = candidates.argmax(0)
        value = candidates.max(0) + e[i]
        value -= value.max()
    out = np.empty(n, np.uint8)
    out[-1] = value.argmax()
    for i in range(n - 1, 0, -1):
        out[i - 1] = back[i, out[i]]
    return out

def decode(data, meta, cfg, kernel=viterbi):
    e, costs = emissions(data, meta, cfg)
    return postprocess(kernel(e, costs), data, c3.Config(min_D_len=cfg['min_D_len']))
