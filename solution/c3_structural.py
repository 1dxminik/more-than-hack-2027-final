"""C3 fixed observation-only rules and quadratic meta features; half-local."""
from dataclasses import dataclass, asdict
import numpy as np
from . import c2_structural as c2

@dataclass(frozen=True)
class Config:
    plateau_threshold: float = 0.8
    plateau_len: int = 65
    plateau_window: int = 75
    min_D_len: int = 40
    delta_event: float = 0.12
    contact_m: float = 1.8
    prominence: float = 0.12
    event_cost: float = 6.0
    flight_cost: float = 14.0
    D_cost: float = 8.0

def mean_window(x, left, right):
    x = np.asarray(x, dtype=float)
    n = len(x)
    cs = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    a = np.maximum(0, np.arange(n) - left)
    z = np.minimum(n, np.arange(n) + right + 1)
    return (cs[z] - cs[a]) / (z - a).reshape((-1,) + (1,) * (x.ndim - 1))

def context(trace, prob, x, config=Config()):
    d = c2.evidence(trace, prob)
    n = len(prob)
    cond = prob[:, :2] / np.maximum(prob[:, :2].sum(1, keepdims=True), 1e-08)
    before = mean_window(cond, 25, 0)
    after = mean_window(cond, 0, 24)
    delta = after - before
    low = (x[:, 8] < 0.1) & ((x[:, 22] + x[:, 34]) / 2 < 0.7)
    pd50 = mean_window(prob[:, 2], 24, 25)
    pd100 = mean_window(prob[:, 2], 49, 50)
    plateau = c2.long_runs(mean_window(prob[:, 2], config.plateau_window // 2, config.plateau_window // 2) >= config.plateau_threshold, config.plateau_len)
    plateau &= (pd50 >= config.plateau_threshold) | (pd100 >= config.plateau_threshold - 0.05) & low
    distances = x[:, [14, 26]] * 30
    contact = (distances <= config.contact_m) & (x[:, [17, 29]] > 0.5) & (x[:, 2, None] > 0.5)
    alignment = x[:, 4, None] * x[:, [18, 30]] + x[:, 5, None] * x[:, [19, 31]] >= 0
    hard = np.column_stack([d['physical'] & (d['raw'] == side) for side in (0, 1)]) & contact & alignment & (x[:, 7, None] > 0.5)
    events = np.zeros((n, 2), bool)
    for side in (0, 1):
        anchors = d['anchors'][1 - side, side]
        accepted = anchors[(delta[anchors, side] > config.delta_event) | hard[anchors, side]]
        for off in range(-4, 5):
            ids = accepted + off
            events[ids[(ids >= 0) & (ids < n)], side] = True
    d.update(prob=prob, cond=cond, x=x, plateau=plateau, low_activity=low, pd50=pd50, pd100=pd100, contact=contact, hard=hard, events=events)
    return d

def decode(data, prob=None, config=Config()):
    p = data['prob'] if prob is None else prob
    n = len(p)
    e = np.log(np.maximum(p, 1e-08))
    raw = data['raw']
    plateau = data['plateau']
    for k in (0, 1):
        e[(raw == k) & data['active'] & ~plateau, k] += 1.0
    e[data['physical'] & data['inside'] & ~plateau, 2] -= 8.0
    e[data['support'] | plateau, 2] += 4.0
    cost = np.array([[0.0, config.flight_cost, config.D_cost], [config.flight_cost, 0.0, config.D_cost], [config.D_cost, config.D_cost, 0.0]])
    back = np.zeros((n, 3), np.uint8)
    value = e[0].copy()
    for i in range(1, n):
        cost[0, 1] = config.event_cost if data['events'][i, 1] else config.flight_cost
        cost[1, 0] = config.event_cost if data['events'][i, 0] else config.flight_cost
        candidates = value[:, None] - cost
        back[i] = candidates.argmax(0)
        value = candidates.max(0) + e[i]
        value -= value.max()
    out = np.empty(n, np.uint8)
    out[-1] = value.argmax()
    for i in range(n - 1, 0, -1):
        out[i - 1] = back[i, out[i]]
    return postprocess(out, data, config)

def postprocess(path, data, config=Config()):
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
            speed = np.linalg.norm(data['x'][:, 4:6], axis=1) * 20
            jump = speed - mean_window(speed, 10, 0)
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

def event_distance(mask):
    n = len(mask)
    idx = np.arange(n)
    before = np.maximum.accumulate(np.where(mask, idx, -250))
    after = np.minimum.accumulate(np.where(mask, idx, n + 250)[::-1])[::-1]
    return (np.minimum(idx - before, 250) / 250, np.minimum(after - idx, 250) / 250)

def meta_features(data, trace):
    p = data['prob']
    logp = np.log(np.maximum(p, 1e-06))
    columns = [logp]
    for w in (12, 25, 50, 100):
        columns.extend([mean_window(logp, w, w), mean_window(logp, 0, w) - mean_window(logp, w, 0)])
    flags = np.column_stack([data['physical'], data['active'], data['raw'] == 2, trace['restart_side'] >= 0, ~data['active'], data['plateau'], data['low_activity']]).astype(float)
    columns.append(flags)
    for mask in (data['physical'], data['raw'] == 2, trace['restart_side'] >= 0, ~data['active']):
        columns.extend([v[:, None] for v in event_distance(mask)])
    columns.extend([np.minimum(trace['age'], 250)[:, None] / 250, data['x']])
    core = np.column_stack([logp, flags, data['x'][:, [2, 6, 14, 26]]])
    a, b = np.triu_indices(core.shape[1])
    columns.append(core[:, a] * core[:, b])
    result = np.concatenate(columns, axis=1).astype(np.float32)
    assert np.isfinite(result).all()
    return result
