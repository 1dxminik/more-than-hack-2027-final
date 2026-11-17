"""C2: full state emissions, event-gated transitions, short-spell protection and snapping."""
from dataclasses import dataclass, asdict
import numpy as np
FIXED = dict(c_D=8.0, event_radius=4, short_frames=35, short_lookahead=10, tau_one_touch=0.65, D_probability=0.8, D_min_frames=75, snap_D_offset=5, w_HA_unconfirmed=0.0)

def codes(labels):
    labels = np.asarray(labels)
    if not np.isin(labels, list('HADX')).all():
        raise ValueError('HADX labels required')
    return np.select([labels == 'H', labels == 'A', labels == 'D'], [0, 1, 2], default=3).astype(np.uint8)

def runs(values):
    values = np.asarray(values)
    if not len(values):
        return []
    edge = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return [(int(a), int(z), int(values[a])) for a, z in zip(edge[:-1], edge[1:])]

def long_runs(mask, width):
    out = np.zeros(len(mask), bool)
    for a, z, k in runs(np.asarray(mask, np.uint8)):
        if k and z - a >= width:
            out[a:z] = True
    return out

def evidence(trace, prob):
    raw = codes(trace['labels'])
    n = len(raw)
    prob = np.asarray(prob)
    if prob.shape != (n, 3) or not np.isfinite(prob).all() or (prob < 0).any() or (not np.allclose(prob.sum(1), 1.0, atol=1e-05)):
        raise ValueError('HAD probability contract')
    live = raw < 2
    same = trace['state'] == np.array([ord('H'), ord('A'), ord('D'), ord('X')], np.uint8)[raw]
    within = trace['age'] <= int(trace['hold'])
    physical = live & same & within & trace['owner_supported']
    reasons = trace['reasons']
    physical |= live & np.isin(reasons, ['confirmed_co_motion', 'contested_co_motion', 'observed_recovery', 'anchored_restart', 'fresh_restart_team'])
    near = np.where((reasons == 'nearby_gap_recovery') & live, raw, 3)
    for a, z, k in runs(near):
        if k < 2 and np.any(trace['coherent'][a:z] & trace['owner_supported'][a:z] & (trace['decision_reason'][a:z] == 'nearby_gap_recovery')):
            physical[a:z] = True
    active = live & (physical | same & within & ~trace['weak'])
    contact = np.full(n, -1, np.int8)
    onset = physical & np.r_[True, ~physical[:-1] | (raw[1:] != raw[:-1])]
    onset |= physical & np.isin(reasons, ['observed_recovery', 'anchored_restart'])
    contact[onset] = raw[onset].astype(np.int8)
    changed = np.r_[False, raw[1:] != raw[:-1]]
    event = changed | (contact >= 0) | (trace['restart_side'] >= 0)
    radius = FIXED['event_radius']
    near_event = np.zeros(n, bool)
    for offset in range(-radius, radius + 1):
        if offset < 0:
            near_event[:offset] |= event[-offset:]
        elif offset > 0:
            near_event[offset:] |= event[:-offset]
        else:
            near_event |= event
    sustained = long_runs(prob[:, 2] > FIXED['D_probability'], FIXED['D_min_frames'])
    support = (raw == 2) | ~active & (raw != 2) & sustained
    cond = prob[:, :2] / np.maximum(prob[:, :2].sum(1, keepdims=True), 1e-12)
    short = []
    segments = runs(raw)
    for i, (a, z, k) in enumerate(segments):
        if 0 < i < len(segments) - 1 and k < 2 and (z - a <= FIXED['short_frames']):
            before, after = (segments[i - 1][2], segments[i + 1][2])
            if before == after and before < 2 and (before != k):
                window = cond[a:min(n, z + FIXED['short_lookahead']), k]
                short.append(dict(a=a, z=z, side=k, surround=before, peak=float(window.max()), mean=float(window.mean()), one_touch=bool(cond[a:z, k].max() > FIXED['tau_one_touch'])))
    anchors = {}
    for before in range(3):
        for after in range(3):
            if before == after:
                continue
            direct = np.r_[False, (raw[:-1] == before) & (raw[1:] == after)]
            if before < 2 and after < 2:
                mask = direct | (contact == after)
            elif after == 2:
                mask = direct | trace['d_entry'] & (raw == 2)
            else:
                mask = direct | (trace['restart_side'] == after)
            anchors[before, after] = np.flatnonzero(mask)
    return dict(raw=raw, active=active, physical=physical, inside=trace['inside'], event=near_event, support=support, short=short, anchors=anchors, logp=np.log(np.maximum(prob.astype(float), 1e-08)))
