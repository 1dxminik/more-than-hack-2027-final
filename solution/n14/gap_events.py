"""Additional event evidence across bounded one/two-frame ball gaps.

Interpolated positions are marked in the feature contract. Original observed
events remain unchanged. Maximum finalization delay: six frames at 25 fps.
"""
from dataclasses import replace
import math
import numpy as np
from . import event_control as event
from . import evidence_control as evidence
from .dribble_evidence import RELATIVE_SPEED_MPS
from .possession import Config, observations
MAX_GAP = 2
COMMON_FEATURES = (*event.COMMON_FEATURES, 'missing_window', 'missing_pre', 'missing_post', 'missing_center', 'gap_length', 'bridge_speed')

def interpolate(ball, fps=25.0, pitch=(105.3, 68.0)):
    """Only bounded gaps; preserve all observed values and longer outages."""
    original = np.asarray(ball, float)
    filled = original.copy()
    n = len(filled)
    finite = np.isfinite(original).all(axis=1)
    dependency = np.full(n, -1, int)
    length = np.zeros(n, int)
    speed = np.zeros(n)
    i = 0
    while i < n:
        if finite[i]:
            i += 1
            continue
        stop = i + 1
        while stop < n and (not finite[stop]):
            stop += 1
        size = stop - i
        if i > 0 and stop < n and (size <= MAX_GAP):
            for j in range(i, stop):
                alpha = (j - i + 1) / (size + 1)
                filled[j] = (1 - alpha) * original[i - 1] + alpha * original[stop]
            dependency[i:stop] = stop
            length[i:stop] = size
            speed[i:stop] = np.linalg.norm((original[stop] - original[i - 1]) * pitch) * fps / (size + 1)
        i = stop
    return (filled, ~finite, dependency, length, speed)

def lookahead_frames(fps=25.0, config=None):
    return event.lookahead_frames(fps, config) + MAX_GAP

def features(players, ball, teams, fps=25.0, pitch=(105.3, 68.0), config=None):
    config = config or Config()
    observations(players, ball, teams, fps, pitch, config)
    filled, missing, dependency, length, speed = interpolate(ball, fps, pitch)
    p = np.asarray(players, float) * pitch
    b = filled * np.asarray(pitch)
    teams = np.asarray(teams)
    distance = np.linalg.norm(p - b[:, None], axis=2)
    width = math.ceil(0.12 * fps)
    result = []
    axis = np.arange(width + 1) - width / 2
    coef = axis / (axis @ axis) * fps
    anchors = np.flatnonzero((length > 0) & np.r_[True, length[:-1] == 0])
    for e in anchors:
        e = int(e)
        window = slice(e - width - 1, e + width + 2)
        if e < width + 1 or e + width + 1 >= len(b) or (not np.isfinite(b[window]).all()):
            continue
        local = distance[e - width:e + width + 1]
        if not np.any(np.isfinite(local) & (local <= config.control_radius_m)):
            continue
        pre = coef @ b[e - width:e + 1]
        post = coef @ b[e:e + width + 1]
        incoming = np.linalg.norm(pre)
        outgoing = np.linalg.norm(post)
        impulse_size = np.linalg.norm(post - pre)
        per_team = []
        for side in (0, 1):
            slots = np.flatnonzero(teams == side)
            ds = local[:, slots]
            safe = np.where(np.isfinite(ds), np.minimum(ds, 12.0), 12.0)
            mins = safe.min(axis=1) if len(slots) else np.full(2 * width + 1, 12.0)
            relative_in = []
            relative_out = []
            impulses = []
            for j in slots:
                path = p[e - width:e + width + 1, j]
                if distance[e, j] > config.control_radius_m or not np.isfinite(path).all():
                    continue
                if np.any(np.linalg.norm(np.diff(path, axis=0), axis=1) * fps > config.max_player_speed_mps):
                    continue
                vin = pre - (p[e, j] - p[e - width, j]) * fps / width
                vout = post - (p[e + width, j] - p[e, j]) * fps / width
                relative_in.append(np.linalg.norm(vin))
                relative_out.append(np.linalg.norm(vout))
                impulses.append(np.linalg.norm(vout - vin))
            per_team.append(np.asarray([mins[width], mins[0], mins[-1], mins[width - 1], mins[width + 1], mins.min(), mins[:width + 1].mean(), mins[width:].mean(), np.mean(mins <= config.control_radius_m), np.sum(ds[width] <= config.control_radius_m), np.log1p(min(relative_in, default=40.0)), np.log1p(min(relative_out, default=40.0)), np.log1p(min(impulses, default=40.0)), float(bool(relative_in))]))
        difference = per_team[0] - per_team[1]

        def straight(segment):
            return float(np.linalg.norm(segment[-1] - segment[0]) / max(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum(), 1e-08))
        finite = distance[e][np.isfinite(distance[e])]
        common = [*np.abs(difference), np.log1p(incoming), np.log1p(outgoing), np.log1p(impulse_size), float(pre @ post / max(incoming * outgoing, 1e-08)), straight(b[e - width:e + 1]), straight(b[e:e + width + 1]), min(float(finite.min()), 12.0) if len(finite) else 12.0, float(np.sum(distance[e] <= config.control_radius_m))]
        quality = [float(np.mean(missing[window])), float(np.mean(missing[e - width:e + 1])), float(np.mean(missing[e:e + width + 1])), float(missing[e]), float(np.max(length[window])), float(np.log1p(np.max(speed[window])))]
        result.append(dict(frame=e, available_at=max(e + width + 1, int(np.max(dependency[window]))), difference=difference.tolist(), common=list(map(float, [*common, *quality]))))
    return result

def decisions(rows, model):
    if not rows:
        return []
    if model['role_features'] != list(event.ROLE_FEATURES) or model['common_features'] != list(COMMON_FEATURES):
        raise ValueError('Gap-event feature contract differs')
    d = np.asarray([r['difference'] for r in rows]) / model['role_scale']
    c = np.asarray([r['common'] for r in rows]) / model['common_scale']
    side = d @ model['role_beta']
    shared = c @ model['common_beta'] + model['bias']
    logits = np.column_stack((np.zeros(len(rows)), shared + side, shared - side))
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    result = []
    for row, prob in zip(rows, probabilities):
        winners = np.flatnonzero(np.isclose(prob, prob.max(), rtol=0, atol=1e-12))
        cls = int(winners[0]) if len(winners) == 1 else 0
        result.append(dict(frame=row['frame'], available_at=row['available_at'], **{'class': cls}, probabilities=prob.tolist()))
    return result

def predict(players, ball, teams, base_model, gap_model, fps=25.0, pitch=(105.3, 68.0), config=None, prepared=None):
    config = config or Config()
    _, dead, present = observations(players, ball, teams, fps, pitch, config)
    original, additional = (event.event_features(players, ball, teams, fps, pitch, config), features(players, ball, teams, fps, pitch, config)) if prepared is None else prepared
    observed = event.decisions(original, base_model)
    inferred = decisions(additional, gap_model)
    choices = sorted([{**r, 'source': 'observed'} for r in observed] + [{**r, 'source': 'interpolated'} for r in inferred], key=lambda r: r['frame'])
    channel = np.full(len(ball), -1, int)
    for row in choices:
        if row['class']:
            channel[row['frame']] = row['class'] - 1
    motion = evidence.motion_evidence(players, ball, fps, pitch, config)
    labels, reasons = evidence.decode(*motion, teams, dead, present, channel, fps, replace(config, still_ball_speed_mps=RELATIVE_SPEED_MPS))
    return (labels, reasons, choices)
