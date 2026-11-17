"""Observation-only event channel for short team control; experimental.

NONE/H/A weights are provided explicitly. No training labels, paths or IO.
Events and the original continuous cue both finalize within four frames at 25fps.
"""
import math
import numpy as np
from .continuous_control import control_observations
from .possession import Config
ROLE_FEATURES = ('distance', 'distance_pre', 'distance_post', 'distance_before', 'distance_after', 'window_min', 'pre_mean', 'post_mean', 'near_fraction', 'near_count', 'relative_in', 'relative_out', 'relative_impulse', 'path_present')
COMMON_FEATURES = tuple(('abs_' + n for n in ROLE_FEATURES)) + ('ball_in', 'ball_out', 'ball_impulse', 'turn_cosine', 'pre_straight', 'post_straight', 'nearest_distance', 'near_count')

def lookahead_frames(fps=25.0, config=None):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive')
    return max(math.ceil(0.12 * fps) + 1, math.ceil((config or Config()).control_confirm_seconds * fps) - 1)

def event_features(players_xy, ball_xy, teams, fps=25.0, pitch=(105.3, 68.0), config=None):
    """Local ball peaks and symmetric team features, never true event times."""
    config = config or Config()
    from .possession import observations
    observations(players_xy, ball_xy, teams, fps, pitch, config)
    p = np.asarray(players_xy, float) * pitch
    b = np.asarray(ball_xy, float) * pitch
    teams = np.asarray(teams)
    n = len(b)
    width = math.ceil(0.12 * fps)
    if n < 2 * (width + 1) + 1:
        return []
    axis = np.arange(width + 1) - width / 2
    coef = axis / (axis @ axis) * fps
    pre = np.full_like(b, np.nan)
    pre[width:] = sum((coef[k] * b[k:n - width + k] for k in range(width + 1)))
    post = np.full_like(b, np.nan)
    post[:-width] = pre[width:]
    strength = np.linalg.norm(post - pre, axis=1)
    distance = np.linalg.norm(p - b[:, None], axis=2)
    rows = []
    last = -100000
    for e in range(width + 1, n - width - 1):
        if not np.isfinite(b[e - width - 1:e + width + 2]).all():
            continue
        incoming, outgoing = (np.linalg.norm(pre[e]), np.linalg.norm(post[e]))
        if max(incoming, outgoing) < 2.0 or strength[e] < 1.5:
            continue
        if strength[e] <= strength[e - 1] or strength[e] < strength[e + 1] or e - last < width + 1:
            continue
        if not np.any(np.isfinite(distance[e]) & (distance[e] <= config.control_radius_m)):
            continue
        team_features = []
        for side in (0, 1):
            slots = np.flatnonzero(teams == side)
            ds = distance[e - width:e + width + 1, slots]
            safe = np.where(np.isfinite(ds), np.minimum(ds, 12.0), 12.0)
            mins = safe.min(axis=1) if len(slots) else np.full(2 * width + 1, 12.0)
            relative_in = []
            relative_out = []
            impulse = []
            for j in slots:
                path = p[e - width:e + width + 1, j]
                if distance[e, j] > config.control_radius_m or not np.isfinite(path).all():
                    continue
                if np.any(np.linalg.norm(np.diff(path, axis=0), axis=1) * fps > config.max_player_speed_mps):
                    continue
                vin = pre[e] - (p[e, j] - p[e - width, j]) * fps / width
                vout = post[e] - (p[e + width, j] - p[e, j]) * fps / width
                relative_in.append(np.linalg.norm(vin))
                relative_out.append(np.linalg.norm(vout))
                impulse.append(np.linalg.norm(vout - vin))
            values = [mins[width], mins[0], mins[-1], mins[width - 1], mins[width + 1], mins.min(), mins[:width + 1].mean(), mins[width:].mean(), np.mean(mins <= config.control_radius_m), np.sum(ds[width] <= config.control_radius_m), np.log1p(min(relative_in, default=40.0)), np.log1p(min(relative_out, default=40.0)), np.log1p(min(impulse, default=40.0)), float(bool(relative_in))]
            team_features.append(np.asarray(values, float))
        difference = team_features[0] - team_features[1]

        def straight(segment):
            return float(np.linalg.norm(segment[-1] - segment[0]) / max(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum(), 1e-08))
        finite = distance[e][np.isfinite(distance[e])]
        common = [*np.abs(difference), np.log1p(incoming), np.log1p(outgoing), np.log1p(strength[e]), float(pre[e] @ post[e] / max(incoming * outgoing, 1e-08)), straight(b[e - width:e + 1]), straight(b[e:e + width + 1]), min(float(finite.min()), 12.0), float(np.sum(distance[e] <= config.control_radius_m))]
        rows.append({'frame': e, 'available_at': e + width + 1, 'difference': difference.tolist(), 'common': list(map(float, common))})
        last = e
    return rows

def probabilities(rows, model):
    if not rows:
        return np.empty((0, 3))
    if model['role_features'] != list(ROLE_FEATURES) or model['common_features'] != list(COMMON_FEATURES):
        raise ValueError('Event model feature contract differs')
    d = np.asarray([r['difference'] for r in rows]) / model['role_scale']
    c = np.asarray([r['common'] for r in rows]) / model['common_scale']
    role = d @ model['role_beta']
    shared = c @ model['common_beta'] + model['bias']
    logits = np.column_stack((np.zeros(len(rows)), shared + role, shared - role))
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)

def decisions(rows, model):
    result = []
    for row, prob in zip(rows, probabilities(rows, model)):
        winners = np.flatnonzero(np.isclose(prob, prob.max(), rtol=0, atol=1e-12))
        cls = int(winners[0]) if len(winners) == 1 else 0
        result.append({'frame': row['frame'], 'available_at': row['available_at'], 'class': cls, 'probabilities': prob.tolist()})
    return result

def decode(candidate, dead_evidence, n_present, event_channel, fps=25.0, config=None):
    """Fresh ordinary confirmation after an accepted event; bounded backdating."""
    config = config or Config()
    hold = math.ceil(config.hold_seconds * fps)
    dead_confirm = math.ceil(config.dead_confirm_seconds * fps)
    confirm = math.ceil(config.control_confirm_seconds * fps)
    if np.shape(event_channel) != np.shape(candidate):
        raise ValueError('Incompatible event channel')
    labels = np.full(len(candidate), 'X', dtype='<U1')
    reasons = np.full(len(candidate), 'uncertain', dtype='<U24')
    state = 'X'
    age = hold + 1
    quiet = count = 0
    pending = -1
    for i in range(len(candidate)):
        if n_present[i] == 0:
            age += 1
            quiet = count = 0
            pending = -1
            if age > hold:
                state = 'X'
            reasons[i] = 'no_player_observations'
            continue
        quiet = quiet + 1 if dead_evidence[i] else 0
        if candidate[i] >= 0:
            count = count + 1 if candidate[i] == pending else 1
            pending = int(candidate[i])
        else:
            pending = -1
            count = 0
        if quiet >= dead_confirm:
            state, age = ('D', 0)
            labels[i], reasons[i] = (state, 'sustained_dead_evidence')
        elif event_channel[i] >= 0:
            state, age = ('H' if event_channel[i] == 0 else 'A', 0)
            labels[i], reasons[i] = (state, 'inferred_control_event')
            pending = -1
            count = 0
        elif count >= confirm:
            state, age = ('H' if pending == 0 else 'A', 0)
            labels[i], reasons[i] = (state, 'confirmed_nearby_ball')
            if count == confirm:
                start = i - confirm + 1
                changed = labels[start:i] != state
                labels[start:i][changed] = state
                reasons[start:i][changed] = 'backdated_confirmation'
        else:
            age += 1
            if state != 'X' and age <= hold:
                labels[i], reasons[i] = (state, 'short_memory')
            else:
                state = 'X'
    return (labels, reasons)

def predict(players_xy, ball_xy, teams, model, fps=25.0, pitch=(105.3, 68.0), config=None, prepared=None):
    config = config or Config()
    candidate, dead, present = control_observations(players_xy, ball_xy, teams, fps, pitch, config)
    rows = event_features(players_xy, ball_xy, teams, fps, pitch, config) if prepared is None else prepared
    choices = decisions(rows, model)
    channel = np.full(len(ball_xy), -1, int)
    for event in choices:
        if event['class']:
            channel[event['frame']] = event['class'] - 1
    labels, reasons = decode(candidate, dead, present, channel, fps, config)
    return (labels, reasons, choices)
