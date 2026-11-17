"""C4: scale gate, protected plateau, tiered events and regularized blending."""
import numpy as np
from . import c3_structural as c3
ALPHAS = (0.25, 0.35, 0.5)
CONFIG = dict(strong_cost=3.5, weak_cost=6.5, flight_cost=14.0, D_cost=8.0, strong_delta=0.12, weak_delta=0.15, contact_m=1.8, prominence=0.08, min_D_len=50, plateau_player_speed_mps=2.0, fresh_radius=37, full_pitch_players=14)

def scale(players):
    players = np.asarray(players)
    if players.ndim != 3 or players.shape[2] != 2 or (not len(players)):
        raise ValueError('Nonempty tracking block required')
    return dict(N_vis=float(np.median(np.isfinite(players).all(2).sum(1))), frames=len(players))

def context(trace, prob, x, players, pitch):
    data = c3.context(trace, prob, x)
    n = len(prob)
    p = np.asarray(pitch)
    ball = (x[:, :2] + 0.5) * p
    interior = (ball[:, 0] > 3) & (ball[:, 0] < p[0] - 3) & (ball[:, 1] > 3) & (ball[:, 1] < p[1] - 3) & (x[:, 2] > 0.5)
    fresh = data['physical'] & interior & data['contact'].any(1)
    no_fresh = c3.mean_window(fresh, 37, 37) == 0
    player_speed = c3.mean_window((x[:, 22] + x[:, 34]) * 5, 37, 37)
    data['plateau'] &= (data['raw'] == 2) | no_fresh & (player_speed < CONFIG['plateau_player_speed_mps'])
    cond = data['cond']
    delta = c3.mean_window(cond, 0, 24) - c3.mean_window(cond, 25, 0)
    strong = np.zeros((n, 2), bool)
    weak = strong.copy()
    for side in (0, 1):
        anchors = data['anchors'][1 - side, side]
        physical = data['physical'][anchors] & (data['raw'][anchors] == side) & data['contact'][anchors, side]
        s = anchors[physical & (delta[anchors, side] > 0.12)]
        w = anchors[(delta[anchors, side] > 0.15) | physical]
        for offset in range(-4, 5):
            for ids, target in ((s + offset, strong), (w + offset, weak)):
                ids = ids[(ids >= 0) & (ids < n)]
                target[ids, side] = True
    data.update(strong_events=strong, weak_events=weak, scale=scale(players), no_fresh=no_fresh, player_speed=player_speed)
    return data

def meta_features(data, trace):
    linear = c3.meta_features(data, trace)[:, :80]
    logp = np.log(np.maximum(data['prob'], 1e-06))
    flags = np.column_stack([data['physical'], data['active'], data['raw'] == 2, trace['restart_side'] >= 0, ~data['active'], data['plateau'], data['low_activity']]).astype(float)
    kinetic = data['x'][:, [2, 6]]
    interactions = [(logp[:, :, None] * flags[:, None, :]).reshape(len(logp), -1), (logp[:, :, None] * kinetic[:, None, :]).reshape(len(logp), -1), (flags[:, :, None] * kinetic[:, None, :]).reshape(len(logp), -1)]
    out = np.column_stack([linear, *interactions]).astype(np.float32)
    assert np.isfinite(out).all()
    return out
