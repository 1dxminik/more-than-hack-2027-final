"""Per-frame features, 25->5 fps binning, team/pitch flips and Viterbi decoding for the TCN model.

numpy only, no IO and no labels: the same code runs on IDSSE-derived files and on competition ATD.
Inputs follow solution/tracking.load_tracking: players (T, slots, 2) and ball (T, 2) normalised 0-1
from the top-left corner, NaN where missing; teams (slots,) with 0 = home, 1 = away.
Slots are anonymous tracklets, so per-team features are built from the K nearest slots sorted by
distance to the ball. That makes them independent of slot order and of identity switches.
"""

import warnings

import numpy as np

FPS = 25
STEP = 5                 # 25 fps -> 5 fps bins
K = 3                    # nearest slots per team
MAX_GAP_S = 2.0          # ball gaps up to this long are linearly interpolated
MAX_PLAYER_SPEED = 12.0  # m/s; faster jumps are identity switches, not motion
CAP_FRAMES = 30 * FPS    # cap for "time since/until ball seen"
BALL_MARGIN_M = 10.0     # ball positions are clipped to the pitch plus this margin
H, A, D, IGNORE = 0, 1, 2, -1


def _names():
    names = ["ball_seen", "ball_interp", "ball_avail", "since_seen", "until_seen",
             "ball_x", "ball_y", "ball_speed", "ball_accel", "all_slow"]
    for t in (0, 1):
        for k in range(K):
            names += [f"t{t}_k{k}_{n}" for n in ("dist", "dx", "dy", "speed", "rel", "valid")]
        names += [f"t{t}_{n}" for n in ("near2", "near5", "present", "cx", "cy", "sx", "speed", "slow")]
    return names


NAMES = _names()


def _flip_maps(names):
    index = {n: i for i, n in enumerate(names)}
    swap = np.array([index[n.replace("t0_", "#").replace("t1_", "t0_").replace("#", "t1_")]
                     if n[:3] in ("t0_", "t1_") else i for i, n in enumerate(names)])
    neg_x = np.array([n.endswith(("_dx", "_cx")) or n == "ball_x" for n in names])
    neg_y = np.array([n.endswith(("_dy", "_cy")) or n == "ball_y" for n in names])
    return swap, neg_x, neg_y


SWAP, NEG_X, NEG_Y = _flip_maps(NAMES)


def flip(features, labels, swap_teams=False, mirror_x=False, mirror_y=False):
    """Symmetry augmentation on (..., F) features and labels. Swapping teams also swaps H and A."""
    x = features[..., SWAP] if swap_teams else features.copy()
    if mirror_x:
        x[..., NEG_X] *= -1
    if mirror_y:
        x[..., NEG_Y] *= -1
    y = labels
    if swap_teams:
        y = labels.copy()
        y[labels == H], y[labels == A] = A, H
    return x, y


def _central_velocity(xy, fps, max_speed=None):
    """Velocity from frames t-2 and t+2, NaN where either end is missing or the jump is too fast."""
    v = np.full_like(xy, np.nan)
    v[2:-2] = (xy[4:] - xy[:-4]) * (fps / 4.0)
    if max_speed is not None:
        speed = np.linalg.norm(v, axis=-1)
        v[speed > max_speed] = np.nan
    return v


def interpolate_ball(ball, max_gap=int(MAX_GAP_S * FPS)):
    """(filled, seen, interpolated, since, until); gaps longer than max_gap stay NaN."""
    n = len(ball)
    seen = np.isfinite(ball).all(axis=1)
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, index, -1))
    nxt = np.minimum.accumulate(np.where(seen, index, n)[::-1])[::-1]
    since = np.where(last >= 0, index - last, CAP_FRAMES).clip(0, CAP_FRAMES)
    until = np.where(nxt < n, nxt - index, CAP_FRAMES).clip(0, CAP_FRAMES)
    gap = nxt - last - 1
    fill = ~seen & (last >= 0) & (nxt < n) & (gap <= max_gap)
    filled = ball.astype(float).copy()
    if fill.any():
        alpha = ((index - last) / np.maximum(nxt - last, 1))[fill, None]
        filled[fill] = (1 - alpha) * ball[last[fill]] + alpha * ball[nxt[fill]]
    return filled, seen, fill, since, until


def frame_features(players, ball, teams, pitch=(105.0, 68.0), fps=FPS):
    """(T, len(NAMES)) float32, finite everywhere. Distances in metres, scaled to roughly unit size."""
    # Frames without any player (before kickoff, halftime) give all-NaN means; they become 0 below.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return _frame_features(players, ball, teams, pitch, fps)


def _frame_features(players, ball, teams, pitch, fps):
    players = np.asarray(players, float)
    ball = np.asarray(ball, float)
    teams = np.asarray(teams)
    pitch = np.asarray(pitch, float)
    n = len(ball)
    filled, seen, interp, since, until = interpolate_ball(ball)
    avail = seen | interp
    p = (players - 0.5) * pitch                     # centred metres
    # An aerial ball projected from one camera can land far off the pitch; keep it near the field.
    b = np.clip((filled - 0.5) * pitch, -pitch / 2 - BALL_MARGIN_M, pitch / 2 + BALL_MARGIN_M)
    pv = _central_velocity(p, fps, MAX_PLAYER_SPEED)
    bv = _central_velocity(b, fps)
    pspeed = np.linalg.norm(pv, axis=-1)            # NaN where unknown
    bspeed = np.linalg.norm(bv, axis=-1)
    bacc = np.full(n, np.nan)
    bacc[2:-2] = np.linalg.norm(bv[4:] - bv[:-4], axis=-1)
    rel = p - b[:, None]
    dist = np.linalg.norm(rel, axis=-1)             # NaN if player or ball unavailable
    rel_speed = np.linalg.norm(pv - bv[:, None], axis=-1)
    present = np.isfinite(players).all(axis=-1)
    slow_all = np.nanmean(np.where(np.isfinite(pspeed), pspeed < 0.5, np.nan), axis=1)

    out = {
        "ball_seen": seen, "ball_interp": interp, "ball_avail": avail,
        "since_seen": np.log1p(since / fps), "until_seen": np.log1p(until / fps),
        "ball_x": np.where(avail, b[:, 0] / 50.0, 0), "ball_y": np.where(avail, b[:, 1] / 34.0, 0),
        "ball_speed": np.clip(bspeed, 0, 40) / 10.0, "ball_accel": np.clip(bacc, 0, 60) / 20.0,
        "all_slow": slow_all,
    }
    for t in (0, 1):
        slots = np.flatnonzero(teams == t)
        d = dist[:, slots]
        order = np.argsort(np.where(np.isfinite(d), d, np.inf), axis=1)[:, :K]
        for k in range(K):
            j = slots[order[:, k]] if len(slots) > k else np.zeros(n, int)
            dk = dist[np.arange(n), j] if len(slots) > k else np.full(n, np.nan)
            valid = np.isfinite(dk)
            r = rel[np.arange(n), j]
            out[f"t{t}_k{k}_dist"] = np.where(valid, np.clip(dk, 0, 30) / 10.0, 3.0)
            out[f"t{t}_k{k}_dx"] = np.where(valid, np.clip(r[:, 0], -30, 30) / 10.0, 0)
            out[f"t{t}_k{k}_dy"] = np.where(valid, np.clip(r[:, 1], -30, 30) / 10.0, 0)
            out[f"t{t}_k{k}_speed"] = np.where(valid, np.clip(pspeed[np.arange(n), j], 0, 12) / 5.0, 0)
            out[f"t{t}_k{k}_rel"] = np.where(valid, np.clip(rel_speed[np.arange(n), j], 0, 40) / 10.0, 0)
            out[f"t{t}_k{k}_valid"] = valid
        out[f"t{t}_near2"] = (np.nan_to_num(d, nan=99) <= 2.0).sum(axis=1) / 2.0
        out[f"t{t}_near5"] = (np.nan_to_num(d, nan=99) <= 5.0).sum(axis=1) / 3.0
        tp = p[:, slots]
        out[f"t{t}_present"] = present[:, slots].sum(axis=1) / 11.0
        out[f"t{t}_cx"] = np.nanmean(tp[..., 0], axis=1) / 50.0
        out[f"t{t}_cy"] = np.nanmean(tp[..., 1], axis=1) / 34.0
        out[f"t{t}_sx"] = np.nanstd(tp[..., 0], axis=1) / 20.0
        ts = pspeed[:, slots]
        out[f"t{t}_speed"] = np.clip(np.nanmean(ts, axis=1), 0, 12) / 5.0
        out[f"t{t}_slow"] = np.nanmean(np.where(np.isfinite(ts), ts < 0.5, np.nan), axis=1)
    x = np.stack([np.asarray(out[name], float) for name in NAMES], axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def bin_features(x, step=STEP):
    """Mean over consecutive step-frame bins; the last bin is padded with its final row."""
    n = len(x)
    nb = -(-n // step)
    pad = nb * step - n
    if pad:
        x = np.concatenate([x, np.repeat(x[-1:], pad, axis=0)])
    return x.reshape(nb, step, -1).mean(axis=1).astype(np.float32)


def bin_labels(codes, step=STEP):
    """Majority H/A/D per bin from 25 fps codes (0=H, 1=A, 2=D, 3=X); IGNORE where all X."""
    n = len(codes)
    nb = -(-n // step)
    padded = np.full(nb * step, 3, dtype=np.int64)
    padded[:n] = codes
    counts = np.stack([(padded.reshape(nb, step) == c).sum(axis=1) for c in (H, A, D)], axis=1)
    y = counts.argmax(axis=1).astype(np.int8)
    y[counts.sum(axis=1) == 0] = IGNORE
    return y


def upsample(bins, n, step=STEP):
    return np.repeat(bins, step)[:n]


def viterbi(log_probs, penalty):
    """Most likely state path when every change of state costs `penalty` (log units).

    log_probs: (T, S). With penalty 0 this is the per-step argmax; a larger penalty removes
    short excursions whose evidence does not pay for two changes.
    """
    return viterbi_many(log_probs, [penalty])[0]


def viterbi_many(log_probs, penalties):
    """viterbi() for several penalties in one pass; returns (len(penalties), T) int8 paths."""
    lp = np.asarray(log_probs, float)
    n, s = lp.shape
    pen = np.asarray(penalties, float)[:, None]
    rows = np.arange(len(pen))
    states = np.arange(s)
    back = np.zeros((n, len(pen), s), dtype=np.int8)
    score = np.repeat(lp[:1], len(pen), axis=0)
    for t in range(1, n):
        best = score.argmax(axis=1)
        switch = score[rows, best][:, None] - pen
        stay = score >= switch
        back[t] = np.where(stay, states, best[:, None])
        score = np.where(stay, score, switch) + lp[t]
    paths = np.empty((len(pen), n), dtype=np.int8)
    paths[:, -1] = score.argmax(axis=1)
    for t in range(n - 1, 0, -1):
        paths[:, t - 1] = back[t, rows, paths[:, t]]
    return paths


def nearest_player_codes(players, ball, teams, pitch=(105.0, 68.0)):
    """Vectorised step 1 of official/starter/baseline.py: 0/1 team of the nearest slot, -1 if none."""
    players = np.asarray(players, float)
    ball = np.asarray(ball, float)
    scale = np.asarray(pitch, float)
    d = (((players - ball[:, None]) * scale) ** 2).sum(axis=-1)
    d = np.where(np.isfinite(d), d, np.inf)
    j = d.argmin(axis=1)
    found = np.isfinite(d[np.arange(len(d)), j])
    return np.where(found, np.asarray(teams)[j], -1).astype(np.int8)
