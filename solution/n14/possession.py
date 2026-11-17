"""Default motion11 inference; the unchanged v0 reference is possession_v0.py.

Pure inference: receives observations only, never labels, scenario names or paths.
All thresholds are initial engineering assumptions, not tuned on competition GT.
This cannot infer a referee's decision from identical observable trajectories.
"""
import math
import numpy as np
from .motion import causal_mean11
from .possession_v0 import Config
METHOD = 'motion11'

def observations(players_xy, ball_xy, teams, fps, pitch, config):
    """Original control geometry with causal means of 11 velocity samples."""
    players = np.asarray(players_xy, dtype=float)
    ball = np.asarray(ball_xy, dtype=float)
    teams = np.asarray(teams)
    pitch = np.asarray(pitch, dtype=float)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive')
    if players.ndim != 3 or players.shape[-1] != 2 or players.shape[1] == 0:
        raise ValueError('players_xy must have shape (frames, tracks, 2)')
    if ball.shape != (len(players), 2) or teams.shape != (players.shape[1],):
        raise ValueError('Ball or team array has incompatible shape')
    if not np.isin(teams, [0, 1]).all() or pitch.shape != (2,) or (not np.isfinite(pitch).all()) or (pitch <= 0).any():
        raise ValueError('Teams must be 0/1; pitch dimensions must be positive')
    present = np.isfinite(players).all(axis=2)
    ball_present = np.isfinite(ball).all(axis=1)
    distances = np.linalg.norm((players - ball[:, None, :]) * pitch, axis=2)
    distances[~present | ~ball_present[:, None]] = np.inf
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(players)), nearest]
    candidate = np.where(nearest_distance <= config.control_radius_m, teams[nearest], -1)
    p, b = (players * pitch, ball * pitch)
    player_velocity = np.full_like(p, np.nan)
    ball_velocity = np.full_like(b, np.nan)
    player_velocity[1:] = np.diff(p, axis=0) * fps
    ball_velocity[1:] = np.diff(b, axis=0) * fps
    player_velocity[np.linalg.norm(player_velocity, axis=2) > config.max_player_speed_mps] = np.nan
    speeds = np.linalg.norm(causal_mean11(player_velocity), axis=2)
    valid_speed = np.isfinite(speeds)
    n_speed = valid_speed.sum(axis=1)
    sufficient = (n_speed >= config.min_motion_tracks) & (n_speed >= 0.75 * present.sum(axis=1))
    quiet = sufficient & ((valid_speed & (speeds < config.quiet_player_speed_mps)).sum(axis=1) >= 0.8 * n_speed)
    ball_speed = np.linalg.norm(causal_mean11(ball_velocity), axis=1)
    distances = np.linalg.norm(p - b[:, None], axis=2)
    distances[~np.isfinite(distances)] = np.inf
    stationary_far = ball_present & (ball_speed < config.still_ball_speed_mps) & (distances.min(axis=1) > 2 * config.control_radius_m)
    dead_evidence = quiet & (~ball_present | stationary_far)
    return (candidate, dead_evidence, present.sum(axis=1))

def predict(players_xy, ball_xy, teams, fps=25.0, pitch=(105.3, 68.0), config=None):
    """Return (H/A/D/X labels, reasons), with causal finite-state inference.

    H/A is held for at most hold_seconds after the last supported control.
    D needs sustained stillness and either absent or distant stationary ball.
    Missing all players returns X, and a long tracking outage resets memory.
    Restart/switch needs a sustained nearby ball; single touches don't suffice.
    X is uncertainty, not an unscored frame; the official temporal scorer fills it.
    """
    config = config or Config()
    candidate, dead_evidence, n_present = observations(players_xy, ball_xy, teams, fps, pitch, config)
    hold = math.ceil(config.hold_seconds * fps)
    dead_confirm = math.ceil(config.dead_confirm_seconds * fps)
    control_confirm = math.ceil(config.control_confirm_seconds * fps)
    labels = np.full(len(candidate), 'X', dtype='<U1')
    reasons = np.full(len(candidate), 'uncertain', dtype='<U24')
    state = 'X'
    age = hold + 1
    quiet_count = control_count = 0
    pending = -1
    for i in range(len(candidate)):
        if n_present[i] == 0:
            age += 1
            quiet_count = control_count = 0
            pending = -1
            if age > hold:
                state = 'X'
            reasons[i] = 'no_player_observations'
            continue
        quiet_count = quiet_count + 1 if dead_evidence[i] else 0
        if candidate[i] >= 0:
            control_count = control_count + 1 if candidate[i] == pending else 1
            pending = int(candidate[i])
        else:
            pending, control_count = (-1, 0)
        if quiet_count >= dead_confirm:
            state, age = ('D', 0)
            labels[i], reasons[i] = ('D', 'sustained_dead_evidence')
        elif control_count >= control_confirm:
            state, age = ('H' if pending == 0 else 'A', 0)
            labels[i], reasons[i] = (state, 'confirmed_nearby_ball')
        else:
            age += 1
            if state != 'X' and age <= hold:
                labels[i], reasons[i] = (state, 'short_memory')
            else:
                state = 'X'
    return (labels, reasons)
