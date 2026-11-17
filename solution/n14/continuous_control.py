"""Experimental continuous co-motion evidence, separate from contact intention.

An existing nearby trajectory moving with the ball can supersede a closer
passing opponent. Otherwise use the original proximity cue. No IO or training.
"""
import math
import numpy as np
from .motion import causal_mean11
from .possession import Config, observations

def control_observations(players_xy, ball_xy, teams, fps, pitch, config):
    candidate, dead, n_present = observations(players_xy, ball_xy, teams, fps, pitch, config)
    players = np.asarray(players_xy, dtype=float) * pitch
    ball = np.asarray(ball_xy, dtype=float) * pitch
    teams = np.asarray(teams)
    relative = players - ball[:, None]
    distances = np.linalg.norm(relative, axis=2)
    near = np.isfinite(distances) & (distances <= config.control_radius_m)
    velocity = np.full_like(relative, np.nan)
    velocity[1:] = np.diff(relative, axis=0) * fps
    player_speed = np.full_like(distances, np.nan)
    player_speed[1:] = np.linalg.norm(np.diff(players, axis=0), axis=2) * fps
    usable = near.copy()
    usable[1:] &= near[:-1]
    usable &= np.isfinite(player_speed) & (player_speed <= config.max_player_speed_mps)
    velocity[~usable] = np.nan
    speed = np.linalg.norm(causal_mean11(velocity), axis=2)
    index = np.arange(len(ball))[:, None]
    reset = np.maximum.accumulate(np.where(usable, -1, index), axis=0)
    enough = index - reset >= math.ceil(config.control_confirm_seconds * fps)
    eligible = enough & (speed < config.still_ball_speed_mps)
    scores = np.where(eligible, distances, np.inf)
    nearest = np.argmin(scores, axis=1)
    use = np.isfinite(scores[np.arange(len(ball)), nearest])
    candidate[use] = teams[nearest[use]]
    return (candidate, dead, n_present)

def decode(candidate, dead_evidence, n_present, fps, config):
    """Original finite-state policy with the already tested H/A backdating.

    Kept separate from frozen motion11 so historical inference stays intact.
    Co-motion changes only the input candidate, never quiet/dead observations.
    """
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
            if control_count == control_confirm:
                start = i - control_confirm + 1
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

def predict(players_xy, ball_xy, teams, fps=25.0, pitch=(105.3, 68.0), config=None):
    """Continuous evidence plus confirmed boundaries, maximum N-1 lookahead."""
    config = config or Config()
    signals = control_observations(players_xy, ball_xy, teams, fps, pitch, config)
    return decode(*signals, fps, config)
