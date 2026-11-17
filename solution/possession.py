"""Step 1: short possession memory and conservative dead-ball evidence.

Pure inference: receives observations only, never labels, scenario names or paths.
All thresholds are initial engineering assumptions, not tuned on competition GT.
This cannot infer a referee's decision from identical observable trajectories.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Config:
    hold_seconds: float = 2.0
    dead_confirm_seconds: float = 1.5
    control_confirm_seconds: float = 0.2
    control_radius_m: float = 2.0
    quiet_player_speed_mps: float = 0.5
    still_ball_speed_mps: float = 0.35
    max_player_speed_mps: float = 12.0
    min_motion_tracks: int = 4

    def __post_init__(self):
        for value in vars(self).values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Configuration values must be finite and positive")
        if not isinstance(self.min_motion_tracks, int):
            raise ValueError("min_motion_tracks must be an integer")


def observations(players_xy, ball_xy, teams, fps, pitch, config):
    """Distances and motion in metres; a missing value never becomes zero speed."""
    players = np.asarray(players_xy, dtype=float)
    ball = np.asarray(ball_xy, dtype=float)
    teams = np.asarray(teams)
    pitch = np.asarray(pitch, dtype=float)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if players.ndim != 3 or players.shape[-1] != 2 or players.shape[1] == 0:
        raise ValueError("players_xy must have shape (frames, tracks, 2)")
    if ball.shape != (len(players), 2) or teams.shape != (players.shape[1],):
        raise ValueError("Ball or team array has incompatible shape")
    if not np.isin(teams, [0, 1]).all() or pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any():
        raise ValueError("Teams must be 0/1; pitch dimensions must be positive")
    present = np.isfinite(players).all(axis=2)
    ball_present = np.isfinite(ball).all(axis=1)
    distances = np.linalg.norm((players - ball[:, None, :]) * pitch, axis=2)
    distances[~present | ~ball_present[:, None]] = np.inf
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(players)), nearest]
    candidate = np.where(nearest_distance <= config.control_radius_m, teams[nearest], -1)

    speeds = np.full(present.shape, np.nan)
    speeds[1:] = np.linalg.norm(np.diff(players, axis=0) * pitch, axis=2) * fps
    # Exclude discontinuities rather than interpreting an identity switch as motion.
    speeds[speeds > config.max_player_speed_mps] = np.nan
    valid_speed = np.isfinite(speeds)
    n_speed = valid_speed.sum(axis=1)
    sufficient = (n_speed >= config.min_motion_tracks) & (n_speed >= 0.75 * present.sum(axis=1))
    quiet = sufficient & ((valid_speed & (speeds < config.quiet_player_speed_mps)).sum(axis=1) >= 0.8 * n_speed)
    ball_speed = np.full(len(ball), np.nan)
    ball_speed[1:] = np.linalg.norm(np.diff(ball, axis=0) * pitch, axis=1) * fps
    # A stationary ball near a player can be live. Do not call that a stoppage.
    stationary_far = ball_present & (ball_speed < config.still_ball_speed_mps) & (nearest_distance > 2 * config.control_radius_m)
    dead_evidence = quiet & (~ball_present | stationary_far)
    return candidate, dead_evidence, present.sum(axis=1)


def fill_abstentions(labels, blocks=None):
    """Write the label the official scorer would assume for each X, block by block.

    SpellF1 stays identical, because the scorer does this same fill before matching possessions.
    MacroF1 can move either way: a right label earns credit, a wrong one adds a false positive that
    X would not. Filling lowered it on IDSSE, where this method's X falls mostly on stoppages.
    A block with no label at all stays X; there is nothing to repeat.
    """
    filled = np.asarray(labels, dtype="<U1").copy()
    for first, last in (blocks or ((0, len(filled) - 1),)):
        window = filled[first:last + 1]
        known = np.flatnonzero(window != "X")
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(window != "X", np.arange(len(window)), -1))
        carry[carry < 0] = known[0]
        filled[first:last + 1] = window[carry]
    return filled


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
    labels = np.full(len(candidate), "X", dtype="<U1")
    reasons = np.full(len(candidate), "uncertain", dtype="<U24")
    state = "X"
    age = hold + 1
    quiet_count = control_count = 0
    pending = -1
    for i in range(len(candidate)):
        if n_present[i] == 0:
            age += 1
            quiet_count = control_count = 0
            pending = -1
            if age > hold:
                state = "X"
            reasons[i] = "no_player_observations"
            continue
        quiet_count = quiet_count + 1 if dead_evidence[i] else 0
        if candidate[i] >= 0:
            control_count = control_count + 1 if candidate[i] == pending else 1
            pending = int(candidate[i])
        else:
            pending, control_count = -1, 0

        if quiet_count >= dead_confirm:
            state, age = "D", 0
            labels[i], reasons[i] = "D", "sustained_dead_evidence"
        elif control_count >= control_confirm:
            state, age = ("H" if pending == 0 else "A"), 0
            labels[i], reasons[i] = state, "confirmed_nearby_ball"
        else:
            age += 1
            if state != "X" and age <= hold:
                labels[i], reasons[i] = state, "short_memory"
            else:
                state = "X"
    return labels, reasons
