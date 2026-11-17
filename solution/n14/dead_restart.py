"""One opt-in stationary-isolated-ball and fresh-restart hypothesis; no GT/fit."""
import math
import numpy as np
from . import possession
from .possession import Config

def rolling_sum(values, width):
    values = np.asarray(values, dtype=float)
    result = np.zeros_like(values)
    if len(values) >= width:
        cumulative = np.concatenate((np.zeros((1, *values.shape[1:])), np.cumsum(values, axis=0)))
        result[width - 1:] = cumulative[width:] - cumulative[:-width]
    return result

def signals(players_xy, ball_xy, teams, fps=25.0, pitch=(105.3, 68.0), config=None):
    config = config or Config()
    _, _, present = possession.observations(players_xy, ball_xy, teams, fps, pitch, config)
    pitch = np.asarray(pitch, float)
    players = np.asarray(players_xy, float) * pitch
    ball = np.asarray(ball_xy, float) * pitch
    teams = np.asarray(teams)
    n = len(ball)
    width = max(2, math.ceil(config.dead_confirm_seconds * fps))
    finite = np.isfinite(ball).all(axis=1)
    padded_field = finite & ((ball >= -2 * config.control_radius_m) & (ball <= pitch + 2 * config.control_radius_m)).all(axis=1)
    inside = finite & ((ball >= 0) & (ball <= pitch)).all(axis=1)
    step = np.full(n, np.inf)
    step[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1)
    usable = padded_field & (step <= config.control_radius_m)
    distances = np.linalg.norm(players - ball[:, None], axis=2)
    distances[~np.isfinite(distances)] = np.inf
    closest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(n), closest]
    near = np.where(usable & (nearest_distance <= config.control_radius_m), teams[closest], -1)
    isolated = usable & (nearest_distance > 2 * config.control_radius_m) & (present >= config.min_motion_tracks)
    clean = np.where(np.isfinite(ball), ball, 0.0)
    total = rolling_sum(clean, width)
    mean = total / width
    index = np.arange(n, dtype=float)
    centered_sum = rolling_sum(clean * index[:, None], width) - (index - (width - 1) / 2)[:, None] * total
    time_square_sum = width * (width * width - 1) / 12
    slope = centered_sum / time_square_sum
    residual = (rolling_sum(clean * clean, width) - total * total / width - slope * slope * time_square_sum).sum(axis=1) / width
    rms = np.sqrt(np.maximum(0.0, residual))
    fit_speed = np.linalg.norm(slope, axis=1) * fps
    complete = rolling_sum(isolated, width) == width
    stopped = complete & (fit_speed < config.still_ball_speed_mps) & (rms <= config.still_ball_speed_mps * config.dead_confirm_seconds)
    return dict(stopped=stopped, usable=usable, inside=inside, near=near, ball_m=ball, mean_m=mean, present=present, fit_speed=fit_speed, residual_rms=rms, fps=fps, config=config)

def state_masks(evidence):
    config = evidence['config']
    fps = evidence['fps']
    n = len(evidence['near'])
    confirmation = max(1, math.ceil(config.control_confirm_seconds * fps))
    hold = math.ceil(config.hold_seconds * fps)
    dead = np.zeros(n, bool)
    restart = np.zeros(n, bool)
    owner = np.full(n, -1, int)
    entered = np.zeros(n, bool)
    expired = np.zeros(n, bool)
    active = False
    center = None
    last_rest = -hold - 1
    pending = -1
    streak = 0
    last_team = -1
    last_confirmed = -confirmation - 1
    restart_team = -1
    owner_until = -1
    for i in range(n):
        side = int(evidence['near'][i])
        streak = streak + 1 if side >= 0 and side == pending else 1 if side >= 0 else 0
        pending = side
        if streak >= confirmation:
            last_team, last_confirmed = (side, i)
        if not active and evidence['stopped'][i]:
            active = True
            center = evidence['mean_m'][i].copy()
            last_rest = i
            entered[i] = True
            owner_until = -1
        if active:
            distance = np.linalg.norm(evidence['ball_m'][i] - center) if evidence['usable'][i] else np.inf
            fresh_team = last_team >= 0 and i - last_confirmed < confirmation
            if evidence['usable'][i] and evidence['inside'][i] and (distance > config.control_radius_m) and fresh_team:
                active = False
                restart[i] = True
                restart_team, owner_until = (last_team, i + hold)
            elif evidence['usable'][i] and distance <= config.control_radius_m:
                last_rest = i
            elif i - last_rest > hold:
                active = False
                expired[i] = True
        if active and evidence['present'][i] >= config.min_motion_tracks:
            dead[i] = True
        if not active and i <= owner_until and (evidence['present'][i] > 0):
            owner[i] = restart_team
    return dict(dead=dead, restart=restart, owner=owner, entered=entered, expired=expired)

def apply(labels, reasons, masks):
    labels = np.asarray(labels)
    reasons = np.asarray(reasons)
    n = len(masks['dead'])
    if labels.shape != (n,) or reasons.shape != (n,) or (not np.isin(labels, list('HADX')).all()):
        raise ValueError('One valid label/reason per observed frame is required')
    result = labels.copy()
    why = np.asarray(reasons, dtype='<U48').copy()
    active = masks['dead']
    result[active] = 'D'
    why[active] = 'confirmed_stationary_isolation'
    restart = (masks['owner'] >= 0) & (np.isin(result, ('D', 'X')) | masks['restart'])
    result[restart] = np.where(masks['owner'][restart] == 0, 'H', 'A')
    why[restart] = 'fresh_restart_team'
    return (result, why)

def predict(players_xy, ball_xy, teams, fps=25.0, pitch=(105.3, 68.0), config=None):
    labels, reasons = possession.predict(players_xy, ball_xy, teams, fps, pitch, config)
    return apply(labels, reasons, state_masks(signals(players_xy, ball_xy, teams, fps, pitch, config)))
