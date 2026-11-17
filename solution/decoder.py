"""Offline segment decoder: the best sequence of whole H/A/D segments for one recording.

Each frame gets evidence for H, A and D. Dynamic programming then picks the best labelling under
a cost per segment change and a minimum segment length. It reads past and future frames, so it
needs the whole half and cannot run live. Inside a block it always commits to H, A or D.

Evidence is computed per team, never per track slot, so fragmented or swapped tracks do not matter.
Thresholds are engineering assumptions, never fitted to the competition match or its GT.
"""

from dataclasses import dataclass, fields
import math

import numpy as np


HOME, AWAY, DEAD = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])
CHUNK_FRAMES = 20000


@dataclass(frozen=True)
class DecoderConfig:
    control_radius_m: float = 2.0
    max_player_speed_mps: float = 12.0
    max_ball_speed_mps: float = 45.0
    activity_window_seconds: float = 2.0
    quiet_speed_mps: float = 1.0
    busy_speed_mps: float = 2.5
    still_ball_speed_mps: float = 0.35
    activity_weight: float = 0.6
    still_far_ball_weight: float = 1.0
    moving_near_ball_weight: float = 0.5
    missing_ball_weight: float = 0.1
    switch_cost_seconds: float = 1.0
    min_live_seconds: float = 2.0
    min_dead_seconds: float = 1.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Decoder settings must be finite and positive")
        if self.quiet_speed_mps >= self.busy_speed_mps:
            raise ValueError("quiet_speed_mps must be below busy_speed_mps")


def centred_mean(values, window):
    """Mean of the finite values in a centred window; NaN where the window holds none."""
    finite = np.isfinite(values)
    totals = np.r_[0.0, np.cumsum(np.where(finite, values, 0.0))]
    counts = np.r_[0, np.cumsum(finite)]
    index = np.arange(len(values))
    low, high = np.maximum(0, index - window), np.minimum(len(values), index + window + 1)
    n = counts[high] - counts[low]
    return np.divide(totals[high] - totals[low], n, out=np.full(len(values), np.nan), where=n > 0)


def evidence(players_xy, ball_xy, teams, fps, pitch, config):
    """Per-frame (team, dead): team > 0 favours home, team < 0 away; dead > 0 favours D."""
    players = np.asarray(players_xy, dtype=float)
    ball = np.asarray(ball_xy, dtype=float)
    teams = np.asarray(teams)
    scale = np.asarray(pitch, dtype=float)
    if players.ndim != 3 or players.shape[-1] != 2 or players.shape[1] == 0:
        raise ValueError("players_xy must have shape (frames, tracks, 2)")
    if ball.shape != (len(players), 2) or teams.shape != (players.shape[1],):
        raise ValueError("Ball or team array has incompatible shape")
    if not np.isin(teams, [0, 1]).all() or scale.shape != (2,) or not (scale > 0).all():
        raise ValueError("Teams must be 0/1; pitch dimensions must be positive")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")

    n = len(players)
    ball_speed = np.full(n, np.nan)
    ball_speed[1:] = np.linalg.norm(np.diff(ball, axis=0) * scale, axis=1) * fps
    # A physically impossible step is a false detection, so that ball position carries no evidence.
    ball = ball.copy()
    ball[ball_speed > config.max_ball_speed_mps] = np.nan
    visible = np.isfinite(ball).all(axis=1)

    nearest = np.full((n, 2), np.inf)
    activity = np.full(n, np.nan)
    previous = None
    for start in range(0, n, CHUNK_FRAMES):
        stop = min(start + CHUNK_FRAMES, n)
        chunk = players[start:stop]
        distance = np.linalg.norm((chunk - ball[start:stop, None, :]) * scale, axis=2)
        distance = np.where(np.isnan(distance), np.inf, distance)
        for team in (0, 1):
            if (teams == team).any():
                nearest[start:stop, team] = distance[:, teams == team].min(axis=1)
        joined = chunk if previous is None else np.concatenate((previous, chunk))
        speed = np.linalg.norm(np.diff(joined, axis=0) * scale, axis=2) * fps
        speed[speed > config.max_player_speed_mps] = np.nan  # an identity switch is not motion
        has_speed = np.isfinite(speed).any(axis=1)
        medians = np.full(len(speed), np.nan)
        if has_speed.any():
            medians[has_speed] = np.nanmedian(speed[has_speed], axis=1)
        if previous is None:
            activity[start + 1:stop] = medians
        else:
            activity[start:stop] = medians
        previous = chunk[-1:]

    closest = nearest.min(axis=1)
    usable = visible & np.isfinite(closest)
    margin = np.zeros(n)
    margin[usable] = np.clip((np.minimum(nearest[usable, 1], 1e6) - np.minimum(nearest[usable, 0], 1e6))
                             / config.control_radius_m, -1.0, 1.0)
    # A ball far from everyone says little about who owns it; in flight it keeps the last owner.
    weight = np.zeros(n)
    weight[usable] = np.minimum(1.0, config.control_radius_m / np.maximum(closest[usable], 1e-6))
    team_evidence = margin * weight

    smooth = centred_mean(activity, math.ceil(config.activity_window_seconds * fps / 2))
    middle = (config.quiet_speed_mps + config.busy_speed_mps) / 2
    half = (config.busy_speed_mps - config.quiet_speed_mps) / 2
    calm = np.nan_to_num(np.clip((middle - smooth) / half, -1.0, 1.0), nan=0.0)
    dead_evidence = config.activity_weight * calm + np.where(visible, 0.0, config.missing_ball_weight)
    still = usable & (ball_speed < config.still_ball_speed_mps)
    dead_evidence[still & (closest > 2 * config.control_radius_m)] += config.still_far_ball_weight
    dead_evidence[usable & ~still & (closest <= config.control_radius_m)] -= config.moving_near_ball_weight
    return team_evidence, dead_evidence


def decode_block(team_evidence, dead_evidence, fps, config):
    """State codes for one block: exact DP over segments, deterministic on ties."""
    n = len(team_evidence)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    live_length = max(1, math.ceil(config.min_live_seconds * fps))
    dead_length = max(1, math.ceil(config.min_dead_seconds * fps))
    lengths = (live_length, live_length, dead_length)
    cost = config.switch_cost_seconds * fps
    totals = np.zeros((n + 1, 3))
    totals[1:, HOME] = np.cumsum(team_evidence)
    totals[1:, AWAY] = np.cumsum(-team_evidence)
    totals[1:, DEAD] = np.cumsum(dead_evidence)
    totals = totals.tolist()
    # best[t][s]: best score of frames 0..t-1 ending in state s with its minimum length satisfied.
    best = [[0.0, 0.0, 0.0]] + [[-math.inf] * 3 for _ in range(n)]
    entered = np.full((n + 1, 3), -1, dtype=np.int8)  # previous state where a segment starts, -1 = stay
    for t in range(1, n + 1):
        row, totals_t, totals_p, best_p = best[t], totals[t], totals[t - 1], best[t - 1]
        for state in range(3):
            # The first segment may be cut by the block start, hence the direct total.
            score = (best_p[state] + totals_t[state] - totals_p[state]) if t > 1 else totals_t[state]
            start = t - lengths[state]
            if start >= 1:
                best_s, totals_s = best[start], totals[start]
                for other in range(3):
                    if other != state:
                        option = best_s[other] + totals_t[state] - totals_s[state] - cost
                        if option > score:
                            score = option
                            entered[t, state] = other
            row[state] = score
    states = np.zeros(n, dtype=np.int8)
    t = n
    state = int(np.argmax(best[n]))
    while t > 0:
        other = int(entered[t, state])
        if other < 0:
            states[t - 1] = state
            t -= 1
        else:
            start = t - lengths[state]
            states[start:t] = state
            t, state = start, other
    return centre_silent_changes(states, team_evidence, live_length)


def centre_silent_changes(states, team_evidence, min_length):
    """Put each H<->A change in the middle of the frames with no team evidence around it.

    With no evidence every position scores the same, so the DP would pick one by tie-break alone.
    The middle has the smallest worst-case error. Minimum lengths stay satisfied.
    """
    states = states.copy()
    changes = np.flatnonzero(states[1:] != states[:-1]) + 1
    bounds = np.r_[0, changes, len(states)]
    for index, change in enumerate(changes.tolist()):
        if DEAD in (states[change - 1], states[change]):
            continue
        low, high = change, change
        floor = bounds[index] + min_length if index else 0
        ceiling = bounds[index + 2] - min_length if index + 2 < len(bounds) - 1 else len(states)
        while low > floor and team_evidence[low - 1] == 0:
            low -= 1
        while high < ceiling and team_evidence[high] == 0:
            high += 1
        before, after = states[change - 1], states[change]
        middle = (low + high) // 2
        states[low:middle] = before
        states[middle:high] = after
        bounds[index + 1] = middle
    return states


def decode(players_xy, ball_xy, teams, fps=25.0, pitch=(105.3, 68.0), blocks=None, config=None):
    """H/A/D for every frame of each inclusive (first, last) block, X outside.

    Blocks are decoded separately, so nothing carries across halftime.
    """
    config = config or DecoderConfig()
    team_evidence, dead_evidence = evidence(players_xy, ball_xy, teams, fps, pitch, config)
    n = len(team_evidence)
    labels = np.full(n, "X", dtype="<U1")
    for first, last in (blocks or ((0, n - 1),)):
        if first < 0 or first > last or last >= n:
            raise ValueError("Block is outside the recording")
        states = decode_block(team_evidence[first:last + 1], dead_evidence[first:last + 1], fps, config)
        labels[first:last + 1] = LETTERS[states]
    return labels
