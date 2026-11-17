"""Possession changes require an event or confirmed relative-motion evidence.

Proximity alone is never a control signal. State transitions reset co-motion
history so evidence from the previous owner cannot undo a new event.
"""
from .trace_support import current_sink
import math
import numpy as np
from . import event_control as event
from .possession import Config, observations
from .motion import WINDOW

def motion_evidence(players_xy, ball_xy, fps, pitch, config):
    p = np.asarray(players_xy, float) * pitch
    b = np.asarray(ball_xy, float) * pitch
    relative = p - b[:, None]
    distance = np.linalg.norm(relative, axis=2)
    near = np.isfinite(distance) & (distance <= config.control_radius_m)
    velocity = np.full_like(relative, np.nan)
    velocity[1:] = np.diff(relative, axis=0) * fps
    speed = np.full_like(distance, np.nan)
    speed[1:] = np.linalg.norm(np.diff(p, axis=0), axis=2) * fps
    valid = near.copy()
    valid[1:] &= near[:-1]
    valid &= np.isfinite(speed) & (speed <= config.max_player_speed_mps)
    index = np.arange(len(b))[:, None]
    last_bad = np.maximum.accumulate(np.where(valid, -1, index), axis=0)
    cumulative = np.concatenate((np.zeros((1, p.shape[1], 2)), np.cumsum(np.where(valid[:, :, None], velocity, 0.0), axis=0)), axis=0)
    return (cumulative, last_bad, valid)

def decode(cumulative, last_bad, valid, teams, dead, present, event_channel, fps=25.0, config=None, trace=None, event_probabilities=None, recovery_channel=None, recovery_dead_channel=None, recovery_release_channel=None, stationary_dead_channel=None, recovery_support_start=None, stationary_acceptance=None, causal_dead=False, stationary_support_start=None, rest_centers=None, ball_positions=None, intermittent_support=None, nearby_recovery_channel=None, nearby_support_start=None, nearby_strength_arbitration=False, supported_owner_control=False, owner_evidence_arbitration=False, nearby_coherent=None):
    config = config or Config()
    teams = np.asarray(teams)
    hold = math.ceil(config.hold_seconds * fps)
    confirm = math.ceil(config.control_confirm_seconds * fps)
    dead_confirm = math.ceil(config.dead_confirm_seconds * fps)
    n = len(present)
    slots = np.arange(len(teams))
    if event_probabilities is not None:
        event_probabilities = np.asarray(event_probabilities, float)
        if event_probabilities.shape != (n, 3):
            raise ValueError('One event distribution per frame required')
        proposed = event_channel >= 0
        values = event_probabilities[proposed]
        if not np.isfinite(values).all() or (values < 0).any() or (not np.allclose(values.sum(axis=1), 1.0)):
            raise ValueError('Finite normalized event mass required')
    if recovery_channel is not None:
        recovery_channel = np.asarray(recovery_channel)
        if recovery_channel.shape != (n,) or not np.isin(recovery_channel, [-1, 0, 1]).all():
            raise ValueError('One optional recovery side per frame required')
    if recovery_dead_channel is not None:
        recovery_dead_channel = np.asarray(recovery_dead_channel)
        if recovery_dead_channel.shape != (n,) or recovery_dead_channel.dtype != np.bool_:
            raise ValueError('One available D guard boolean per frame required')
    if recovery_release_channel is not None:
        recovery_release_channel = np.asarray(recovery_release_channel)
        if recovery_release_channel.shape != (n,) or recovery_release_channel.dtype != np.bool_:
            raise ValueError('One certified recovery release boolean per frame required')
    if stationary_dead_channel is not None:
        stationary_dead_channel = np.asarray(stationary_dead_channel)
        if stationary_dead_channel.shape != (n,) or stationary_dead_channel.dtype != np.bool_:
            raise ValueError('One stationary D proposal boolean per frame required')
        if recovery_channel is not None and recovery_support_start is None:
            raise ValueError('Recovery support timestamps required for D arbitration')
    if recovery_support_start is not None:
        recovery_support_start = np.asarray(recovery_support_start)
        if recovery_support_start.shape != (n,) or not np.issubdtype(recovery_support_start.dtype, np.integer) or (recovery_support_start < -1).any() or (recovery_support_start > np.arange(n)).any():
            raise ValueError('Past or current recovery support timestamps required')
    if stationary_acceptance is not None:
        if stationary_acceptance.shape != (n,) or stationary_acceptance.dtype != np.bool_ or (not stationary_acceptance.flags.writeable):
            raise ValueError('Writable stationary D acceptance trace required')
        stationary_acceptance[:] = False
    if stationary_support_start is not None:
        stationary_support_start = np.asarray(stationary_support_start)
        if not causal_dead or stationary_support_start.shape != (n,) or (not np.issubdtype(stationary_support_start.dtype, np.integer)) or (stationary_support_start < -1).any() or (stationary_support_start > np.arange(n)).any():
            raise ValueError('A causal stationary certificate support interval is required')
    if intermittent_support is not None:
        intermittent_support = np.asarray(intermittent_support)
        if intermittent_support.shape != (n, len(teams)) or not np.issubdtype(intermittent_support.dtype, np.integer) or (intermittent_support < -1).any() or (intermittent_support > np.arange(n)[:, None]).any():
            raise ValueError('Past actual support for each control slot required')
    if nearby_recovery_channel is not None:
        nearby_recovery_channel = np.asarray(nearby_recovery_channel)
        nearby_support_start = np.asarray(nearby_support_start)
        if nearby_recovery_channel.shape != (n,) or not np.isin(nearby_recovery_channel, [-1, 0, 1]).all() or nearby_support_start.shape != (n,) or (not np.issubdtype(nearby_support_start.dtype, np.integer)) or (nearby_support_start < -1).any() or (nearby_support_start > np.arange(n)).any():
            raise ValueError('Current nearby recovery and past support required')
    if owner_evidence_arbitration:
        nearby_coherent = np.asarray(nearby_coherent)
        if nearby_coherent.shape != (n,) or nearby_coherent.dtype != np.bool_:
            raise ValueError('One actual proximity coherence flag per frame required')
    supported_side = -1
    owner_supported = True
    episode = None
    if rest_centers is not None:
        from .rest_episode import RestEpisode
        rest_centers = np.asarray(rest_centers, float)
        ball_positions = np.asarray(ball_positions, float)
        if not causal_dead or rest_centers.shape != (n, 2) or ball_positions.shape != (n, 2):
            raise ValueError('Causal rest geometry required')
        if np.isinf(rest_centers).any() or np.isinf(ball_positions).any():
            raise ValueError('Finite or missing rest geometry required')
        episode = RestEpisode(config.control_radius_m, dead_confirm, hold)
    weak_live = False
    recovery_pending_D = False
    last_available_D = -1
    last_live = -n - 1
    index = np.arange(n)[:, None]
    starts = np.maximum(last_bad + 1, index - WINDOW + 1)
    counts = np.maximum(0, index - starts + 1)
    means = (cumulative[index + 1, slots] - cumulative[np.minimum(starts, index + 1), slots]) / np.maximum(counts[:, :, None], 1)
    eligible = valid & (counts >= confirm) & (np.linalg.norm(means, axis=2) < config.still_ball_speed_mps)
    cached_votes = np.any(eligible[:, teams == 0], axis=1).astype(np.uint8) + 2 * np.any(eligible[:, teams == 1], axis=1).astype(np.uint8)
    del means, eligible, starts, counts, index
    labels = np.full(n, 'X', dtype='<U1')
    reasons = np.full(n, 'uncertain', dtype='<U24')
    state = 'X'
    age = hold + 1
    quiet = 0
    floor = -1
    _capture_sink = current_sink()
    if owner_evidence_arbitration and _capture_sink is not None:
        _capture_sink.begin(n)
    for i in range(n):
        try:
            if episode is not None:
                episode.observe(i, ball_positions[i], rest_centers[i], bool(stationary_dead_channel is not None and stationary_dead_channel[i]))
            if recovery_dead_channel is not None and recovery_dead_channel[i]:
                recovery_pending_D = True
                last_available_D = i
            if trace is not None:
                trace[i, :4] = (ord(state), age, floor, quiet)
            if present[i] == 0:
                age += 1
                quiet = 0
                floor = i
                if age > hold:
                    state = 'X'
                reasons[i] = 'no_player_observations'
                continue
            quiet = quiet + 1 if dead[i] else 0
            if i - floor < confirm:
                votes = 0
            elif i - floor >= WINDOW:
                votes = int(cached_votes[i])
            else:
                start = np.maximum(np.maximum(last_bad[i] + 1, i - WINDOW + 1), floor + 1)
                count = np.maximum(0, i - start + 1)
                average = (cumulative[i + 1] - cumulative[np.minimum(start, i + 1), slots]) / np.maximum(count[:, None], 1)
                supported = valid[i] & (count >= confirm) & (np.linalg.norm(average, axis=1) < config.still_ball_speed_mps)
                votes = int(np.any(supported & (teams == 0))) + 2 * int(np.any(supported & (teams == 1)))
            maintenance = False
            if intermittent_support is not None:
                certified = intermittent_support[i] > floor
                if supported_owner_control:
                    incumbent = 0 if state == 'H' else 1 if state == 'A' else -1
                    maintenance = incumbent >= 0 and bool(np.any(certified & (teams == incumbent)))
                else:
                    votes |= int(np.any(certified & (teams == 0))) + 2 * int(np.any(certified & (teams == 1)))
            owner = 0 if state == 'H' else 1 if state == 'A' else -1
            if owner_evidence_arbitration and owner >= 0 and nearby_coherent[i] and (nearby_recovery_channel is not None) and (nearby_recovery_channel[i] == owner) and (nearby_support_start[i] > floor):
                supported_side = owner
                owner_supported = True
            chosen_event = event_channel[i]
            if event_probabilities is not None and owner >= 0 and (chosen_event >= 0) and (chosen_event != owner):
                mass = event_probabilities[i]
                if mass[0] + mass[owner + 1] >= mass[chosen_event + 1] - 1e-12:
                    chosen_event = -1
            if trace is not None:
                trace[i, 4:6] = (votes, event_channel[i])
            rest_locked = episode is not None and episode.locked(recovery_pending_D)
            if episode is not None and recovery_pending_D:
                if rest_locked:
                    recovery_side = int(recovery_channel[i]) if recovery_channel is not None else -1
                    if not nearby_strength_arbitration and recovery_side < 0 and (nearby_recovery_channel is not None) and (nearby_support_start[i] > floor):
                        recovery_side = int(nearby_recovery_channel[i])
                    episode.control(i, votes, recovery_side)
                elif votes == 3:
                    episode.control(i, votes)
            if votes and (not rest_locked) or chosen_event >= 0:
                last_live = i
            if quiet >= dead_confirm and (not causal_dead or (not votes and chosen_event < 0 and (owner < 0 or age >= hold or weak_live) and (i - last_live >= dead_confirm))):
                if state != 'D':
                    floor = i
                weak_live = False
                if owner_evidence_arbitration:
                    supported_side = -1
                    owner_supported = True
                state, age = ('D', 0)
                labels[i], reasons[i] = (state, 'sustained_dead_evidence')
                if stationary_support_start is not None and stationary_support_start[i] >= 0:
                    first = max(int(stationary_support_start[i]), last_live + 1)
                    labels[first:i + 1] = 'D'
                    reasons[first:i + 1] = 'sustained_dead_evidence'
                if causal_dead:
                    if episode is not None:
                        episode.accept(i, recovery_pending_D, np.isfinite(rest_centers[i]).all())
                    recovery_pending_D = True
                    last_available_D = i
            elif chosen_event >= 0:
                if owner_evidence_arbitration:
                    owner_supported = chosen_event == supported_side
                weak_live = False
                if recovery_dead_channel is None or not recovery_dead_channel[i]:
                    recovery_pending_D = False
                if episode is not None:
                    episode.release()
                state, age = ('H' if chosen_event == 0 else 'A', 0)
                floor = i
                labels[i], reasons[i] = (state, 'inferred_control_event')
            elif votes in (1, 2) and (not rest_locked):
                if owner_evidence_arbitration:
                    supported_side = 0 if votes == 1 else 1
                    owner_supported = True
                weak_live = False
                if recovery_dead_channel is None or not recovery_dead_channel[i]:
                    recovery_pending_D = False
                if episode is not None:
                    episode.release()
                chosen = 0 if votes == 1 else 1
                next_state = 'H' if chosen == 0 else 'A'
                if next_state != state:
                    first = max(i - confirm + 1, floor + 1)
                    labels[first:i] = next_state
                    reasons[first:i] = 'confirmed_co_motion'
                    floor = i
                state, age = (next_state, 0)
                labels[i], reasons[i] = (state, 'confirmed_co_motion')
            elif owner >= 0 and votes & 1 << owner and (not rest_locked):
                if owner_evidence_arbitration:
                    supported_side = owner
                    owner_supported = True
                weak_live = False
                age = 0
                labels[i], reasons[i] = (state, 'contested_co_motion')
            elif recovery_channel is not None and recovery_channel[i] >= 0 and (votes != 3) and (not rest_locked) and (state == 'X' or (owner >= 0 and age >= hold) or (stationary_dead_channel is not None and state == 'D')) and (not recovery_pending_D or (recovery_release_channel is not None and recovery_release_channel[i] and (recovery_dead_channel is None or not recovery_dead_channel[i]) and (recovery_support_start is None or recovery_support_start[i] > last_available_D))):
                if owner_evidence_arbitration:
                    supported_side = int(recovery_channel[i])
                    owner_supported = True
                weak_live = False
                recovery_pending_D = False
                if episode is not None:
                    episode.release()
                state, age = ('H' if recovery_channel[i] == 0 else 'A', 0)
                floor = i
                last_live = i
                labels[i], reasons[i] = (state, 'observed_recovery')
            elif nearby_recovery_channel is not None and nearby_recovery_channel[i] >= 0 and (nearby_support_start[i] > floor) and (votes == 0) and (not rest_locked) and (state == 'X' or (owner >= 0 and (age >= hold or (nearby_strength_arbitration and nearby_recovery_channel[i] != owner))) or state == 'D') and (not recovery_pending_D or (not nearby_strength_arbitration and episode is not None and episode.departed)) and (rest_centers is None or not np.isfinite(rest_centers[i]).all()) and (not owner_evidence_arbitration or nearby_coherent[i] or (not owner_supported) or (nearby_recovery_channel[i] == supported_side)):
                state, age = ('H' if nearby_recovery_channel[i] == 0 else 'A', 0)
                floor = i
                if owner_evidence_arbitration:
                    if nearby_coherent[i]:
                        supported_side = int(nearby_recovery_channel[i])
                    owner_supported = int(nearby_recovery_channel[i]) == supported_side
                weak_live = bool(nearby_strength_arbitration)
                if not nearby_strength_arbitration:
                    last_live = i
                first = int(nearby_support_start[i]) if supported_owner_control else i
                labels[first:i + 1] = state
                reasons[first:i + 1] = 'nearby_gap_recovery'
                recovery_pending_D = False
                if episode is not None:
                    episode.release()
            elif episode is not None and recovery_pending_D and (episode.restart(i) >= 0) and (votes == 0) and (recovery_channel is None or recovery_channel[i] < 0 or recovery_channel[i] == episode.restart(i)):
                weak_live = False
                side = episode.restart(i)
                if owner_evidence_arbitration:
                    supported_side = side
                    owner_supported = True
                state, age = ('H' if side == 0 else 'A', 0)
                floor = i
                last_live = i
                labels[i], reasons[i] = (state, 'anchored_restart')
                recovery_pending_D = False
                episode.release()
            elif stationary_dead_channel is not None and stationary_dead_channel[i] and (i - last_live >= dead_confirm) and (owner < 0 or age >= hold or weak_live) and (episode is None or recovery_pending_D or episode.latest_frame >= 0):
                first = i if causal_dead else max(0, i - dead_confirm + 1)
                if stationary_support_start is not None and stationary_support_start[i] >= 0:
                    first = max(int(stationary_support_start[i]), last_live + 1)
                labels[first:i + 1] = 'D'
                reasons[first:i + 1] = 'unowned_stationary_D'
                if state != 'D':
                    floor = i
                if episode is not None:
                    episode.accept(i, recovery_pending_D, np.isfinite(rest_centers[i]).all())
                weak_live = False
                if owner_evidence_arbitration:
                    supported_side = -1
                    owner_supported = True
                state, age = ('D', 0)
                recovery_pending_D = True
                last_available_D = i if episode is None or episode.latest_frame < 0 else max(last_available_D, episode.latest_frame)
                if stationary_acceptance is not None:
                    stationary_acceptance[i] = True
            elif supported_owner_control and maintenance and (owner >= 0) and (not rest_locked):
                if owner_evidence_arbitration:
                    supported_side = owner
                    owner_supported = True
                age = 0
                weak_live = True
                labels[i], reasons[i] = (state, 'intermittent_maintenance')
            else:
                age += 1
                if state != 'X' and age <= hold:
                    if causal_dead and state == 'D':
                        reasons[i] = 'unconfirmed_dead_memory'
                    else:
                        labels[i], reasons[i] = (state, 'short_memory')
                else:
                    state = 'X'
            if trace is not None:
                trace[i, 6:] = (ord(state), age, floor)
        finally:
            if owner_evidence_arbitration and _capture_sink is not None:
                _capture_sink.capture(i, state, age, owner_supported, weak_live, supported_side, reasons[i], bool(nearby_coherent[i]) if nearby_coherent is not None else False, hold)
    return (labels, reasons)

def predict(players_xy, ball_xy, teams, model, fps=25.0, pitch=(105.3, 68.0), config=None, prepared=None):
    config = config or Config()
    _, dead, present = observations(players_xy, ball_xy, teams, fps, pitch, config)
    evidence = motion_evidence(players_xy, ball_xy, fps, pitch, config)
    rows = event.event_features(players_xy, ball_xy, teams, fps, pitch, config) if prepared is None else prepared
    decisions = event.decisions(rows, model)
    channel = np.full(len(ball_xy), -1, int)
    for decision in decisions:
        if decision['class']:
            channel[decision['frame']] = decision['class'] - 1
    labels, reasons = decode(*evidence, teams, dead, present, channel, fps, config)
    return (labels, reasons, decisions)
