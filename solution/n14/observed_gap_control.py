"""Self-contained opt-in inference from observations and explicit frozen weights."""
from dataclasses import replace
import numpy as np
from . import capacity_control, evidence_control, dead_anchor, dead_entry_suffix, observed_dead, observed_return_events
from .dribble_evidence import RELATIVE_SPEED_MPS
from .possession import Config

def predict(players, ball, teams, model, fps=25.0, pitch=(105.3, 68.0), config=None, suffix=True, gap_D=False, gap_HA=False, prepared=None, robust_suffix=False, none_recovery=False, recovery_dead_guard=False, recovery_completed_guard=False, unowned_D=False, dead_control_arbitration=False, causal_dead=False, supported_dead_backdating=False, bounded_stationary_support=False, anchored_restart=False, intermittent_HA=False, nearby_gap_recovery=False, nearby_strength_arbitration=False, supported_owner_control=False, owner_evidence_arbitration=False):
    c = config or Config()
    if dead_control_arbitration and (not (gap_HA and unowned_D and recovery_completed_guard)):
        raise ValueError('D arbitration requires observed HA, unowned D and completed recovery guard')
    if causal_dead and (not dead_control_arbitration):
        raise ValueError('Causal D requires evidence arbitration')
    if supported_dead_backdating and (not causal_dead):
        raise ValueError('Supported D intervals require causal evidence arbitration')
    if bounded_stationary_support and (not supported_dead_backdating):
        raise ValueError('Bounded stationary support requires supported D intervals')
    if anchored_restart and (not causal_dead):
        raise ValueError('Anchored restart requires causal D arbitration')
    if intermittent_HA and (not anchored_restart):
        raise ValueError('Intermittent control requires anchored D arbitration')
    if nearby_gap_recovery and (not anchored_restart):
        raise ValueError('Nearby recovery requires anchored D arbitration')
    if nearby_strength_arbitration and (not nearby_gap_recovery):
        raise ValueError('Strength arbitration requires nearby recovery')
    if supported_owner_control and (not nearby_strength_arbitration):
        raise ValueError('Supported ownership requires strength arbitration')
    if owner_evidence_arbitration and (not supported_owner_control):
        raise ValueError('Ownership provenance requires supported ownership')
    ready = capacity_control.prepare(players, ball, teams, fps, pitch, c) if prepared is None else prepared
    raw, why, events = capacity_control.predict(players, ball, teams, model, fps, pitch, c, prepared=ready)
    extras = []
    recoveries = []
    evidence = None
    masks = None
    proof = {}
    if gap_HA:
        extras = observed_return_events.choices(players, ball, teams, model, events, fps, pitch, c)
        if none_recovery:
            from . import none_recovery as recovery
            recoveries = recovery.choices(players, ball, teams, extras, fps, pitch, c)
        guard = None
        release = None
        stationary = None
        starts = None
        accepted = None
        stationary_starts = None
        centers = None
        if (recovery_dead_guard or recovery_completed_guard) and recoveries or dead_control_arbitration:
            from .recovery_dead_guard import available_dead, certified_releases, support_starts
            evidence, masks, proof = state_masks(players, ball, teams, fps, pitch, c, suffix and (not causal_dead), robust_suffix, gap_D, unowned_D and (not dead_control_arbitration))
            guard = available_dead(masks)
            if recovery_completed_guard:
                release = certified_releases(masks, recoveries)
            if dead_control_arbitration:
                from . import unowned_stationary
                _, detail = unowned_stationary.combine(masks, evidence, players, ball, teams, fps, pitch)
                stationary = detail['confirmed']
                starts = support_starts(recoveries, len(ball))
                accepted = np.zeros(len(ball), bool)
                if supported_dead_backdating:
                    width = int(np.ceil(c.dead_confirm_seconds * fps))
                    stationary_starts = np.where(stationary, np.arange(len(ball)) - width + 1, -1)
                    if bounded_stationary_support:
                        from .stationary_support import bounded_support_starts
                        stationary_starts = bounded_support_starts(ball, stationary, fps, pitch, c)
                if anchored_restart:
                    from .rest_episode import certificate_centers
                    fresh = stationary | evidence['stopped'] | proof.get('confirmed', np.zeros(len(ball), bool))
                    centers = certificate_centers(ball, fresh, fps, pitch, c)
                if causal_dead:
                    stationary = stationary | guard
                    guard = np.zeros(len(ball), bool)
                    release = np.ones(len(ball), bool)
                proof['arbitration'] = dict(proposed=stationary, accepted=accepted, support_start=stationary_starts)
        nearby = None
        nearby_starts = None
        nearby_coherent = None
        if nearby_gap_recovery:
            from .nearby_recovery import channel as nearby_channel
            nearby, nearby_starts = nearby_channel(players, ball, teams, fps, pitch, c, coherent=supported_owner_control)
            if owner_evidence_arbitration:
                nearby_coherent = nearby >= 0
                nearby, nearby_starts = nearby_channel(players, ball, teams, fps, pitch, c, coherent=False)
        intermittent = None
        if intermittent_HA:
            from .intermittent_control import support
            intermittent = support(players, ball, teams, fps, pitch, c)
        if any((e['class'] for e in extras)) or recoveries or dead_control_arbitration:
            channel = np.full(len(ball), -1, int)
            for e in events + extras:
                if e['class']:
                    channel[e['frame']] = e['class'] - 1
            if causal_dead:
                restart = masks['restart'] & (masks['owner'] >= 0)
                channel[restart] = masks['owner'][restart]
            raw, why = evidence_control.decode(*ready['motion'], ready['teams'], ready['dead'], ready['present'], channel, fps, replace(c, still_ball_speed_mps=RELATIVE_SPEED_MPS), recovery_channel=recovery.channel(recoveries, len(ball)) if recoveries else None, recovery_dead_channel=guard, recovery_release_channel=release, stationary_dead_channel=stationary, recovery_support_start=starts, stationary_acceptance=accepted, causal_dead=causal_dead, stationary_support_start=stationary_starts, rest_centers=centers, ball_positions=np.asarray(ball) * pitch if anchored_restart else None, intermittent_support=intermittent, nearby_recovery_channel=nearby, nearby_support_start=nearby_starts, nearby_strength_arbitration=nearby_strength_arbitration, supported_owner_control=supported_owner_control, owner_evidence_arbitration=owner_evidence_arbitration, nearby_coherent=nearby_coherent)
    del ready
    if evidence is None:
        evidence, masks, proof = state_masks(players, ball, teams, fps, pitch, c, suffix, robust_suffix, gap_D, unowned_D)
    if causal_dead:
        masks = {k: v.copy() for k, v in masks.items()}
        masks['dead'][:] = False
    labels, reasons = dead_anchor.apply(raw, why, masks)
    return (labels, reasons, dict(events=sorted(events + extras, key=lambda e: e['frame']), extra_events=extras, recovery_events=recoveries, masks=masks, D_support=proof))

def state_masks(players, ball, teams, fps, pitch, c, suffix, robust_suffix, gap_D, unowned_D=False):
    evidence = dead_anchor.signals(players, ball, teams, fps, pitch, c)
    masks = dead_entry_suffix.state_masks(evidence, robust=robust_suffix) if suffix else dead_anchor.state_masks(evidence)
    proof = {}
    if gap_D:
        masks, proof = observed_dead.combine(masks, evidence, players, ball, teams, fps, pitch)
    if unowned_D:
        from . import unowned_stationary
        masks, detail = unowned_stationary.combine(masks, evidence, players, ball, teams, fps, pitch)
        proof['unowned_stationary'] = detail
    return (evidence, masks, proof)
