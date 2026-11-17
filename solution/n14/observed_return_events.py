"""Bounded return impulses, not continuous sparse co-motion or forced ownership."""
import math
import numpy as np
from .observed_sample_contract import limits, fit, continuous_slot
from . import reappearance_control, control_basis
from .possession import Config

def rows(players, ball, teams, fps=25.0, pitch=(105.3, 68.0), config=None):
    c = config or Config()
    policy = limits(fps, c)
    width = policy['HA_half_window_frames']
    p = np.asarray(players, float) * pitch
    b = np.asarray(ball, float) * pitch
    t = np.asarray(teams)
    finite = np.isfinite(b).all(axis=1)
    anchors = {}
    quality = {}
    last = -1
    for returned in np.flatnonzero(finite):
        gap = int(returned - last - 1)
        previous = last
        last = int(returned)
        e = int(returned) + 1
        if previous < 0 or gap <= 0 or gap + 1 > policy['max_sample_gap_frames']:
            continue
        if e < width + 1 or e + width + 1 >= len(b) or (not finite[e:e + 2].all()):
            continue
        incoming, pre = fit(b[e - width:e + 1], fps, minimum=2, max_gap=policy['max_sample_gap_frames'])
        outgoing, post = fit(b[e:e + width + 1], fps, minimum=2, max_gap=policy['max_sample_gap_frames'])
        if incoming is None or outgoing is None:
            continue
        a, z = (np.linalg.norm(incoming['velocity']), np.linalg.norm(outgoing['velocity']))
        impulse = np.linalg.norm(outgoing['velocity'] - incoming['velocity'])
        if max(a, z) < 2.0 or impulse < 1.5:
            continue
        distances = np.linalg.norm(p[e] - b[e], axis=1)
        near = np.flatnonzero(np.isfinite(distances) & (distances <= c.control_radius_m))
        if not len(near) or len(np.unique(t[near])) != 1:
            continue
        witnesses = [int(j) for j in near if continuous_slot(p[previous:e + width + 1, j], fps, c)]
        if not witnesses:
            continue
        anchors[e] = (gap, previous)
        quality[e] = dict(return_frame=int(returned), last_before_gap=int(previous), gap_frames=gap, pre={k: int(pre[k]) for k in ('count', 'span_frames', 'max_gap_frames', 'freshness_frames')}, post={k: int(post[k]) for k in ('count', 'span_frames', 'max_gap_frames', 'freshness_frames')}, continuous_slots=witnesses, observed_impulse_mps=float(impulse))
    result = reappearance_control.recovery_rows(players, ball, teams, fps, pitch, c, anchors=anchors)
    return [dict(r, source='observed_return_impulse', support=quality[r['frame']]) for r in result]

def choices(players, ball, teams, model, existing=(), fps=25.0, pitch=(105.3, 68.0), config=None):
    if model.get('supervision') != 'acquisition' or model.get('basis') != 'linear':
        raise ValueError('Frozen acquisition Linear required')
    proposed = rows(players, ball, teams, fps, pitch, config)
    occupied = {e['frame'] for e in existing}
    proposed = [r for r in proposed if r['frame'] not in occupied]
    probability = control_basis.probabilities(*control_basis.design(proposed, 'linear'), model)
    result = []
    for row, prob in zip(proposed, probability):
        winners = np.flatnonzero(np.isclose(prob, prob.max(), rtol=0, atol=1e-12))
        cls = int(winners[0]) if len(winners) == 1 else 0
        result.append(dict(frame=row['frame'], available_at=row['available_at'], source=row['source'], support=row['support'], **{'class': cls}, probabilities=prob.tolist()))
    return result
