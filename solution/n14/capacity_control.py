"""Same control decoder with a chosen symmetric observation feature basis."""
from dataclasses import replace
import numpy as np
from . import unified_control as unified
from . import control_basis as basis
from . import evidence_control as evidence
from .dribble_evidence import RELATIVE_SPEED_MPS
from .gap_events import lookahead_frames

def purpose_cost(probabilities, rows, offsets):
    """A declared training-only class-cost correction; identical for H and A."""
    if set(offsets) != {'observed', 'interpolated', 'reappearance'} or not all((np.isfinite(v) for v in offsets.values())):
        raise ValueError('Finite declared proposal-purpose offsets required')
    if not len(rows):
        return probabilities
    with np.errstate(divide='ignore'):
        scores = np.log(probabilities)
    scores[:, 1:] += np.asarray([offsets[r['source']] for r in rows])[:, None]
    values = np.exp(scores - scores.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)

def prepare(players, ball, teams, fps=25.0, pitch=(105.3, 68.0), config=None):
    ready = unified.prepare(players, ball, teams, fps, pitch, config)
    return {**ready, 'designs': {kind: basis.design(ready['rows'], kind) for kind in ('linear', 'quadratic')}}

def predict(players, ball, teams, model, fps=25.0, pitch=(105.3, 68.0), config=None, prepared=None, trace=None, event_state=False):
    ready = prepare(players, ball, teams, fps, pitch, config) if prepared is None else prepared
    kind = model['basis']
    role_names, common_names = basis.names(kind)
    if model['role_features'] != role_names or model['common_features'] != common_names:
        raise ValueError('Control basis model contract differs')
    probabilities = basis.probabilities(*ready['designs'][kind], model)
    if 'purpose_cost' in model:
        probabilities = purpose_cost(probabilities, ready['rows'], model['purpose_cost'])
    channel = np.full(len(ball), -1, int)
    choices = []
    for row, prob in zip(ready['rows'], probabilities):
        winners = np.flatnonzero(np.isclose(prob, prob.max(), rtol=0, atol=1e-12))
        cls = int(winners[0]) if len(winners) == 1 else 0
        choices.append(dict(frame=row['frame'], available_at=row['available_at'], source=row['source'], **{'class': cls}, probabilities=prob.tolist()))
        if cls:
            channel[row['frame']] = cls - 1
    mass = None
    if event_state:
        mass = np.full((len(ball), 3), np.nan)
        for row, prob in zip(ready['rows'], probabilities):
            mass[row['frame']] = prob
    labels, reasons = evidence.decode(*ready['motion'], ready['teams'], ready['dead'], ready['present'], channel, ready['fps'], replace(ready['config'], still_ball_speed_mps=RELATIVE_SPEED_MPS), trace=trace, event_probabilities=mass)
    return (labels, reasons, choices)
