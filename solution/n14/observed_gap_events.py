"""Gap classifier extended with motion measured on observed points only."""
from dataclasses import replace
import numpy as np
from . import observed_gap_features as feature
from . import event_control as event
from . import evidence_control as evidence
from .gap_events import lookahead_frames
from .dribble_evidence import RELATIVE_SPEED_MPS
from .possession import Config, observations

def decisions(rows, model):
    if not rows:
        return []
    if model['role_features'] != list(feature.ROLE_FEATURES) or model['common_features'] != list(feature.COMMON_FEATURES):
        raise ValueError('Observed-gap feature contract differs')
    d = np.asarray([r['difference'] for r in rows]) / model['role_scale']
    c = np.asarray([r['common'] for r in rows]) / model['common_scale']
    side = d @ model['role_beta']
    shared = c @ model['common_beta'] + model['bias']
    logits = np.column_stack((np.zeros(len(rows)), shared + side, shared - side))
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    values /= values.sum(axis=1, keepdims=True)
    output = []
    for row, prob in zip(rows, values):
        winners = np.flatnonzero(np.isclose(prob, prob.max(), rtol=0, atol=1e-12))
        output.append(dict(frame=row['frame'], available_at=row['available_at'], **{'class': int(winners[0]) if len(winners) == 1 else 0}, probabilities=prob.tolist()))
    return output

def predict(players, ball, teams, base_model, gap_model, fps=25.0, pitch=(105.3, 68.0), config=None, prepared=None):
    config = config or Config()
    _, dead, present = observations(players, ball, teams, fps, pitch, config)
    original, additional = (event.event_features(players, ball, teams, fps, pitch, config), feature.features(players, ball, teams, fps, pitch, config)) if prepared is None else prepared
    choices = sorted([{**r, 'source': 'observed'} for r in event.decisions(original, base_model)] + [{**r, 'source': 'interpolated'} for r in decisions(additional, gap_model)], key=lambda r: r['frame'])
    channel = np.full(len(ball), -1, int)
    for row in choices:
        if row['class']:
            channel[row['frame']] = row['class'] - 1
    motion = evidence.motion_evidence(players, ball, fps, pitch, config)
    labels, reasons = evidence.decode(*motion, teams, dead, present, channel, fps, replace(config, still_ball_speed_mps=RELATIVE_SPEED_MPS))
    return (labels, reasons, choices)
