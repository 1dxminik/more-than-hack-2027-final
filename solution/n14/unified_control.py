"""One observed/gap feature contract; supervision mode is only in training."""
from dataclasses import replace
import numpy as np
from . import event_control as event
from . import gap_events as gap
from . import observed_gap_features as feature
from . import observed_gap_events as classifier
from . import evidence_control as evidence
from .dribble_evidence import RELATIVE_SPEED_MPS
from .possession import Config, observations

def event_rows(players, ball, teams, fps=25.0, pitch=(105.3, 68.0), config=None):
    observed = [{**row, 'source': 'observed', 'common': [*row['common'], *[0.0] * 6]} for row in event.event_features(players, ball, teams, fps, pitch, config)]
    inferred = [{**row, 'source': 'interpolated'} for row in gap.features(players, ball, teams, fps, pitch, config)]
    rows = sorted([*observed, *inferred], key=lambda row: row['frame'])
    if len({r['frame'] for r in rows}) != len(rows):
        raise ValueError('Duplicate observed/gap proposal')
    return feature.features(players, ball, teams, fps, pitch, config, prepared=rows)

def prepare(players, ball, teams, fps=25.0, pitch=(105.3, 68.0), config=None):
    config = config or Config()
    _, dead, present = observations(players, ball, teams, fps, pitch, config)
    return dict(rows=event_rows(players, ball, teams, fps, pitch, config), motion=evidence.motion_evidence(players, ball, fps, pitch, config), dead=dead, present=present, teams=np.asarray(teams), fps=fps, config=config)

def predict(players, ball, teams, model, fps=25.0, pitch=(105.3, 68.0), config=None, prepared=None):
    ready = prepare(players, ball, teams, fps, pitch, config) if prepared is None else prepared
    choices = classifier.decisions(ready['rows'], model)
    channel = np.full(len(ball), -1, int)
    for row, choice in zip(ready['rows'], choices):
        choice['source'] = row['source']
        if choice['class']:
            channel[choice['frame']] = choice['class'] - 1
    labels, reasons = evidence.decode(*ready['motion'], ready['teams'], ready['dead'], ready['present'], channel, ready['fps'], replace(ready['config'], still_ball_speed_mps=RELATIVE_SPEED_MPS))
    return (labels, reasons, choices)
