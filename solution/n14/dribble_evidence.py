"""Separate relative dribble speed from the stationary-ball dead-ball test.

Research variant: 1.8 m/s is a synthetic-development setting. Dead-ball
observations and event features still use the original Config unchanged.
"""
from dataclasses import replace
import numpy as np
from . import event_control as event
from . import evidence_control as evidence
from .possession import Config, observations
RELATIVE_SPEED_MPS = 1.8

def predict(players_xy, ball_xy, teams, model, fps=25.0, pitch=(105.3, 68.0), config=None, prepared=None):
    config = config or Config()
    _, dead, present = observations(players_xy, ball_xy, teams, fps, pitch, config)
    motion = evidence.motion_evidence(players_xy, ball_xy, fps, pitch, config)
    rows = event.event_features(players_xy, ball_xy, teams, fps, pitch, config) if prepared is None else prepared
    decisions = event.decisions(rows, model)
    channel = np.full(len(ball_xy), -1, int)
    for decision in decisions:
        if decision['class']:
            channel[decision['frame']] = decision['class'] - 1
    decode_config = replace(config, still_ball_speed_mps=RELATIVE_SPEED_MPS)
    labels, reasons = evidence.decode(*motion, teams, dead, present, channel, fps, decode_config)
    return (labels, reasons, decisions)
