"""Opt-in truncation of anchored D; preserve original entries and owner evidence."""
import math
import numpy as np
from . import dead_restart
from .dead_restart import signals, apply

def state_masks(evidence):
    original = dead_restart.state_masks(evidence)
    masks = {k: v.copy() for k, v in original.items()}
    n = len(original['dead'])
    masks['anchor_exit'] = np.zeros(n, bool)
    config = evidence['config']
    confirmation = max(1, math.ceil(config.control_confirm_seconds * evidence['fps']))
    center = None
    released = False
    departure = 0
    for i in range(n):
        if original['entered'][i]:
            center = evidence['mean_m'][i].copy()
            released = False
            departure = 0
        if center is not None and original['dead'][i] and (not released):
            departed = evidence['usable'][i] and evidence['inside'][i] and (not evidence['stopped'][i]) and (np.linalg.norm(evidence['ball_m'][i] - center) > config.control_radius_m)
            departure = departure + 1 if departed else 0
            if departure >= confirmation:
                released = True
                masks['anchor_exit'][i] = True
                masks['restart'][i] = True
        else:
            departure = 0
        if released:
            masks['dead'][i] = False
    return masks
