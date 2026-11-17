"""Observable D episode geometry; departure permits proof, never assigns a side."""
import numpy as np
from .observed_sample_contract import limits, fit
from .possession import Config

def certificate_centers(ball, confirmed, fps=25.0, pitch=(105.3, 68.0), config=None):
    c = config or Config()
    policy = limits(fps, c)
    width = policy['D_window_frames']
    b = np.asarray(ball, float) * pitch
    confirmed = np.asarray(confirmed)
    if b.shape != (len(confirmed), 2) or confirmed.dtype != np.bool_:
        raise ValueError('One actual certificate per ball frame required')
    result = np.full_like(b, np.nan)
    for i in np.flatnonzero(confirmed):
        if i < width - 1:
            raise ValueError('Complete confirmation window required')
        value, q = fit(b[i - width + 1:i + 1], fps, policy['D_min_points'], policy['max_sample_gap_frames'])
        if value is None or q['freshness_frames']:
            raise ValueError('Fresh available D samples required')
        result[i] = value['center']
    return result

class RestEpisode:

    def __init__(self, radius, window, hold=50):
        self.radius = radius
        self.window = window
        self.anchor = None
        self.latest = None
        self.latest_frame = -1
        self.departed = False
        self.departure_frame = -1
        self.hold = hold
        self.control_side = -1
        self.control_frame = -hold - 1

    def observe(self, i, ball, center, proposed):
        if np.isfinite(center).all():
            self.latest = center.copy()
            self.latest_frame = i
        elif not proposed:
            self.latest = None
            self.latest_frame = -1
        if self.anchor is not None and (not self.departed) and np.isfinite(ball).all() and (np.linalg.norm(ball - self.anchor) > self.radius):
            self.departed = True
            self.departure_frame = i

    def locked(self, pending):
        return pending and self.anchor is not None and (not self.departed)

    def control(self, i, votes, recovery=-1):
        if votes == 3:
            self.control_side = -1
            return
        side = 0 if votes == 1 else 1 if votes == 2 else int(recovery)
        if side >= 0:
            self.control_side = side
            self.control_frame = i

    def restart(self, i):
        return self.control_side if self.departed and i - self.control_frame <= self.hold else -1

    def accept(self, i, pending, fresh):
        if not pending or self.anchor is None or (self.departed and fresh and (i - self.departure_frame >= self.window)):
            self.anchor = None if self.latest is None else self.latest.copy()
            self.departed = False
            self.departure_frame = -1
            self.control_side = -1
            self.control_frame = -self.hold - 1

    def release(self):
        self.anchor = None
        self.latest = None
        self.latest_frame = -1
        self.departed = False
        self.departure_frame = -1
        self.control_side = -1
        self.control_frame = -self.hold - 1
