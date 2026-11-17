"""Observe the frozen N14 decoder through its explicit provenance hook."""
import numpy as np

from .n14 import observed_gap_control
from .n14.trace_support import SINK


class Recorder:
    def __init__(self):
        self.calls = 0
        self.values = {}

    def begin(self, n):
        self.calls += 1
        self.values = dict(
            state=np.zeros(n, np.uint8), age=np.zeros(n, np.int32),
            owner_supported=np.zeros(n, bool), weak=np.zeros(n, bool),
            supported_side=np.full(n, -1, np.int8), decision_reason=np.full(n, "", dtype="<U48"),
            coherent=np.zeros(n, bool), hold=np.array(0),
        )

    def capture(self, i, state, age, owner_supported, weak, supported_side, reason, coherent, hold):
        v = self.values
        v["state"][i] = ord(state)
        v["age"][i] = age
        v["owner_supported"][i] = owner_supported
        v["weak"][i] = weak
        v["supported_side"][i] = supported_side
        v["decision_reason"][i] = reason
        v["coherent"][i] = coherent
        v["hold"] = np.array(hold)


def trace(players, ball, teams, model, flags, pitch):
    recorder = Recorder()
    token = SINK.set(recorder)
    try:
        labels, reasons, info = observed_gap_control.predict(
            players, ball, teams, model, pitch=tuple(pitch), **flags,
        )
    finally:
        SINK.reset(token)
    if recorder.calls != 1:
        raise ValueError("Expected exactly one N14 ownership decoder trace")
    captures = recorder.values
    live = np.isin(labels, ["H", "A"])
    same = captures["state"] == np.array([ord(x) for x in labels], np.uint8)
    within = captures["age"] <= int(captures["hold"])
    physical = live & same & within & captures["owner_supported"]
    physical |= live & np.isin(reasons, [
        "confirmed_co_motion", "contested_co_motion", "observed_recovery", "anchored_restart", "fresh_restart_team",
    ])
    near = (reasons == "nearby_gap_recovery") & live
    edges = np.flatnonzero(np.r_[True, near[1:] != near[:-1], True])
    for start, stop in zip(edges[:-1], edges[1:]):
        certificate = (captures["coherent"][start:stop] & captures["owner_supported"][start:stop]
                       & (captures["decision_reason"][start:stop] == "nearby_gap_recovery"))
        if near[start] and np.any(certificate):
            physical[start:stop] = True
    active = live & (physical | (same & within & ~captures["weak"]))
    inside = np.isfinite(ball).all(1) & ((ball >= 0) & (ball <= 1)).all(1)
    side = np.where(labels == "H", 0, np.where(labels == "A", 1, -1))
    contact_side = np.full(len(ball), -1, np.int8)
    onset = physical & np.r_[True, (~physical[:-1]) | (side[1:] != side[:-1])]
    onset |= physical & np.isin(reasons, ["observed_recovery", "anchored_restart"])
    contact_side[onset] = side[onset]
    restart_side = np.full(len(ball), -1, np.int8)
    restart = info["masks"]["restart"] & live
    restart |= (np.isin(reasons, ["anchored_restart", "fresh_restart_team"]) & live
                & np.r_[True, labels[1:] != labels[:-1]])
    restart_side[restart] = side[restart]
    d_entry = (labels == "D") & np.r_[True, labels[:-1] != "D"]
    arbitration = info["D_support"].get("arbitration", {})
    starts = arbitration.get("support_start")
    if starts is not None:
        ids = np.asarray(starts)[np.asarray(arbitration["accepted"], bool)]
        d_entry[ids[ids >= 0]] = True
    return dict(labels=labels, reasons=reasons, active=active, physical=physical, inside=inside,
                contact_side=contact_side, restart_side=restart_side, d_entry=d_entry, **captures)
