"""Linear control-state updates, including observed ball reappearance.

No IO, labels or fitted parameters. Long gaps are never interpolated here.
The original observed/bounded-gap proposals and decoder remain available.
"""
import math
import numpy as np
from . import unified_control as unified, observed_gap_features as feature
from . import event_control as event, capacity_control as capacity, control_basis as basis
from .possession import Config

def recovery_rows(players, ball, teams, fps=25.0, pitch=(105.3, 68.0), config=None, anchors=None):
    config = config or Config()
    p = np.asarray(players, float) * pitch
    b = np.asarray(ball, float) * pitch
    teams = np.asarray(teams)
    finite = np.isfinite(b).all(axis=1)
    width = math.ceil(0.12 * fps)
    rows = []
    last_visible = -1
    for e in range(len(b)):
        if not finite[e]:
            continue
        gap = e - last_visible - 1
        previous = last_visible
        last_visible = e
        if anchors is not None:
            if e not in anchors:
                continue
            gap, previous = anchors[e]
        if not gap or previous < 0 or e < width + 1 or (e + width + 1 >= len(b)):
            continue
        local = np.linalg.norm(p[e - width:e + width + 1] - b[e - width:e + width + 1, None], axis=2)
        if not np.any(np.isfinite(local[width]) & (local[width] <= config.control_radius_m)):
            continue
        before = b[e - width:e + 1]
        after = b[e:e + width + 1]
        vin, _ = feature.velocity(before, fps)
        vout, _ = feature.velocity(after, fps)
        a = np.linalg.norm(vin) if vin is not None else 40.0
        z = np.linalg.norm(vout) if vout is not None else 40.0
        impulse = np.linalg.norm(vout - vin) if vin is not None and vout is not None else 40.0
        per_team = []
        for side in (0, 1):
            slots = np.flatnonzero(teams == side)
            ds = local[:, slots]
            safe = np.where(np.isfinite(ds), np.minimum(ds, 12.0), 12.0)
            mins = safe.min(axis=1) if len(slots) else np.full(2 * width + 1, 12.0)
            ri = []
            ro = []
            change = []
            for j in slots:
                path = p[e - width:e + width + 1, j]
                if not np.isfinite(local[width, j]) or local[width, j] > config.control_radius_m:
                    continue
                if not np.isfinite(path).all() or np.any(np.linalg.norm(np.diff(path, axis=0), axis=1) * fps > config.max_player_speed_mps):
                    continue
                rel = path - b[e - width:e + width + 1]
                x, _ = feature.velocity(rel[:width + 1], fps)
                y, _ = feature.velocity(rel[width:], fps)
                if x is not None:
                    ri.append(np.linalg.norm(x))
                if y is not None:
                    ro.append(np.linalg.norm(y))
                if x is not None and y is not None:
                    change.append(np.linalg.norm(y - x))
            per_team.append(np.asarray([mins[width], mins[0], mins[-1], mins[width - 1], mins[width + 1], mins.min(), mins[:width + 1].mean(), mins[width:].mean(), np.mean(mins <= config.control_radius_m), np.sum(ds[width] <= config.control_radius_m), np.log1p(min(ri, default=40.0)), np.log1p(min(ro, default=40.0)), np.log1p(min(change, default=40.0)), float(bool(ri))]))
        d = per_team[0] - per_team[1]
        missing = ~finite[e - width - 1:e + width + 2]
        cosine = float(vin @ vout / max(a * z, 1e-08)) if vin is not None and vout is not None else 0.0
        nearest = local[width][np.isfinite(local[width])]
        c = [*np.abs(d), np.log1p(a), np.log1p(z), np.log1p(impulse), cosine, feature.straight(before), feature.straight(after), min(float(nearest.min()), 12.0), float(np.sum(local[width] <= config.control_radius_m)), float(missing.mean()), float(np.mean(~finite[e - width:e + 1])), float(np.mean(~finite[e:e + width + 1])), 0.0, float(gap), float(np.log1p(np.linalg.norm(b[e] - b[previous]) * fps / (e - previous)))]
        rows.append(dict(frame=e, available_at=e + width + 1, difference=d.tolist(), common=list(map(float, c)), source='reappearance'))
    return feature.features(players, ball, teams, fps, pitch, config, prepared=rows)

def event_rows(players, ball, teams, fps=25.0, pitch=(105.3, 68.0), config=None):
    rows = unified.event_rows(players, ball, teams, fps, pitch, config)
    existing = {r['frame'] for r in rows}
    rows += [r for r in recovery_rows(players, ball, teams, fps, pitch, config) if r['frame'] not in existing]
    return sorted(rows, key=lambda r: r['frame'])

def prepare(players, ball, teams, fps=25.0, pitch=(105.3, 68.0), config=None):
    ready = unified.prepare(players, ball, teams, fps, pitch, config)
    existing = {r['frame'] for r in ready['rows']}
    ready['rows'] += [r for r in recovery_rows(players, ball, teams, fps, pitch, config) if r['frame'] not in existing]
    ready['rows'].sort(key=lambda r: r['frame'])
    ready['designs'] = {'linear': basis.design(ready['rows'], 'linear')}
    return ready

def predict(players, ball, teams, model, fps=25.0, pitch=(105.3, 68.0), config=None, prepared=None):
    if model.get('supervision') != 'possession_update' or model.get('basis') != 'linear':
        raise ValueError('Explicit linear possession-update model required')
    ready = prepare(players, ball, teams, fps, pitch, config) if prepared is None else prepared
    return capacity.predict(players, ball, teams, model, fps, pitch, config, prepared=ready)
