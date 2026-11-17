"""Additional gap features whose velocity fits use observed positions only.

Feature prototype, separate from the frozen interpolation-based classifier.
No model, labels, IO or event-time ground truth enters this module.
"""
import math
import numpy as np
from . import gap_events as gap
from . import event_control as event
from .possession import Config
EXTRA_ROLE = ('observed_relative_in', 'observed_relative_out', 'observed_relative_impulse', 'incoming_pairs', 'outgoing_pairs', 'observed_path', 'post_distance')
ROLE_FEATURES = (*event.ROLE_FEATURES, *EXTRA_ROLE)
EXTRA_COMMON = ('observed_ball_in', 'observed_ball_out', 'observed_ball_impulse', 'observed_turn_cosine', 'observed_pre_straight', 'observed_post_straight', 'observed_pre_points', 'observed_post_points')
COMMON_FEATURES = (*gap.COMMON_FEATURES, *('abs_' + name for name in EXTRA_ROLE), *EXTRA_COMMON)

def velocity(path, fps):
    """Least-squares velocity at actual sample times, without filling gaps."""
    path = np.asarray(path, float)
    indices = np.flatnonzero(np.isfinite(path).all(axis=1))
    if len(indices) < 2:
        return (None, int(len(indices)))
    axis = indices - indices.mean()
    return (axis @ path[indices] / (axis @ axis) * fps, int(len(indices)))

def straight(path):
    visible = path[np.isfinite(path).all(axis=1)]
    if len(visible) < 2:
        return 0.0
    return float(np.linalg.norm(visible[-1] - visible[0]) / max(np.linalg.norm(np.diff(visible, axis=0), axis=1).sum(), 1e-08))

def features(players, ball, teams, fps=25.0, pitch=(105.3, 68.0), config=None, prepared=None):
    config = config or Config()
    rows = gap.features(players, ball, teams, fps, pitch, config) if prepared is None else prepared
    p = np.asarray(players, float) * pitch
    b = np.asarray(ball, float) * pitch
    teams = np.asarray(teams)
    width = math.ceil(0.12 * fps)
    output = []
    for row in rows:
        e = row['frame']
        pre = b[e - width:e + 1]
        post = b[e:e + width + 1]
        incoming, n_in = velocity(pre, fps)
        outgoing, n_out = velocity(post, fps)
        per_team = []
        for side in (0, 1):
            vin = []
            vout = []
            impulse = []
            counts_in = []
            counts_out = []
            complete = False
            post_distance = []
            for j in np.flatnonzero(teams == side):
                before = p[e - width:e + 1, j]
                after = p[e:e + width + 1, j]
                relative_in = before - pre
                relative_out = after - post
                distances = np.linalg.norm(np.concatenate((relative_in, relative_out)), axis=1)
                visible = np.linalg.norm(relative_out, axis=1)
                post_distance.extend(visible[np.isfinite(visible)].tolist())
                if not np.any(np.isfinite(distances) & (distances <= config.control_radius_m)):
                    continue

                def fit(path, relative):
                    if not np.isfinite(path).all() or np.any(np.linalg.norm(np.diff(path, axis=0), axis=1) * fps > config.max_player_speed_mps):
                        return (None, 0)
                    return velocity(relative, fps)
                a, ca = fit(before, relative_in)
                z, cz = fit(after, relative_out)
                if a is not None:
                    vin.append(np.linalg.norm(a))
                    counts_in.append(ca)
                if z is not None:
                    vout.append(np.linalg.norm(z))
                    counts_out.append(cz)
                if a is not None and z is not None:
                    impulse.append(np.linalg.norm(z - a))
                    complete = True
            per_team.append(np.asarray([np.log1p(min(vin, default=40.0)), np.log1p(min(vout, default=40.0)), np.log1p(min(impulse, default=40.0)), max(counts_in, default=0), max(counts_out, default=0), float(complete), min(min(post_distance, default=12.0), 12.0)]))
        difference = per_team[0] - per_team[1]
        a = np.linalg.norm(incoming) if incoming is not None else 40.0
        z = np.linalg.norm(outgoing) if outgoing is not None else 40.0
        change = np.linalg.norm(outgoing - incoming) if incoming is not None and outgoing is not None else 40.0
        cosine = float(incoming @ outgoing / max(a * z, 1e-08)) if incoming is not None and outgoing is not None else 0.0
        common = [np.log1p(a), np.log1p(z), np.log1p(change), cosine, straight(pre), straight(post), n_in, n_out]
        output.append({**row, 'difference': [*row['difference'], *difference.tolist()], 'common': [*row['common'], *np.abs(difference).tolist(), *list(map(float, common))]})
    return output
