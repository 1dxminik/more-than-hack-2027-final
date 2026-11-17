"""Experimental dense H/A/D TCN. Observation-only, half-local inference."""
import numpy as np
import torch
from torch import nn
FPS = 25.0
GLOBAL = 13
TEAM = 12
CHANNELS = GLOBAL + 2 * TEAM
DILATIONS = (1, 2, 4, 8, 16, 32)
RADIUS = 2 * sum(DILATIONS)
SWAP = np.r_[np.arange(GLOBAL), np.arange(GLOBAL + TEAM, CHANNELS), np.arange(GLOBAL, GLOBAL + TEAM)]
LABELS = np.array(list('HAD'))

def velocity(x, lag=1):
    result = np.zeros_like(x)
    ok = np.zeros(x.shape[:-1], bool)
    ok[lag:] = np.isfinite(x[lag:]).all(axis=-1) & np.isfinite(x[:-lag]).all(axis=-1)
    result[lag:] = np.nan_to_num((x[lag:] - x[:-lag]) * (FPS / lag), nan=0.0, posinf=0.0, neginf=0.0)
    result[~ok] = 0
    return (result, ok)

def features(players, ball, teams, pitch=(105.3, 68.0)):
    """No labels, player IDs, clock, events or match identifiers enter features."""
    p = np.asarray(players, dtype=np.float64) * pitch
    b = np.asarray(ball, dtype=np.float64) * pitch
    t = np.asarray(teams)
    if p.shape != (len(b), len(t), 2) or b.shape != (len(b), 2) or (not np.isin(t, [0, 1]).all()):
        raise ValueError('Tracking shape/team contract')
    if not np.isfinite(pitch).all() or min(pitch) <= 0:
        raise ValueError('Positive pitch required')
    n = len(b)
    valid = np.isfinite(b).all(axis=1)
    bp = np.nan_to_num(b / pitch - 0.5, nan=0.0, posinf=0.0, neginf=0.0)
    bp[~valid] = 0
    seen = np.maximum.accumulate(np.where(valid, np.arange(n), -125))
    age = np.minimum(np.arange(n) - seen, 125) / 125.0
    bv, vok = velocity(b)
    bv5, vok5 = velocity(b, 5)
    acc, aok = velocity(bv)
    aok &= vok & np.r_[False, vok[:-1]]
    acc[~aok] = 0
    columns = [bp[:, 0], bp[:, 1], valid, age, bv[:, 0] / 20, bv[:, 1] / 20, np.linalg.norm(bv, axis=1) / 40, vok, np.linalg.norm(acc, axis=1) / 100, aok, bv5[:, 0] / 20, bv5[:, 1] / 20, vok5]
    pv, pok = velocity(p)
    speed = np.linalg.norm(pv, axis=2)
    for side in (0, 1):
        ids = np.flatnonzero(t == side)
        if not len(ids):
            columns.extend([np.zeros(n)] * TEAM)
            continue
        pp = p[:, ids]
        observed = np.isfinite(pp).all(axis=2)
        dist = np.linalg.norm(pp - b[:, None], axis=2)
        dist[~np.isfinite(dist)] = np.inf
        nearest = np.argmin(dist, axis=1)
        row = np.arange(n)
        near_ok = np.isfinite(dist[row, nearest])
        tied = np.isclose(dist, dist[row, nearest, None], rtol=0, atol=1e-12) & np.isfinite(dist)
        tied_count = np.maximum(tied.sum(axis=1), 1)
        sorted_dist = np.sort(dist, axis=1)
        ds = [np.minimum(sorted_dist[:, min(k, len(ids) - 1)], 30) / 30 for k in range(3)]
        nearest_position = (np.nan_to_num(pp, nan=0.0, posinf=0.0, neginf=0.0) * tied[:, :, None]).sum(axis=1) / tied_count[:, None]
        delta = np.nan_to_num(nearest_position - b, nan=0.0, posinf=0.0, neginf=0.0)
        delta[~near_ok] = 0
        rel_ok = ((pok[:, ids] & tied).sum(axis=1) == tied_count) & vok & near_ok
        nearest_velocity = (pv[:, ids] * tied[:, :, None]).sum(axis=1) / tied_count[:, None]
        rel = np.linalg.norm(nearest_velocity - bv, axis=1)
        rel[~rel_ok] = 0
        count = pok[:, ids].sum(axis=1)
        columns.extend([observed.sum(axis=1) / 11, *ds, near_ok, delta[:, 0] / 30, delta[:, 1] / 30, rel / 30, rel_ok, speed[:, ids].sum(axis=1) / np.maximum(count, 1) / 10, ((speed[:, ids] < 0.5) & pok[:, ids]).sum(axis=1) / np.maximum(count, 1), count / 11])
    result = np.asarray(columns, np.float32).T.copy()
    if result.shape != (n, CHANNELS) or not np.isfinite(result).all():
        raise ValueError('Invalid dense features')
    return result

class Residual(nn.Module):

    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation), nn.ReLU(), nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation))

    def forward(self, x):
        return torch.relu(x + self.net(x))

class DenseHAD(nn.Module):

    def __init__(self):
        super().__init__()
        self.input = nn.Conv1d(CHANNELS, 32, 1)
        self.blocks = nn.Sequential(*(Residual(32, d) for d in DILATIONS))
        self.output = nn.Conv1d(32, 3, 1)

    def forward(self, x):
        return self.output(self.blocks(torch.relu(self.input(x))))

def normalize(x, mean, std):
    return np.clip((x - mean) / std, -10.0, 10.0).astype(np.float32)

def load(path):
    model = DenseHAD()
    with np.load(path, allow_pickle=False) as f:
        keys = set(model.state_dict())
        if set(f.files) != keys | {'mean', 'std'}:
            raise ValueError('Unexpected checkpoint schema')
        mean, std = (f['mean'], f['std'])
        if mean.shape != (CHANNELS,) or std.shape != mean.shape or (not np.isfinite(mean).all()) or (not np.isfinite(std).all()) or (std <= 0).any():
            raise ValueError('Invalid normalization')
        weights = {k: torch.from_numpy(f[k]) for k in keys}
        if any((not torch.isfinite(v).all() for v in weights.values())):
            raise ValueError('Invalid weights')
        model.load_state_dict(weights, strict=True)
    return (model.eval(), mean, std)

@torch.inference_mode()
def probabilities(x, model, mean, std, chunk=4096):
    if chunk <= 0:
        raise ValueError('Positive chunk required')
    x = normalize(x, mean, std)
    result = np.empty((len(x), 3), np.float32)
    model.eval()
    for start in range(0, len(x), chunk):
        stop = min(len(x), start + chunk)
        left, right = (max(0, start - RADIUS), min(len(x), stop + RADIUS))
        a = torch.from_numpy(x[left:right].T.copy()).unsqueeze(0)
        b = torch.from_numpy(x[left:right, SWAP].T.copy()).unsqueeze(0)
        pred = (model(a).softmax(1) + model(b).softmax(1)[:, [1, 0, 2]]) / 2
        result[start:stop] = pred[0, :, start - left:stop - left].T.numpy()
    return result
