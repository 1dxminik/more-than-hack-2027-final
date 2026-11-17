"""Fixed causal motion estimator for production inference; no paths or IO."""
import numpy as np
WINDOW = 11

def causal_mean11(values, width=WINDOW):
    """Historical mean: up to 11 finite consecutive values, reset after NaN.

    Vectorised prefix sums replace only the research implementation's loops.
    The 11 velocity samples span at most 12 observed positions; no future frame
    is used, and startup uses the available shorter history.
    """
    if width != WINDOW:
        raise ValueError('This experiment fixes the motion window at 11')
    original = np.asarray(values, dtype=float)
    if len(original) == 0:
        return original.copy()
    flat = original.reshape(len(original), -1)
    valid = np.isfinite(flat)
    index = np.arange(len(flat))[:, None]
    last_missing = np.maximum.accumulate(np.where(valid, -1, index), axis=0)
    start = np.maximum(last_missing + 1, index - WINDOW + 1)
    cumulative = np.concatenate((np.zeros((1, flat.shape[1])), np.cumsum(np.where(valid, flat, 0.0), axis=0)), axis=0)
    columns = np.arange(flat.shape[1])[None, :]
    sums = cumulative[index + 1, columns] - cumulative[start, columns]
    count = np.maximum(1, index - start + 1)
    result = np.full(flat.shape, np.nan)
    np.divide(sums, count, out=result, where=valid)
    return result.reshape(original.shape)
