"""Symmetric linear/quadratic bases for the same observation-only features."""
import numpy as np
from .observed_gap_features import ROLE_FEATURES, COMMON_FEATURES
CONTEXT = ('observed_ball_in', 'observed_ball_out', 'observed_ball_impulse', 'observed_turn_cosine', 'missing_window', 'gap_length', 'observed_pre_points', 'observed_post_points')
CONTEXT_INDEX = tuple((COMMON_FEATURES.index(name) for name in CONTEXT))
PAIRS = tuple(((i, j) for i in range(len(CONTEXT)) for j in range(i, len(CONTEXT))))

def names(kind):
    if kind == 'linear':
        return (list(ROLE_FEATURES), list(COMMON_FEATURES))
    if kind != 'quadratic':
        raise ValueError('Unknown control feature basis')
    role = [*ROLE_FEATURES, *(r + '*' + c for r in ROLE_FEATURES for c in CONTEXT)]
    common = [*COMMON_FEATURES, *(CONTEXT[i] + '*' + CONTEXT[j] for i, j in PAIRS), *(r + '^2' for r in ROLE_FEATURES)]
    return (role, common)

def design(rows, kind):
    role_names, common_names = names(kind)
    d = np.asarray([r['difference'] for r in rows], float).reshape(len(rows), len(ROLE_FEATURES))
    c = np.asarray([r['common'] for r in rows], float).reshape(len(rows), len(COMMON_FEATURES))
    if not np.isfinite(d).all() or not np.isfinite(c).all():
        raise ValueError('Nonfinite control features')
    if kind == 'linear':
        return (d, c)
    context = c[:, CONTEXT_INDEX]
    role = np.concatenate((d, (d[:, :, None] * context[:, None, :]).reshape(len(rows), len(ROLE_FEATURES) * len(CONTEXT))), axis=1)
    pairs = np.column_stack([context[:, i] * context[:, j] for i, j in PAIRS])
    common = np.concatenate((c, pairs, d * d), axis=1)
    if role.shape[1] != len(role_names) or common.shape[1] != len(common_names):
        raise ValueError('Control basis name mismatch')
    return (role, common)

def probabilities(d, c, model):
    d = np.asarray(d) / model['role_scale']
    c = np.asarray(c) / model['common_scale']
    side = d @ model['role_beta']
    shared = c @ model['common_beta'] + model['bias']
    scores = np.column_stack((np.zeros(len(d)), shared + side, shared - side))
    value = np.exp(scores - scores.max(axis=1, keepdims=True))
    return value / value.sum(axis=1, keepdims=True)
