"""Use only already available D evidence to guard recovery, never evaluation labels."""
import numpy as np

def available_dead(masks):
    dead = np.asarray(masks['dead'], bool).copy()
    if 'entry_backdated' in masks:
        unavailable = masks['entry_backdated'] & (masks['entry_available_at'] > np.arange(len(dead)))
        dead[unavailable] = False
    if 'unowned_added' in masks:
        unavailable = masks['unowned_added'] & (masks['unowned_available_at'] > np.arange(len(dead)))
        dead[unavailable] = False
        dead |= masks['unowned_confirmed']
    return dead

def certified_releases(masks, recoveries):
    """Consume a current recovery certificate only after the last available D.

    Certificates come from none_recovery.choices, including opponent coverage
    and exclusive bounded co-motion. No elapsed-time or ball-only release.
    """
    dead = available_dead(masks)
    n = len(dead)
    last = np.maximum.accumulate(np.where(dead, np.arange(n), -1))
    release = np.zeros(n, bool)
    for event in recoveries:
        i = event['frame']
        indices = np.asarray(event['support']['indices'])
        if not 0 <= i < n or event['available_at'] != i or event['source'] != 'observed_recovery' or (indices.ndim != 1) or (not len(indices)) or (indices < 0).any() or (indices > i).any():
            raise ValueError('A currently available recovery certificate is required')
        if indices.min() > last[i]:
            release[i] = True
    return release

def support_starts(recoveries, n):
    """Earliest actual support at each currently available recovery frame."""
    starts = np.full(n, -1, np.int64)
    for event in recoveries:
        i = event['frame']
        indices = np.asarray(event['support']['indices'])
        if not 0 <= i < n or event['available_at'] != i or (not len(indices)) or (indices < 0).any() or (indices > i).any():
            raise ValueError('Currently available actual recovery support required')
        value = int(indices.min())
        starts[i] = value if starts[i] < 0 else min(starts[i], value)
    return starts
