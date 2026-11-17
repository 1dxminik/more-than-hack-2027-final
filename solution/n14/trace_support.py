"""Per-call provenance capture without replacing functions or modifying module state."""
from contextvars import ContextVar

SINK = ContextVar("n14_provenance_sink", default=None)


def current_sink():
    return SINK.get()
