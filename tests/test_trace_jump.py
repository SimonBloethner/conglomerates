"""C29: the trace records the floor's log-size jump (post-floor minus pre-floor) per firm and step."""
import numpy as np
import pytest

import collaborative_growth as cg
from tests.test_floor_exit import PARAMS, KW, M, N, T

F = M * N


@pytest.fixture(scope='module')
def trace(tmp_path_factory):
    """The seeded M=N=20, T=600 run of tests/test_floor_exit.py, with a trace."""
    trace_path = str(tmp_path_factory.mktemp('c29') / 'trace.npz')
    cg.model(PARAMS, trace_path=trace_path, **KW)
    z = np.load(trace_path)
    return z['jump'], z['floor']


def test_jump_shape_and_sign(trace):
    """(a) jump has shape (T+1, F) and is never negative.
    Catches: a transposed or per-step-only array, and a jump taken as pre minus post (sign flip)."""
    jump, _ = trace
    assert jump.shape == (T + 1, F)
    assert (jump >= 0).all()


def test_jump_iff_floor_hit(trace):
    """(b) (jump[1:] > 0).T == floor elementwise: a firm is lifted at step t iff it hit the floor at t.
    Catches: recording the jump at the wrong step (e.g. row step instead of step + 1) or from the wrong
    buffer (curr_idx instead of next_idx, or after renormalisation), and nonzero jumps for unlifted firms."""
    jump, floor = trace
    assert floor.shape == (F, T)
    assert ((jump[1:] > 0).T == floor).all()


def test_jump_positive_on_hits(trace):
    """(c) every recorded floor hit has a strictly positive jump, and there are hits (not vacuous).
    Catches: a jump array left at zero (e.g. never written, or overwritten) while floor hits occur."""
    jump, floor = trace
    n_hits = int(floor.sum())
    assert n_hits > 0
    assert jump[1:][floor.T].min() > 0
