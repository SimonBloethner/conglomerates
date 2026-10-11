"""C31: limited liability under equal sharing (process_conglomerate_pooling_numba)."""
import numpy as np

import collaborative_growth as cg
from tests.test_floor_exit import PARAMS, KW
from tests.test_c31_identity import equal_run  # noqa: F401  (shared seeded equal-sharing fixture)

CAP = -1.0 + 1e-12


def _pool_two_member_unit():
    """Two-member unit, sizes 100 and 0.5, alpha = 0.3, equal sharing, Phi = 0.
    The large member returns -20%, the small one 0%: the pool loss shared equally per firm is about
    12 times the small member's capital, so its delta is far below -1 before the cap."""
    cong_firms = np.array([[0, 1]], dtype=np.int32)
    cong_size = np.array([2], dtype=np.int16)
    active = np.array([0], dtype=np.int64)
    log_returns = np.array([np.log(0.8), 0.0])
    log_states = np.log(np.array([100.0, 0.5]))
    costs = np.zeros(3)
    cong_alpha = np.full(1, 0.3)
    out = cg.process_conglomerate_pooling_numba(cong_firms, cong_size, active, log_returns, log_states, costs,
                                                0.3, False, 50, 0, 0, cong_alpha, False)
    return log_states, out


def test_small_member_delta_capped():
    """Hand-built unit: the small member's delta is capped at -1 + 1e-12, its new log state is finite, and
    pool_ruin counts exactly 1 (the large member's delta, about -0.17, is not capped).
    Catches: the old exit-and-reset branch (new log state 0 = log 1, i.e. a gain), an uncapped delta
    (log1p of a value below -1 is NaN), and a counter that misses or double-counts."""
    log_states, (new_log_states, pools, ruin) = _pool_two_member_unit()
    assert np.isfinite(new_log_states).all()
    delta_small = np.expm1(new_log_states[1] - log_states[1])
    assert abs(delta_small - CAP) < 1e-15
    assert ruin == 1
    delta_large = np.expm1(new_log_states[0] - log_states[0])
    assert -0.25 < delta_large < -0.1


def test_ruined_member_at_floor_and_standalone(equal_run):  # noqa: F811
    """After the floor block a ruined member is at the floor and standalone. The floor block is inline in
    model(), so this is checked on the seeded equal-sharing model run (fixture from test_c31_identity): every
    capped member hit the floor in that step and is standalone in the next trace row.
    Catches: a ruined member that is not lifted (no floor hit) or stays affiliated."""
    hp, z, events = equal_run
    assert len(events) > 0
    for step, firm in events:
        assert z['floor'][firm, step]
        assert z['cong'][step + 1, firm] == -1


def test_proportional_never_ruins():
    """Seeded M=N=20 proportional-sharing run: no member's delta is capped.
    Catches: the cap firing under proportional sharing (which would change the main configuration)."""
    hp = cg.model(PARAMS, **KW)[-1]
    assert hp['pool_ruin_per_period'].sum() == 0
