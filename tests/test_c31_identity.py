"""C31: the model corrections must not leak into configurations they do not concern.

Constants recorded on the unmodified code (head c144756) for the seeded configuration of
tests/test_floor_exit.py (M=N=20, T=600, loggain, alpha_endogenous=False).

Identity is expected only in runs without pool-ruin events: limited liability (C31) changes any run in which
a member's delta reaches -1, under either sharing rule. The seeded laplace proportional run has none, so it
must stay bit-identical and guards the main configuration; the seeded equal-sharing run has four.
"""
import hashlib

import numpy as np
import pytest

import collaborative_growth as cg
from tests.test_floor_exit import PARAMS, KW

PROPORTIONAL_C144756 = dict(
    K_mean=3.426865671641791,
    cong_capital_share_median=0.7363162421228838,
    mergers_per_period=8.08,
    floor_exits_per_period=7.934,
    sha256='36ccb5bae861d937bcec669d645da894be4d0a9fc77ad28af73964cde7ae3d67',
)

# Pre-change values of the equal-sharing run. Four pool-ruin events occur in this seeded run (steps 204, 353,
# 497, 530 on c144756, where they triggered an exit and a reset to log size 0), so C31 changes it by design;
# the values are kept for reference (also in diagnostics/deviations.md) and not asserted.
EQUAL_C144756 = dict(
    K_mean=2.458955223880597,
    cong_capital_share_median=0.3359852344423004,
    mergers_per_period=6.638,
    floor_exits_per_period=6.214,
    sha256='d331af00dbf6f06a3b49536433ab82b8b98a57200d3eb62543146a8dbb3835db',
)

RUIN_STEP = np.log1p(-1.0 + 1e-12)  # log-size change of a member whose delta is capped


def test_proportional_unchanged():
    """Proportional sharing, alpha_endogenous=False: bit-identical to c144756.
    Catches any leak of the endogenous-alpha changes (cong_alpha must equal share everywhere) or of the
    limited-liability change into the main configuration."""
    hp = cg.model(PARAMS, **KW)[-1]
    s = hp['summary']
    for k in ('K_mean', 'cong_capital_share_median', 'mergers_per_period', 'floor_exits_per_period'):
        assert abs(s[k] - PROPORTIONAL_C144756[k]) < 1e-12, k
    assert hashlib.sha256(np.asarray(hp['final_log_states']).tobytes()).hexdigest() == PROPORTIONAL_C144756['sha256']


@pytest.fixture(scope='module')
def equal_run(tmp_path_factory):
    """Seeded equal-sharing run with a trace; records every member whose delta was capped (step, firm)."""
    events = []
    orig = cg.process_conglomerate_pooling_numba

    def wrapped(*a):
        new_log_states, pools, ruin = orig(*a)
        cong_firms, cong_size, active_ids, log_states_curr, step = a[0], a[1], a[2], a[4], a[9]
        for cid in active_ids:
            for f in cong_firms[cid, :cong_size[cid]]:
                if abs((new_log_states[f] - log_states_curr[f]) - RUIN_STEP) < 1e-6:
                    events.append((int(step), int(f)))
        return new_log_states, pools, ruin

    trace_path = str(tmp_path_factory.mktemp('c31eq') / 'trace.npz')
    cg.process_conglomerate_pooling_numba = wrapped
    try:
        hp = cg.model(PARAMS, trace_path=trace_path, **dict(KW, sharing_rule='equal'))[-1]
    finally:
        cg.process_conglomerate_pooling_numba = orig
    return hp, np.load(trace_path), events


def test_equal_ruin_events_floor_and_standalone(equal_run):
    """Equal sharing: pool-ruin events occur in this seeded run, so instead of identity assert that they are
    counted and that every ruined firm hit the floor in that step and is standalone in the following step.
    Catches: ruin still routed through exit_ (no floor hit), a ruined member left affiliated, or a counter
    that does not match the capped deltas."""
    hp, z, events = equal_run
    ruin = hp['pool_ruin_per_period']
    assert ruin.sum() > 0
    assert len(events) == int(ruin.sum())
    floor, cong = z['floor'], z['cong']
    for step, firm in events:
        assert floor[firm, step], (step, firm)
        assert cong[step + 1, firm] == -1, (step, firm)
