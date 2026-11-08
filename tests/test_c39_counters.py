"""C39 Task 1: the C38 (K, w) counters split by proposal type, and unit+unit proposals by volatility bin."""
import numpy as np
import pytest

import collaborative_growth as cg
from tests.test_c38_acceptance_by_K_w import kernel
from tests.test_floor_exit import PARAMS, KW

PAIRS = (('p', 'pt'), ('m', 'mt'), ('s1', 's1t'), ('s2', 's2t'), ('n', 'nt'))


def test_types_in_the_kernel():
    """solo+solo -> type 0, solo+unit -> type 1, unit+unit -> type 2, in the C38 column; the type axis sums to the
    C38 arrays."""
    for sizes, units, K, t in (([3.0, 1.2], (), 2, 0), ([1.0, 2.0, 1.2], [(0, 1)], 3, 1),
                               ([1.0, 1.4, 3.0, 3.0], [(0, 1), (2, 3)], 4, 2)):
        a, props, _ = kernel(np.log(sizes), units=units)
        assert props[K] > 0 and a['pt'][K, 6, t] == props[K] == a['pt'].sum()
        for c38, c39 in PAIRS:
            assert np.allclose(a[c39].sum(2), a[c38], rtol=1e-12, atol=0)


@pytest.mark.parametrize('iqr, calm', [([0.1, 0.1, 0.3, 0.3], True), ([0.3, 0.3, 0.1, 0.1], False),
                                       ([0.2, 0.2, 0.2, 0.2], False)])
def test_volatility_bin(iqr, calm):
    """Units {0, 1} (capital 2.4, the entrant) and {2, 3} (capital 6), one firm per market: every proposal is
    unit+unit with K = 4. Bin 0 iff the entrant unit's mean member IQR is lower (ties in bin 1). The binding member
    is counted once per proposal under the moment rule, in the entrant unit or not."""
    a, props, mergs = kernel(np.log([1.0, 1.4, 3.0, 3.0]), units=[(0, 1), (2, 3)], iqr=iqr)
    b = 0 if calm else 1
    assert a['vp'][4, b] == props[4] > 0 and a['vp'][4, 1 - b] == 0 and a['vp'].sum() == props[4]
    assert a['vm'][4, b] == mergs[4]
    assert a['vn'][4, b] == props[4] and np.isclose(a['vs1'][4, b], a['s1t'][4, :, 2].sum())
    assert 0 <= a['ve'][4, b] <= props[4]


@pytest.fixture(scope='module', params=['moment', 'loggain'])
def run(request):
    return cg.model(**dict(PARAMS, search_prob=0.3), **dict(KW, decision_rule=request.param))[-1]


def test_model_type_axis_sums_to_c38(run):
    """Post-burn-in: the C39 arrays summed over type are the C38 arrays; the unit+unit counts summed over w are the
    volatility-bin counts; all three types occur."""
    for k in ('proposals', 'mergers', 'binding_n'):
        assert np.array_equal(run[f'{k}_by_K_w_type'].sum(2), run[f'{k}_by_K_w'])
    for k in ('binding_sum', 'binding_sq'):
        assert np.allclose(run[f'{k}_by_K_w_type'].sum(2), run[f'{k}_by_K_w'], rtol=1e-12, atol=1e-18)
    P = run['proposals_by_K_w_type']
    assert P.shape == (PARAMS['M'] + 1, cg.K_W_BINS + 1, 3) and (P.sum((0, 1)) > 0).all()
    assert not P[:3, :, 2].any() and not P[:, :, 0].sum(1)[3:].any()   # unit+unit needs K >= 4; solo+solo is K = 2
    assert np.array_equal(run['proposals_by_K_type_volbin'].sum(1), P[:, :, 2].sum(1))
    assert np.array_equal(run['mergers_by_K_type_volbin'].sum(1), run['mergers_by_K_w_type'][:, :, 2].sum(1))
    assert np.array_equal(run['binding_entrant_by_K_type_volbin'] <= run['proposals_by_K_type_volbin'],
                          np.ones_like(run['proposals_by_K_type_volbin'], bool))
    assert run['proposals_by_K_type_volbin'][:, 0].sum() > 0 and run['proposals_by_K_type_volbin'][:, 1].sum() > 0
