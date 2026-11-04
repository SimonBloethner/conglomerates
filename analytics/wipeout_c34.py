#!/usr/bin/env python3
"""
C34 diagnostic: record every pooling event with a capped member in one pilot scenario.

    python analytics/wipeout_c34.py <scenario_id> <out_csv>

Runs the scenario as run_pilot_c.run_scenario does (without its wipeout assertion) with
process_conglomerate_pooling_numba wrapped. For every unit with a capped member it writes the step, the unit
size K, Phi(K), the members' capital weights and relative returns, the unit's capital-weighted gross return
sum_j w_j (1 + r_j) and whether the unit was wiped out (sum_j w_j (1 + r_j) <= Phi(K) in exact arithmetic).
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import collaborative_growth as cg  # noqa: E402

CAP = -1.0 + 1e-12


def main():
    sid, out_csv = int(sys.argv[1]), sys.argv[2]
    s = json.load(open('pilot_c/scenarios.json'))[sid]
    rows = []
    orig = cg.process_conglomerate_pooling_numba

    def wrapped(*a):
        out = orig(*a)
        members_arr, size_arr, active_ids, rel, log_states, phi = a[:6]
        step = a[8]
        new_log = out[0]
        if out[2] > 0:
            for uid in active_ids:
                m = members_arr[uid, :size_arr[uid]]
                capped = np.abs((new_log[m] - log_states[m]) - np.log1p(CAP)) < 1e-6
                if not capped.any():
                    continue
                w = np.exp(log_states[m] - log_states[m].max())
                w = w / w.sum()
                r = np.expm1(rel[m])
                K = len(m)
                rows.append(dict(step=int(step), K=K, phi=float(phi[K]), n_capped=int(capped.sum()),
                                 wiped=bool(capped.all()), gross_return=float((w * (1 + r)).sum()),
                                 weights=' '.join(f'{x:.4g}' for x in w), rel_returns=' '.join(f'{x:.6g}' for x in r),
                                 log_rel_returns=' '.join(f'{x:.4g}' for x in rel[m])))
        return out

    cg.process_conglomerate_pooling_numba = wrapped
    try:
        hp = cg.model(M=s['M'], N=s['N'], T=s['T'], alpha=s['alpha'], search_prob=s['merge_thresh'],
                      window=s['lookback'], cost_form=s['cost_type'], b0=s['c0'], b1=s['c1'], b2=s['c2'],
                      seed=s['seed'], growth_process=s['growth_process'], family=s['log_family'],
                      mu_range=tuple(s['mu_range']), iqr_range=tuple(s['sigma_range']), floor_c=s['floor_c'],
                      cross_corr=s['cross_corr'], metric_every=s['metric_every'], sharing_rule=s['sharing_rule'],
                      burn_in=s['burn_in'], alpha_endogenous=s.get('alpha_endogenous', False), g=s.get('g'),
                      market_size_fixed=s.get('market_size_fixed', False),
                      decision_rule=s.get('decision_rule', 'loggain'))[-1]
    finally:
        cg.process_conglomerate_pooling_numba = orig
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['step'])
        w.writeheader()
        w.writerows(rows)
    sm = hp['summary']
    print(f"scenario {sid}: {len(rows)} unit-events with capped members, "
          f"pool_ruin_total {sm['pool_ruin_total']}, pool_wipeout_total {sm['pool_wipeout_total']}")
    for r in rows:
        if r['wiped']:
            print('WIPEOUT', r)


if __name__ == '__main__':
    main()
