#!/usr/bin/env python3
"""
C38 identity check: the (K, w) counters of merger_kernel_numba are additive outputs. Compare the pilot D result
pickles (C37) with reruns on the C38 code, leaving out only the C38 keys.

    python analytics/identity_c38.py <ref_dir> <new_dir> <out_csv>

For every scenario in <new_dir> that has a result in <ref_dir>:
- identical: the whole result object without the runtime fields (elapsed_seconds, ms_per_step) and, in the new
  one, without the C38 keys (run_pilot_c.C38_KEYS), compared as pickle.dumps bytes; differing_fields lists the
  fields whose bytes differ, and any field beyond the C38 keys;
- identical_tidy: the tidy row (summarize_pilot_c.extract_post_burnin_metrics) as CSV text, without runtime;
- the counters are consistent (counters_consistent): proposals_by_K_w and mergers_by_K_w summed over w equal
  proposals_by_K and mergers_by_K; mergers <= proposals in every cell; binding_n <= proposals, and binding_n is 0
  where there are no proposals; the mean w of every w column lies inside its bin; binding_sq >= binding_sum^2 / n.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analytics.identity_c35 import RUNTIME, b, load, tidy_text  # noqa: E402
from run_pilot_c import C38_KEYS  # noqa: E402


def strip(res, keys=()):
    return {k: v for k, v in res.items() if k not in RUNTIME and k not in keys}


def compare(ref, new):
    diffs = [k for k in ref if k not in RUNTIME and (k not in new or b(ref[k]) != b(new[k]))]
    extra = set(new) - set(ref) - set(C38_KEYS)
    if extra:
        diffs.append('unexpected_fields:' + '|'.join(sorted(extra)))
    missing = set(C38_KEYS) - set(new)
    if missing:
        diffs.append('missing_c38:' + '|'.join(sorted(missing)))
    return diffs


def counter_checks(new):
    P, Mg = new['proposals_by_K_w'], new['mergers_by_K_w']
    n, s1, s2, ws = new['binding_n_by_K_w'], new['binding_sum_by_K_w'], new['binding_sq_by_K_w'], new['w_sum_by_K_w']
    edges = new['K_w_edges']
    ok = (np.array_equal(P.sum(1), new['proposals_by_K']) and np.array_equal(Mg.sum(1), new['mergers_by_K'])
          and (Mg <= P).all() and (n <= P).all())
    with np.errstate(invalid='ignore', divide='ignore'):
        wbar = ws / P
        var_ok = np.all((s2 - np.where(n > 0, s1 ** 2 / np.maximum(n, 1), 0)) >= -1e-12)
    for j in range(1, P.shape[1]):
        sel = P[:, j] > 0
        ok &= bool(np.all((wbar[sel, j] >= edges[j - 1] - 1e-12) & (wbar[sel, j] <= edges[j] + 1e-12)))
    return bool(ok and var_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref_dir')
    ap.add_argument('new_dir')
    ap.add_argument('out_csv')
    args = ap.parse_args()
    ids = sorted(int(f[9:13]) for f in os.listdir(args.new_dir) if f.startswith('scenario_')
                 and os.path.exists(os.path.join(args.ref_dir, f)))
    rows = []
    for i in ids:
        ref, new = load(args.ref_dir, i), load(args.new_dir, i)
        diffs = compare(ref, new)
        s = ref['scenario']
        rows.append(dict(scenario_id=i, block=s['block'], log_family=s['log_family'], cost_type=s['cost_type'],
                         alpha=s['alpha'], decision_rule=s['decision_rule'],
                         identical=b(strip(ref)) == b(strip(new, C38_KEYS)) and not diffs,
                         identical_tidy=tidy_text(ref) == tidy_text(new), differing_fields='|'.join(diffs),
                         counters_consistent=counter_checks(new),
                         seconds_c37=round(ref['elapsed_seconds'], 1), seconds_c38=round(new['elapsed_seconds'], 1)))
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_id = sum(r['identical'] and r['identical_tidy'] for r in rows)
    n_ck = sum(r['counters_consistent'] for r in rows)
    print(f'{len(rows)} compared, {n_id} identical (object without runtime and C38 keys, and tidy row), '
          f'{len(rows) - n_id} differ; counters consistent in {n_ck}/{len(rows)}; runtime ratio C38/C37 median '
          f'{np.median([r["seconds_c38"] / r["seconds_c37"] for r in rows]):.2f}')


if __name__ == '__main__':
    main()
