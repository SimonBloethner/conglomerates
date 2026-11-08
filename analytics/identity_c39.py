#!/usr/bin/env python3
"""
C39 identity check: the proposal-type counters (Task 1) and the cv rule (Task 2) leave every existing pilot D run
unchanged. Compare the pilot D result pickles (C37 / C38) with reruns on the C39 code, leaving out only the keys the
reference does not have.

    python analytics/identity_c39.py <ref_dir> <new_dir> <out_csv>

For every scenario in <new_dir> that has a result in <ref_dir>:
- identical: every field of the reference (runtime fields elapsed_seconds, ms_per_step left out) is byte-identical
  (pickle.dumps) in the rerun, and the rerun has no fields beyond the reference's except C38_KEYS and C39_KEYS
  (the C37 pickles have neither, the C38 alpha-0.9 pickles have the C38 keys, which are then compared too);
- identical_tidy: the tidy row as CSV text, without runtime;
- counters_consistent: the C39 arrays summed over proposal type equal the C38 arrays; the unit+unit (type 2)
  proposals and mergers summed over w equal the volatility-bin counts; binding counts and the binding-member
  counts do not exceed the proposals.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analytics.identity_c35 import RUNTIME, b, load, tidy_text  # noqa: E402
from run_pilot_c import C38_KEYS, C39_KEYS  # noqa: E402

NEW_KEYS = set(C38_KEYS) | set(C39_KEYS)


def compare(ref, new):
    diffs = [k for k in ref if k not in RUNTIME and (k not in new or b(ref[k]) != b(new[k]))]
    extra = set(new) - set(ref) - NEW_KEYS
    if extra:
        diffs.append('unexpected_fields:' + '|'.join(sorted(extra)))
    missing = NEW_KEYS - set(new)
    if missing:
        diffs.append('missing_new:' + '|'.join(sorted(missing)))
    return diffs


def counter_checks(new):
    ok = True
    for k in ('proposals', 'mergers', 'binding_sum', 'binding_sq', 'binding_n'):
        t, c = new[f'{k}_by_K_w_type'], new[f'{k}_by_K_w']
        # float sums: the type split adds in a different order, so allow rounding (binding gains are ~1e-4)
        same = np.array_equal(t.sum(2), c) if t.dtype.kind == "i" else np.allclose(t.sum(2), c, rtol=1e-12, atol=1e-15)
        ok &= bool(same)
    P, Mg = new['proposals_by_K_w_type'][:, :, 2].sum(1), new['mergers_by_K_w_type'][:, :, 2].sum(1)
    ok &= bool(np.array_equal(new['proposals_by_K_type_volbin'].sum(1), P))
    ok &= bool(np.array_equal(new['mergers_by_K_type_volbin'].sum(1), Mg))
    vp = new['proposals_by_K_type_volbin']
    ok &= bool((new['binding_n_by_K_type_volbin'] <= vp).all() and (new['binding_entrant_by_K_type_volbin'] <= vp).all())
    ok &= bool((new['mergers_by_K_w_type'] <= new['proposals_by_K_w_type']).all())
    return ok


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
                         alpha=s['alpha'], decision_rule=s['decision_rule'], identical=not diffs,
                         identical_tidy=tidy_text(ref) == tidy_text(new), differing_fields='|'.join(diffs),
                         counters_consistent=counter_checks(new),
                         seconds_ref=round(ref['elapsed_seconds'], 1), seconds_c39=round(new['elapsed_seconds'], 1)))
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_id = sum(r['identical'] and r['identical_tidy'] for r in rows)
    n_ck = sum(r['counters_consistent'] for r in rows)
    print(f'{len(rows)} compared, {n_id} identical (every reference field and the tidy row), {len(rows) - n_id} '
          f'differ; counters consistent in {n_ck}/{len(rows)}; runtime ratio C39/ref median '
          f'{np.median([r["seconds_c39"] / r["seconds_ref"] for r in rows]):.2f}')


if __name__ == '__main__':
    main()
