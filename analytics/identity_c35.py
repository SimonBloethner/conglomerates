#!/usr/bin/env python3
"""
C35 identity check: the C35 kernel diagnostics are additive outputs. Compare C34 and C35 result pickles
field by field, byte for byte, leaving out only the C35 keys.

    python analytics/identity_c35.py <c34_dir> <c35_dir> <out_csv> [--ids FILE]

For every scenario in <c35_dir> (or in FILE):
- every pickled field of the C34 result except the runtime is compared as pickle.dumps bytes; the `summary`
  dict on all keys except the C35 key `acceptance_by_K` (the C35 summary must have exactly the C34 keys
  plus that one);
- the C35 result may hold only the C35 fields beyond the C34 ones (C35_FIELDS);
- the tidy row (summarize_pilot_c.extract_post_burnin_metrics) is compared as CSV text, without runtime.
It also checks the new counters against the existing ones over the whole run:
  proposals_by_K[:, 2] == proposals_ss and proposals_by_K[:, 3:].sum(1) == proposals_sc + proposals_cc per
  period (same for mergers), and that the binding-gain histogram holds one count per post-burn-in proposal
  under loggain (none under replay).
"""
import argparse
import csv
import gzip
import io
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from summarize_pilot_c import extract_post_burnin_metrics  # noqa: E402

RUNTIME = ('elapsed_seconds', 'ms_per_step')
C35_SUMMARY_KEYS = ('acceptance_by_K',)
C35_FIELDS = ('proposals_by_K', 'mergers_by_K', 'proposals_by_K_per_period', 'mergers_by_K_per_period',
              'binding_gain_hist_by_K', 'binding_gain_edges',
              'proposals_ss_per_period', 'proposals_sc_per_period', 'proposals_cc_per_period',
              'mergers_ss_per_period', 'mergers_sc_per_period', 'mergers_cc_per_period')


def load(d, i):
    with gzip.open(os.path.join(d, f'scenario_{i:04d}.pkl.gz'), 'rb') as f:
        return pickle.load(f)


def b(v):
    return pickle.dumps(v, protocol=4)


def tidy_text(result):
    row = extract_post_burnin_metrics(result)
    buf = io.StringIO()
    csv.writer(buf).writerow([v for k, v in row.items() if k not in RUNTIME])
    return buf.getvalue()


def compare(ref, new):
    diffs = []
    for k in ref:
        if k in RUNTIME:
            continue
        a, c = ref[k], new.get(k)
        if k == 'summary':
            if set(c) - set(a) != set(C35_SUMMARY_KEYS) or set(a) - set(c):
                diffs.append('summary_keys')
            c = {kk: vv for kk, vv in c.items() if kk not in C35_SUMMARY_KEYS}
        if b(a) != b(c):
            diffs.append(k)
    extra = set(new) - set(ref) - set(C35_FIELDS)
    if extra:
        diffs.append('unexpected_fields:' + '|'.join(sorted(extra)))
    return diffs


def counter_checks(new):
    s = new['scenario']
    P, Mg = new['proposals_by_K_per_period'], new['mergers_by_K_per_period']
    ok = (np.array_equal(P[:, 2], new['proposals_ss_per_period'])
          and np.array_equal(P[:, 3:].sum(1), new['proposals_sc_per_period'] + new['proposals_cc_per_period'])
          and np.array_equal(Mg[:, 2], new['mergers_ss_per_period'])
          and np.array_equal(Mg[:, 3:].sum(1), new['mergers_sc_per_period'] + new['mergers_cc_per_period'])
          and not P[:, :2].any() and not Mg[:, :2].any()
          and np.array_equal(new['proposals_by_K'], P[s['burn_in']:].sum(0))
          and np.array_equal(new['mergers_by_K'], Mg[s['burn_in']:].sum(0)))
    H = new['binding_gain_hist_by_K']
    if s.get('decision_rule', 'loggain') == 'loggain':
        hist_ok = np.array_equal(H.sum(1), new['proposals_by_K'])
    else:
        hist_ok = not H.any()
    return bool(ok), bool(hist_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref_dir')
    ap.add_argument('new_dir')
    ap.add_argument('out_csv')
    ap.add_argument('--ids')
    args = ap.parse_args()
    if args.ids:
        ids = [int(x) for x in open(args.ids) if x.strip()]
    else:
        ids = sorted(int(f[9:13]) for f in os.listdir(args.new_dir) if f.startswith('scenario_'))
    rows = []
    for i in ids:
        ref, new = load(args.ref_dir, i), load(args.new_dir, i)
        diffs = compare(ref, new)
        ck, hk = counter_checks(new)
        s = ref['scenario']
        rows.append(dict(scenario_id=i, block=s['block'], cell_id=s['cell_id'], log_family=s['log_family'],
                         cost_type=s['cost_type'], alpha=s['alpha'], merge_thresh=s['merge_thresh'],
                         decision_rule=s.get('decision_rule', 'loggain'),
                         identical_fields=not diffs, identical_tidy=tidy_text(ref) == tidy_text(new),
                         differing_fields='|'.join(diffs), counters_consistent=ck, binding_hist_consistent=hk,
                         seconds_c34=round(ref['elapsed_seconds'], 1), seconds_c35=round(new['elapsed_seconds'], 1)))
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_id = sum(r['identical_fields'] and r['identical_tidy'] for r in rows)
    n_ck = sum(r['counters_consistent'] and r['binding_hist_consistent'] for r in rows)
    print(f'{len(rows)} compared, {n_id} identical (fields and tidy row), {len(rows) - n_id} differ; '
          f'counters consistent in {n_ck}/{len(rows)}; '
          f'runtime ratio C35/C34 median {np.median([r["seconds_c35"] / r["seconds_c34"] for r in rows]):.2f}')


if __name__ == '__main__':
    main()
