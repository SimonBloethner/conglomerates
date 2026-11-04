#!/usr/bin/env python3
"""
C34 identity check: compare two sets of pilot result pickles field by field, byte for byte.

    python analytics/identity_c34.py <ref_dir> <new_dir> <out_csv> [--ids FILE] [--all-summary]

For every scenario present in both directories:
- every pickled field except the runtime (`elapsed_seconds`, `ms_per_step`) is compared as
  pickle.dumps(field) bytes; the `summary` dict is compared on the reference's keys only, unless
  --all-summary (C34b vs C34c: both have the C34 keys);
- the tidy row (summarize_pilot_c.extract_post_burnin_metrics) is compared as CSV text on the reference's
  columns, without the runtime columns.
Writes one row per scenario: identical_fields, identical_tidy, the differing fields, and the C34 counters
(pool_ruin_total, pool_shortfall_per_period, pool_wipeout_total) of the new run.
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
C34_SUMMARY_KEYS = ('pool_shortfall_per_period', 'pool_ruin_total', 'pool_wipeout_total')


def load(d, i):
    with gzip.open(os.path.join(d, f'scenario_{i:04d}.pkl.gz'), 'rb') as f:
        return pickle.load(f)


def field_bytes(v):
    return pickle.dumps(v, protocol=4)


def tidy_text(result, columns):
    row = extract_post_burnin_metrics(result)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([row.get(c, '') for c in columns])
    return buf.getvalue(), row


def compare(ref, new, all_summary):
    diffs = []
    for k in ref:
        if k in RUNTIME:
            continue
        a, b = ref[k], new.get(k)
        if k == 'summary':
            keys = list(a) if not all_summary else sorted(set(a) | set(b))
            a = {kk: a.get(kk) for kk in keys}
            b = {kk: b.get(kk) for kk in keys}
        if field_bytes(a) != field_bytes(b):
            diffs.append(k)
    extra = sorted(set(new) - set(ref))
    if extra:
        diffs.append('new_fields:' + '|'.join(extra))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref_dir')
    ap.add_argument('new_dir')
    ap.add_argument('out_csv')
    ap.add_argument('--ids')
    ap.add_argument('--all-summary', action='store_true')
    args = ap.parse_args()

    if args.ids:
        ids = [int(x) for x in open(args.ids) if x.strip()]
    else:
        ids = sorted(int(f[9:13]) for f in os.listdir(args.new_dir) if f.startswith('scenario_'))
    rows = []
    for i in ids:
        ref, new = load(args.ref_dir, i), load(args.new_dir, i)
        diffs = compare(ref, new, args.all_summary)
        ref_row = extract_post_burnin_metrics(ref)
        cols = [c for c in ref_row if c not in RUNTIME and c not in C34_SUMMARY_KEYS]  # C33 has no C34 columns
        if args.all_summary:
            cols = [c for c in extract_post_burnin_metrics(new) if c not in RUNTIME]
        t_ref, _ = tidy_text(ref, cols)
        t_new, new_row = tidy_text(new, cols)
        s = new['summary']
        rows.append(dict(
            scenario_id=i, block=ref['scenario']['block'], log_family=ref['scenario']['log_family'],
            sharing_rule=ref['scenario']['sharing_rule'],
            ref_pool_ruin_per_period=ref['summary'].get('pool_ruin_per_period'),
            pool_ruin_total=s.get('pool_ruin_total'), pool_shortfall_per_period=s.get('pool_shortfall_per_period'),
            pool_wipeout_total=s.get('pool_wipeout_total'),
            identical_fields=not diffs, identical_tidy=t_ref == t_new, differing_fields='|'.join(diffs)))
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_id = sum(r['identical_fields'] and r['identical_tidy'] for r in rows)
    print(f'{len(rows)} compared, {n_id} identical (fields and tidy row), {len(rows) - n_id} differ')


if __name__ == '__main__':
    main()
