#!/usr/bin/env python3
"""
C42 identity check: the accept rule (decision_rule='accept', code 4) leaves every existing pilot D run unchanged.
Compare the C39 result pickles (the C39 identity reruns of ids 0-2009 and the C39 cv runs, ids 2010-2054, both with
every C38 and C39 key) with reruns on the C42 code. C42 adds no output key, so the comparison has no exception:

    python analytics/identity_c42.py <new_dir> <out_csv> <ref_dir> [<ref_dir> ...]

For every scenario in <new_dir> (the reference is the first <ref_dir> holding it):
- identical: every field of the reference (runtime fields elapsed_seconds, ms_per_step left out) is byte-identical
  (pickle.dumps) in the rerun, and the rerun has no field the reference does not have;
- identical_tidy: the tidy row as CSV text, without runtime.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analytics.identity_c35 import RUNTIME, b, load, tidy_text  # noqa: E402


def compare(ref, new):
    diffs = [k for k in ref if k not in RUNTIME and (k not in new or b(ref[k]) != b(new[k]))]
    extra = set(new) - set(ref)
    if extra:
        diffs.append('unexpected_fields:' + '|'.join(sorted(extra)))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('new_dir')
    ap.add_argument('out_csv')
    ap.add_argument('ref_dirs', nargs='+')
    args = ap.parse_args()
    ids = sorted(int(f[9:13]) for f in os.listdir(args.new_dir) if f.startswith('scenario_'))
    rows = []
    for i in ids:
        name = f'scenario_{i:04d}.pkl.gz'
        ref_dir = next((d for d in args.ref_dirs if os.path.exists(os.path.join(d, name))), None)
        if ref_dir is None:
            continue
        ref, new = load(ref_dir, i), load(args.new_dir, i)
        diffs = compare(ref, new)
        s = ref['scenario']
        rows.append(dict(scenario_id=i, block=s['block'], log_family=s['log_family'], cost_type=s['cost_type'],
                         alpha=s['alpha'], decision_rule=s['decision_rule'], identical=not diffs,
                         identical_tidy=tidy_text(ref) == tidy_text(new), differing_fields='|'.join(diffs),
                         seconds_ref=round(ref['elapsed_seconds'], 1), seconds_c42=round(new['elapsed_seconds'], 1)))
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_id = sum(r['identical'] and r['identical_tidy'] for r in rows)
    print(f'{len(rows)} compared, {n_id} identical (every reference field and the tidy row), {len(rows) - n_id} '
          f'differ; runtime ratio C42/ref median '
          f'{np.median([r["seconds_c42"] / r["seconds_ref"] for r in rows]):.2f}')


if __name__ == '__main__':
    main()
