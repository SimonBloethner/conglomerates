#!/usr/bin/env python3
"""
C43: the main block at alpha 0.6, 0.7, 0.8 (pilot D block alpha-gap, ids 2100-2144).

    python analytics/alpha_gap_c43.py --design-identity <stored scenarios.json> <out_csv>
    python analytics/alpha_gap_c43.py --resummary <alpha-0.9 results dir> <out_csv>
    python analytics/alpha_gap_c43.py --append-tidy <alpha-gap results dir>
    python analytics/alpha_gap_c43.py --table      # pilot_d/alpha_gap.csv and pilot_d/alpha_gap_report.md

--design-identity: every one of the 2,100 scenarios in the stored scenarios.json (before C43) is byte-identical
  (json.dumps, indent 2, key order kept) to the same scenario in the regenerated pilot_d/scenarios.json, and the
  stored file's text up to its last scenario is a prefix of the new file's text.
--resummary: the 15 alpha-0.9 result pickles (ids 1995-2009) are summarised again with
  summarize_pilot_c.create_tidy_df on the current code; substituted into the stored tidy.csv, the file must be
  byte-identical to the stored one (every column, runtime included).
--append-tidy: as analytics/accept_c42.py: the 45 tidy rows are appended to pilot_d/tidy.csv, which is written only
  if the existing 2,100 rows stay byte-identical.
--table: pilot_d/alpha_gap.csv, one row per family x alpha x rep of the alpha-gap block (COLS); the report gives
  the medians over the 5 reps for each family x alpha beside main alpha = 0.5 and alpha-0.9, nothing else.
"""
import csv
import io
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pilot_design_d as pdd  # noqa: E402
from analytics.identity_c35 import load  # noqa: E402

D = os.path.join(ROOT, 'pilot_d')
TIDY = os.path.join(D, 'tidy.csv')
IDS = range(pdd.TOTAL_C42, pdd.TOTAL)
IDS_09 = range(pdd.TOTAL_C37, pdd.TOTAL_C38)
COLS = ['K_mean', 'cong_capital_share_median', 'hill_exponent_median', 'hhi_within_median',
        'floor_exits_per_period', 'voluntary_exits_per_period', 'acceptance_rate',
        'floor_hit_rate_member_status', 'floor_hit_rate_standalone_status', 'top10pct_aggregate_median',
        'K_eff_over_K_median', 'elapsed_seconds']
FAMILIES = ('normal', 'laplace', 't3')


def design_identity(stored_path, out_csv):
    stored_text = open(stored_path).read()
    new_text = open(os.path.join(D, 'scenarios.json')).read()
    stored, new = json.loads(stored_text), json.loads(new_text)
    assert len(stored) == pdd.TOTAL_C42 and len(new) == pdd.TOTAL, (len(stored), len(new))
    rows = []
    for s, n in zip(stored, new[:pdd.TOTAL_C42]):
        same = json.dumps(s, indent=2) == json.dumps(n, indent=2)
        rows.append(dict(scenario_id=s['scenario_id'], block=s['block'], identical=same,
                         differing_fields='|'.join(sorted(k for k in set(s) | set(n) if s.get(k, '<missing>')
                                                          != n.get(k, '<missing>')))))
    prefix = stored_text.rstrip()
    assert prefix.endswith('}\n]'), 'unexpected end of the stored file'
    prefix_ok = new_text.startswith(prefix[:-2])          # the stored text up to the closing brace of id 2099
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_id = sum(r['identical'] for r in rows)
    print(f'design: {n_id:,}/{len(rows):,} existing scenarios byte-identical; text prefix identical: {prefix_ok}')
    return n_id == len(rows) and prefix_ok


def resummary(results_dir, out_csv):
    from summarize_pilot_c import create_tidy_df
    old_text = open(TIDY).read()
    old = pd.read_csv(TIDY, float_precision='round_trip')   # the default parser drops the last digit
    assert len(old) == pdd.TOTAL_C42, len(old)
    again = create_tidy_df([load(results_dir, i) for i in IDS_09])
    assert list(again.columns) == list(old.columns)
    assert list(again.scenario_id) == list(IDS_09) and (again.block == 'alpha-0.9').all()
    # the 15 stored rows replaced by the re-summarised ones, concatenated as the file was built (C38 append)
    ids = old.scenario_id
    whole = pd.concat([old[ids < IDS_09.start], again, old[ids >= IDS_09.stop]], ignore_index=True)
    buf = io.StringIO()
    whole.to_csv(buf, index=False)
    new_text = buf.getvalue()
    whole_ok = new_text == old_text
    old_lines, new_lines = old_text.splitlines(True), new_text.splitlines(True)
    pos = {i: k + 1 for k, i in enumerate(old.scenario_id)}               # line of each id (header is line 0)
    rows = [dict(scenario_id=i, identical=old_lines[pos[i]] == new_lines[pos[i]]) for i in IDS_09]
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_id = sum(r['identical'] for r in rows)
    print(f'alpha-0.9 re-summarised: {n_id}/{len(rows)} tidy rows byte-identical; whole tidy.csv identical: '
          f'{whole_ok}')
    return n_id == len(rows) and whole_ok


def append_tidy(results_dir):
    from summarize_pilot_c import create_tidy_df
    old_text = open(TIDY).read()
    old = pd.read_csv(TIDY, float_precision='round_trip')
    assert len(old) == pdd.TOTAL_C42 and old.scenario_id.max() == pdd.TOTAL_C42 - 1, len(old)
    new = create_tidy_df([load(results_dir, i) for i in IDS])
    assert list(new.columns) == list(old.columns), set(new.columns) ^ set(old.columns)
    assert (new.decision_rule == 'moment').all() and (new.block == 'alpha-gap').all()
    assert list(new.scenario_id) == list(IDS)
    buf = io.StringIO()
    pd.concat([old, new], ignore_index=True).to_csv(buf, index=False)
    text = buf.getvalue()
    assert text.startswith(old_text), 'existing rows would change'
    with open(TIDY, 'w') as f:
        f.write(text)
    print(f'appended {len(new)} rows; the {pdd.TOTAL_C42} existing rows are byte-identical')


def table():
    t = pd.read_csv(TIDY)
    pl = t[(t.cost_type == 'power_law') & (t.decision_rule == 'moment')]
    gap = pl[pl.block == 'alpha-gap']
    assert len(gap) == 45
    per_rep = gap[['scenario_id', 'block', 'log_family', 'alpha', 'rep', 'seed'] + COLS].rename(
        columns={'log_family': 'family'})
    per_rep.to_csv(os.path.join(D, 'alpha_gap.csv'), index=False)

    cells = pd.concat([pl[(pl.block == 'main') & (pl.alpha == 0.5)], gap, pl[pl.block == 'alpha-0.9']])
    med = cells.groupby(['log_family', 'alpha', 'block'], sort=False).agg(
        n_reps=('rep', 'size'), **{c: (c, 'median') for c in COLS}).reset_index()
    assert (med.n_reps == 5).all() and len(med) == 15

    head = ['α', 'cell', 'K', 'pooled share', 'Hill', 'HHI within', 'floor exits / period',
            'voluntary exits / period', 'acceptance', 'floor hit (member)', 'floor hit (standalone)',
            'top 10% share', 'K_eff / K', 'seconds']
    fmt = ['.2f', '.3f', '.3f', '.3f', '.2f', '.3f', '.3f', '.4f', '.4f', '.3f', '.3f', '.0f']
    out = ['# C43: main block at α ∈ {0.6, 0.7, 0.8}',
           '',
           'Pilot D, moment rule, power law, main IQR range, T = 11,000, retained window 8,000–11,000. Medians over '
           '5 reps (the same seeds in every row of a family). α = 0.5: block `main`; α = 0.6–0.8: block '
           '`alpha-gap` (ids 2100–2144); α = 0.9: block `alpha-0.9` (ids 1995–2009). Per-rep values: '
           '`pilot_d/alpha_gap.csv`.',
           '']
    for fam in FAMILIES:
        out += [f'## {fam}', '', '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
        for _, r in med[med.log_family == fam].sort_values('alpha').iterrows():
            vals = [format(r[c], f) for c, f in zip(COLS, fmt)]
            out.append(f'| {r.alpha:.1f} | {r.block} | ' + ' | '.join(vals) + ' |')
        out.append('')
    with open(os.path.join(D, 'alpha_gap_report.md'), 'w') as f:
        f.write('\n'.join(out))
    print('\n'.join(out))


if __name__ == '__main__':
    a = sys.argv
    if '--design-identity' in a:
        sys.exit(0 if design_identity(a[2], a[3]) else 1)
    elif '--resummary' in a:
        sys.exit(0 if resummary(a[2], a[3]) else 1)
    elif '--append-tidy' in a:
        append_tidy(a[2])
    elif '--table' in a:
        table()
    else:
        sys.exit(__doc__)
