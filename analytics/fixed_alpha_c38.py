#!/usr/bin/env python3
"""
C38 Task 4: the fixed alpha = 0.9 cell (pilot D block alpha-0.9, ids 1995-2009).

    python analytics/fixed_alpha_c38.py --append-tidy   # append the 15 tidy rows to pilot_d/tidy.csv
    python analytics/fixed_alpha_c38.py --table         # pilot_d/fixed_alpha_09.csv and a markdown table

--append-tidy builds the tidy rows of the 15 runs in pilot_d/results with summarize_pilot_c.create_tidy_df,
concatenates them under the existing pilot_d/tidy.csv and writes the result only if its first 1,996 lines (header
and the 1,995 C37 rows) are byte-identical to the existing file; it stops otherwise. Medians and the other pilot D
outputs are C37 outputs and are not regenerated.
--table: by family, medians over the 5 reps of K_mean, pooled capital share (cong_capital_share_median), Hill
(hill_exponent_median), floor exits per period and the adopted alpha, for main power law alpha = 0.5, the fixed
alpha = 0.9 cell and the endogenous-alpha block (power law), all under the moment rule.
"""
import io
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pilot_design_d as pdd  # noqa: E402
from analytics.identity_c35 import load  # noqa: E402
from summarize_pilot_c import create_tidy_df  # noqa: E402

TIDY = os.path.join(ROOT, 'pilot_d', 'tidy.csv')
IDS = range(pdd.TOTAL_C37, pdd.TOTAL)
COLS = ['K_mean', 'cong_capital_share_median', 'hill_exponent_median', 'floor_exits_per_period',
        'alpha_adopted_median', 'acceptance_rate']


def append_tidy():
    old_text = open(TIDY).read()
    old = pd.read_csv(TIDY, float_precision='round_trip')   # the default parser drops the last digit
    assert len(old) == pdd.TOTAL_C37 and old.scenario_id.max() == pdd.TOTAL_C37 - 1, len(old)
    new = create_tidy_df([load(os.path.join(ROOT, 'pilot_d', 'results'), i) for i in IDS])
    assert list(new.columns) == list(old.columns), set(new.columns) ^ set(old.columns)
    buf = io.StringIO()
    pd.concat([old, new], ignore_index=True).to_csv(buf, index=False)
    text = buf.getvalue()
    n = pdd.TOTAL_C37 + 1
    assert text.splitlines(True)[:n] == old_text.splitlines(True)[:n], 'C37 rows would change'
    assert text.startswith(old_text)
    with open(TIDY, 'w') as f:
        f.write(text)
    print(f'appended {len(new)} rows; the {pdd.TOTAL_C37} C37 rows are byte-identical')


def table():
    t = pd.read_csv(TIDY)
    pl = t[t.cost_type == 'power_law']
    cells = {'main α = 0.5': pl[(pl.block == 'main') & (pl.alpha == 0.5)],
             'fixed α = 0.9': pl[pl.block == 'alpha-0.9'],
             'endogenous α': pl[pl.block == 'endogenous-alpha']}
    rows = []
    for name, d in cells.items():
        assert (d.decision_rule == 'moment').all()
        for fam, g in d.groupby('log_family', sort=False):
            rows.append(dict(cell=name, family=fam, n_reps=len(g), **g[COLS].median().to_dict()))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, 'pilot_d', 'fixed_alpha_09.csv'), index=False)
    print('| family | cell | K | pooled share | Hill | floor exits / period | adopted α | acceptance |')
    print('|---|---|---|---|---|---|---|---|')
    for fam in ('normal', 'laplace', 't3'):
        for _, r in out[out.family == fam].iterrows():
            a = '–' if pd.isna(r.alpha_adopted_median) else f'{r.alpha_adopted_median:.2f}'
            print(f'| {fam} | {r.cell} | {r.K_mean:.2f} | {r.cong_capital_share_median:.3f} | '
                  f'{r.hill_exponent_median:.3f} | {r.floor_exits_per_period:.2f} | {a} | {r.acceptance_rate:.3f} |')


if __name__ == '__main__':
    if '--append-tidy' in sys.argv:
        append_tidy()
    elif '--table' in sys.argv:
        table()
    else:
        sys.exit(__doc__)
