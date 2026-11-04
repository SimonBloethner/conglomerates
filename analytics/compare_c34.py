#!/usr/bin/env python3
"""
C34: C33 -> C34 change in the headline statistics of the blocks the corrections touch.

    python analytics/compare_c34.py <c33_tidy.csv> <c34_tidy.csv> <c33_alpha_scatter.csv> <c34_alpha_scatter.csv>

Prints markdown tables (equal-split block by alpha; endogenous-alpha block by family; the main-block runs with
capped members) to stdout. Statistics are means over runs of the tidy columns (post-burn-in summaries).
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EQ_COLS = [('K_mean', 'K (mean)', '{:.3f}'), ('cong_capital_share_median', 'pooled capital share', '{:.3f}'),
           ('acceptance_rate', 'acceptance rate', '{:.4f}'), ('acceptance_rate_ss', 'acc. solo+solo', '{:.4f}'),
           ('acceptance_rate_sc', 'acc. solo+unit', '{:.4f}'), ('acceptance_rate_cc', 'acc. unit+unit', '{:.4f}'),
           ('pool_ruin_per_period', 'capped members / period', '{:.5f}')]
ENDO_COLS = [('alpha_adopted_mean', 'α mean', '{:.3f}'), ('alpha_adopted_median', 'α median', '{:.3f}'),
             ('alpha_adopted_share1', 'share at α = 1', '{:.3f}'), ('K_mean', 'K (mean)', '{:.2f}'),
             ('cong_capital_share_median', 'pooled capital share', '{:.3f}'),
             ('hill_exponent_median', 'Hill', '{:.3f}')]


def fmt(v, f):
    return 'n/a' if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def pair_table(old, new, by, cols, extra_new=()):
    o = old.groupby(by)[[c for c, _, _ in cols]].mean()
    n = new.groupby(by)[[c for c, _, _ in cols] + [c for c, _, _ in extra_new]].mean()
    head = [by] + [f'{lab} C33 → C34' for _, lab, _ in cols] + [f'{lab} (C34)' for _, lab, _ in extra_new]
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for k in n.index:
        cells = [str(k)] + [f'{fmt(o.loc[k, c], f)} → {fmt(n.loc[k, c], f)}' for c, _, f in cols]
        cells += [fmt(n.loc[k, c], f) for c, _, f in extra_new]
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def spearman_by_family(path):
    df = pd.read_csv(path)
    out = {}
    for fam, sub in df.groupby('log_family'):
        sub = sub.dropna(subset=['alpha_tavg', 'mean_iqr'])
        rho, p = spearmanr(sub['alpha_tavg'], sub['mean_iqr'])
        out[fam] = (len(sub), rho, p)
    return out


def main():
    old, new = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
    sp_old, sp_new = spearman_by_family(sys.argv[3]), spearman_by_family(sys.argv[4])

    eq_o, eq_n = old[old.block == 'equal-split'], new[new.block == 'equal-split']
    print('### Equal-split block (180 runs; means over 4 cost forms × 5 reps per α)\n')
    print(pair_table(eq_o, eq_n, 'alpha', EQ_COLS,
                     extra_new=[('pool_shortfall_per_period', 'shortfall / period', '{:.5f}')]))
    print()
    for lab, df in (('C33', eq_o), ('C34', eq_n)):
        n_ruin = int((df.pool_ruin_per_period > 0).sum())
        line = f'- {lab}: {n_ruin} of {len(df)} runs with capped members after burn-in; ' \
               f'mean capped members per period {df.pool_ruin_per_period.mean():.5f}'
        if 'pool_shortfall_per_period' in df:
            line += f'; mean shortfall per period {df.pool_shortfall_per_period.mean():.5f} (capital units, market mean firm size = 1)'
        print(line)
    print()

    en_o, en_n = old[old.block == 'endogenous-alpha'], new[new.block == 'endogenous-alpha']
    print('### Endogenous-α block (60 runs; means over 4 cost forms × 5 reps per family)\n')
    print(pair_table(en_o, en_n, 'log_family', ENDO_COLS))
    print()
    print('| family | n units C33 → C34 | Spearman ρ(α_tavg, mean member IQR) C33 → C34 | p (C34) |')
    print('|---|---|---|---|')
    for fam in sorted(sp_new):
        no, ro, _ = sp_old.get(fam, (0, np.nan, np.nan))
        nn, rn, pn = sp_new[fam]
        print(f'| {fam} | {no} → {nn} | {ro:.3f} → {rn:.3f} | {pn:.3g} |')
    print()
    for lab, df in (('C33', en_o), ('C34', en_n)):
        print(f'- {lab}: block mean α {df.alpha_adopted_mean.mean():.3f}, median of run medians '
              f'{df.alpha_adopted_median.median():.3f}, share at α = 1 {df.alpha_adopted_share1.mean():.3f}, '
              f'K {df.K_mean.mean():.2f}, pooled capital share {df.cong_capital_share_median.mean():.3f}, '
              f'Hill {df.hill_exponent_median.mean():.3f}')
    print()

    ids = old[(old.block != 'equal-split') & (old.block != 'endogenous-alpha') & (old.pool_ruin_per_period > 0)].scenario_id
    if len(ids):
        print('### Other runs with capped members in C33\n')
        cols = [('K_mean', 'K', '{:.3f}'), ('cong_capital_share_median', 'pooled share', '{:.4f}'),
                ('hill_exponent_median', 'Hill', '{:.4f}'), ('pool_ruin_per_period', 'capped / period', '{:.5f}')]
        print('| id | block | family | cost | α | ' + ' | '.join(f'{l} C33 → C34' for _, l, _ in cols) + ' |')
        print('|' + '---|' * (5 + len(cols)))
        for i in ids:
            o, n = old[old.scenario_id == i].iloc[0], new[new.scenario_id == i].iloc[0]
            cells = [str(i), o.block, o.log_family, o.cost_type, str(o.alpha)]
            cells += [f'{fmt(o[c], f)} → {fmt(n[c], f)}' for c, _, f in cols]
            print('| ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
