#!/usr/bin/env python3
"""
C37: the tables of diagnostics/c37_note.md -> pilot_d/tables_c37.md (python analytics/pilot_d.py --tables).

Reads pilot_d/ and pilot_c/ outputs (tidy, comparison_c_vs_d, acceptance_by_K_pooled, status panel, decomposition,
event study, burstiness, variance_ratio, alpha_scatter) and analytics/benchmarks.csv. Every cell shows the moment
value (pilot D) and its loggain twin (pilot C, same seeds) as "moment / loggain", medians over the 5 reps unless
stated; the homogeneous block has both rules in pilot D.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMS = ['normal', 'laplace', 't3']
FORD = {'normal': 0, 'laplace': 1, 't3': 2, 'student_t': 3}


def rd(pilot, name):
    return pd.read_csv(os.path.join(ROOT, f'pilot_{pilot}', name))


def f(x, fmt='{:.3f}'):
    return '—' if x is None or pd.isna(x) else fmt.format(x)


def pair(m, lg, fmt='{:.3f}'):
    return f'{f(m, fmt)} / {f(lg, fmt)}'


def kstar_lookup():
    bm = pd.read_csv(os.path.join(ROOT, 'analytics', 'benchmarks.csv'))
    return {(r.family, r.cost_type, float(r.cost_multiplier), float(r.alpha), int(r.M)): int(r.K_star)
            for r in bm.itertuples()}


KS = None


def kstar(fam, cost='power_law', mult=1.0, alpha=None, M=50):
    return KS.get((fam, cost, float(mult), float(alpha), int(M)), np.nan)


def table(lines, header, rows):
    lines += ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    lines.append('')


def sort_fam(df, cols):
    return df.sort_values(cols, key=lambda c: c.map(FORD) if c.name == 'log_family' else c)


# ------------------------------------------------------------------------------------------------- 1. main block
def main_block(comp, lines):
    sub = sort_fam(comp[(comp.block == 'main') & (comp.cost_type == 'power_law') & (comp.alpha > 0)],
                   ['log_family', 'alpha'])
    lines += ['## 1. Main block (power law, p = 0.05; α = 0 has no units under either rule)', '']
    lines += ['**1a. Scope.** K = post-burn-in mean unit size (K_mean), medians over 5 reps; ΔK = median over reps '
              'of the paired difference moment − loggain [p25, p75]; K\\* from analytics/benchmarks.csv (M = 50, '
              'IQR 0.2).', '']
    rows = []
    for r in sub.itertuples():
        ks = kstar(r.log_family, alpha=r.alpha)
        rows.append([r.log_family, f'{r.alpha:g}', f(ks, '{:.0f}'), pair(r.K_mean_moment_median, r.K_mean_loggain_median),
                     f'{r.K_mean_diff_median:+.2f} [{r.K_mean_diff_p25:+.2f}, {r.K_mean_diff_p75:+.2f}]',
                     pair(r.K_mean_moment_median / ks, r.K_mean_loggain_median / ks),
                     pair(r.cong_capital_share_median_moment_median, r.cong_capital_share_median_loggain_median),
                     pair(r.acceptance_rate_moment_median, r.acceptance_rate_loggain_median),
                     pair(r.mergers_per_period_moment_median, r.mergers_per_period_loggain_median, '{:.1f}'),
                     pair(r.floor_exits_per_period_moment_median, r.floor_exits_per_period_loggain_median, '{:.1f}'),
                     pair(r.voluntary_exits_per_period_moment_median, r.voluntary_exits_per_period_loggain_median,
                          '{:.2f}')])
    table(lines, ['family', 'α', 'K*', 'K mom / log', 'ΔK [IQR]', 'K/K* mom / log', 'pooled share', 'acceptance',
                  'mergers / period', 'floor exits / period', 'voluntary exits / period'], rows)
    lines += ['**1b. Distribution and floor hits.** Hill = median over post-burn-in snapshots of the pooled Hill '
              'exponent; HHI = within-market HHI (mean over markets, median over snapshots); top-10% = share of '
              'aggregate capital held by the largest 10% of units. Floor-hit rates in two definitions:',
              '- **all-firm** (tidy `floor_hit_rate_member`, `_standalone`): floor hits of members (standalones) per '
              'firm-period over *all* firms, so the member rate rises with the share of firms in units;',
              '- **per status** (`floor_hit_rate_*_status`): hits per firm-period *of that status*, the all-firm rate '
              'divided by the share of firms in units (standalones: by one minus it), the share taken from the '
              'post-burn-in snapshots every 100 periods. Hazard ratio = member / standalone per-status rate.', '']
    rows = []
    for r in sub.itertuples():
        hm = r.floor_hit_rate_member_status_moment_median / r.floor_hit_rate_standalone_status_moment_median
        hl = r.floor_hit_rate_member_status_loggain_median / r.floor_hit_rate_standalone_status_loggain_median
        rows.append([r.log_family, f'{r.alpha:g}',
                     pair(r.hill_exponent_median_moment_median, r.hill_exponent_median_loggain_median),
                     pair(r.hhi_within_median_moment_median, r.hhi_within_median_loggain_median, '{:.4f}'),
                     pair(r.top10pct_aggregate_median_moment_median, r.top10pct_aggregate_median_loggain_median),
                     pair(r.member_firm_share_moment_median, r.member_firm_share_loggain_median),
                     pair(r.floor_hit_rate_member_moment_median, r.floor_hit_rate_member_loggain_median, '{:.4f}'),
                     pair(r.floor_hit_rate_standalone_moment_median, r.floor_hit_rate_standalone_loggain_median,
                          '{:.4f}'),
                     pair(r.floor_hit_rate_member_status_moment_median, r.floor_hit_rate_member_status_loggain_median,
                          '{:.4f}'),
                     pair(r.floor_hit_rate_standalone_status_moment_median,
                          r.floor_hit_rate_standalone_status_loggain_median, '{:.4f}'),
                     pair(hm, hl, '{:.2f}')])
    table(lines, ['family', 'α', 'Hill', 'HHI within', 'top-10% unit share', 'share of firms in units',
                  'floor-hit member (all-firm)', 'floor-hit standalone (all-firm)', 'floor-hit member (per status)',
                  'floor-hit standalone (per status)', 'hazard ratio'], rows)
    # all four cost forms, K only
    sub = sort_fam(comp[comp.block == 'main'], ['log_family', 'cost_type', 'alpha'])
    sub = sub[sub.alpha.isin([0.1, 0.3, 0.5])]
    lines += ['**1c. K under every cost form** (α ∈ {0.1, 0.3, 0.5}; K mom / log, K\\*).', '']
    rows = [[r.log_family, r.cost_type, f'{r.alpha:g}', f(kstar(r.log_family, r.cost_type, alpha=r.alpha), '{:.0f}'),
             pair(r.K_mean_moment_median, r.K_mean_loggain_median),
             pair(r.cong_capital_share_median_moment_median, r.cong_capital_share_median_loggain_median)]
            for r in sub.itertuples()]
    table(lines, ['family', 'cost', 'α', 'K*', 'K mom / log', 'pooled share mom / log'], rows)


# ------------------------------------------------------------------------------------------ 2. homogeneous block
def homogeneous(tidy_d, comp, lines):
    h = tidy_d[tidy_d.block == 'homogeneous']
    hm = h.groupby(['log_family', 'alpha', 'decision_rule'])[['K_mean', 'acceptance_rate',
                                                             'cong_capital_share_median']].median()
    het = comp[(comp.block == 'main') & (comp.cost_type == 'power_law')].set_index(['log_family', 'alpha'])
    lines += ['## 2. Homogeneous block (every market at IQR 0.2) against the heterogeneous main cells', '',
              '**2a. K by rule and economy** (medians over 5 reps; the homogeneous cells use the seeds of the main '
              'power-law cell of their family). Heterogeneous loggain = pilot C main.', '']
    rows, dec = [], []
    for fam in FAMS:
        for a in [0.05, 0.1, 0.3, 0.5]:
            mh, lh = hm.loc[(fam, a, 'moment')], hm.loc[(fam, a, 'loggain')]
            r = het.loc[(fam, a)]
            ks = kstar(fam, alpha=a)
            rows.append([fam, f'{a:g}', f(ks, '{:.0f}'), f(r.K_mean_moment_median), f(r.K_mean_loggain_median),
                         f(mh.K_mean), f(lh.K_mean),
                         pair(r.acceptance_rate_moment_median, r.acceptance_rate_loggain_median),
                         pair(mh.acceptance_rate, lh.acceptance_rate),
                         pair(r.cong_capital_share_median_moment_median, r.cong_capital_share_median_loggain_median),
                         pair(mh.cong_capital_share_median, lh.cong_capital_share_median)])
            if not pd.isna(ks):
                dec.append(dict(family=fam, alpha=a, K_star=ks, K_mom_hom=mh.K_mean, K_log_hom=lh.K_mean,
                                K_mom_het=r.K_mean_moment_median, K_log_het=r.K_mean_loggain_median))
    table(lines, ['family', 'α', 'K*', 'K mom het', 'K log het', 'K mom hom', 'K log hom', 'acceptance het mom / log',
                  'acceptance hom mom / log', 'pooled share het mom / log', 'pooled share hom mom / log'], rows)
    d = pd.DataFrame(dec)
    d['total'] = d.K_star - d.K_log_het
    d['a_attrition'] = d.K_star - d.K_mom_hom
    d['b_transfer'] = d.K_mom_hom - d.K_mom_het
    d['c_verif_het'] = d.K_mom_het - d.K_log_het
    d['c_verif_hom'] = d.K_mom_hom - d.K_log_hom
    d['b_transfer_log'] = d.K_log_hom - d.K_log_het
    lines += ['**2b. Decomposition of the consent gap K\\* − K_loggain (heterogeneous)**, in members and as shares of '
              'the total:',
              '- (a) attrition and feasibility = K\\* − K_moment,hom (no transfer conflict, little estimation noise in '
              'the transfer: what remains between the moment rule and K\\*);',
              '- (b) transfer conflict = K_moment,hom − K_moment,het (the same rule, heterogeneous volatilities);',
              '- (c) verifiability = K_moment − K_loggain, in each economy (het, the third term of the sum; hom for '
              'reference).',
              '(a) + (b) + (c het) = total. The alternative order (a) + (c hom) + (b under loggain, K_log,hom − '
              'K_log,het) also sums to the total.', '']
    rows = []
    for r in d.itertuples():
        t = r.total
        rows.append([r.family, f'{r.alpha:g}', f'{r.K_star:.0f}', f'{t:.2f}',
                     f'{r.a_attrition:.2f} ({r.a_attrition / t:.0%})', f'{r.b_transfer:.2f} ({r.b_transfer / t:.0%})',
                     f'{r.c_verif_het:.2f} ({r.c_verif_het / t:.0%})', f'{r.c_verif_hom:.2f} ({r.c_verif_hom / t:.0%})',
                     f'{r.b_transfer_log:.2f} ({r.b_transfer_log / t:.0%})'])
    table(lines, ['family', 'α', 'K*', 'K* − K_log,het', '(a) attrition + feasibility', '(b) transfer conflict',
                  '(c) verifiability, het', '(c) verifiability, hom', '(b) under loggain'], rows)
    d.to_csv(os.path.join(ROOT, 'pilot_d', 'consent_gap_decomposition.csv'), index=False)
    return d


def acceptance_by_K(lines):
    pd_ = rd('d', 'acceptance_by_K_pooled.csv')
    pc_ = rd('c', 'acceptance_by_K_pooled.csv')
    sel = lambda df, blk: df[(df.block == blk) & (df.alpha == 0.3) & (df.p == 0.05) & (df.cost_multiplier == 1.0)]
    mh = sel(pd_, 'homogeneous')
    src = {'mom het': sel(pd_, 'main'), 'log het': sel(pc_, 'main'),
           'mom hom': mh[mh.decision_rule == 'moment'], 'log hom': mh[mh.decision_rule == 'loggain']}
    lines += ['**2c. Acceptance and binding-gain median by K, α = 0.3, p = 0.05** (pooled over 5 reps; binding gain '
              '= min over members of gain(merged) − max(0, gain(current)), median; columns mom het / log het / '
              'mom hom / log hom).', '']
    for fam in FAMS:
        lines += [f'*{fam}*', '']
        rows = []
        for K in list(range(2, 13)) + [15, 20]:
            acc, bind, props = [], [], []
            for k in ('mom het', 'log het', 'mom hom', 'log hom'):
                x = src[k][(src[k].family == fam) & (src[k].K == K)]
                acc.append(f(x.acceptance.iloc[0]) if len(x) else '—')
                bind.append(f(x.binding_median.iloc[0], '{:.1e}') if len(x) else '—')
                props.append(str(int(x.proposals.iloc[0])) if len(x) else '0')
            rows.append([K, ' / '.join(acc), ' / '.join(bind), ' / '.join(props)])
        table(lines, ['K', 'acceptance', 'binding median', 'proposals'], rows)


# ---------------------------------------------------------------------------------------------------- 3. nu block
def nu_block(tidy_d, comp, lines):
    nu = tidy_d[tidy_d.block == 'nu']
    t3 = tidy_d[(tidy_d.block == 'main') & (tidy_d.cost_type == 'power_law') & (tidy_d.log_family == 't3')
                & tidy_d.alpha.isin(nu.alpha.unique())]
    cols = ['K_mean', 'hill_exponent_median', 'cong_capital_share_median', 'acceptance_rate', 'floor_hit_rate_member',
            'floor_hit_rate_standalone', 'floor_hit_rate_member_status', 'floor_hit_rate_standalone_status',
            'floor_exits_per_period']
    g = pd.concat([nu.assign(nu_=nu.nu), t3.assign(nu_=3.0)]).groupby(['nu_', 'alpha'])[cols].median()
    vr = rd('d', 'variance_ratio.csv')
    vr = vr[(vr.family == 't') & vr.source.str.startswith('iid')].set_index('nu')
    tw = comp[(comp.block == 'main') & (comp.cost_type == 'power_law') & (comp.log_family == 't3')].set_index('alpha')
    lines += ['## 3. ν block (Student-t, power law, main IQR range, moment rule)', '',
              'ν = 3 is the main t₃ cell (same seeds; same market IQRs for all ν). σ̂²/σ²: median [q05, q95] over '
              '200,000 i.i.d. 50-period windows at IQR 0.2 (pilot_d/variance_ratio.csv). Floor-hit rates per status '
              '(definition in 1b). K for ν = 3 under loggain (pilot C) in brackets.', '']
    rows = []
    for (n, a), r in g.iterrows():
        v = vr.loc[n]
        rows.append([f'{n:g}', f'{a:g}', f(r.K_mean) + (f' [{tw.loc[a].K_mean_loggain_median:.3f}]' if n == 3.0 else ''),
                     f(r.hill_exponent_median), f(r.cong_capital_share_median), f(r.acceptance_rate),
                     f(r.floor_hit_rate_member_status, '{:.4f}'), f(r.floor_hit_rate_standalone_status, '{:.4f}'),
                     f(r.floor_hit_rate_member_status / r.floor_hit_rate_standalone_status, '{:.2f}'),
                     f(r.floor_exits_per_period, '{:.1f}'),
                     f'{v.q50:.2f} [{v.q05:.2f}, {v.q95:.2f}]', f'{v.share_below_half:.3f}'])
    table(lines, ['ν', 'α', 'K', 'Hill', 'pooled share', 'acceptance', 'floor-hit member (per status)',
                  'floor-hit standalone (per status)', 'hazard ratio', 'floor exits / period', 'σ̂²/σ² median [q05, q95]',
                  'share σ̂²/σ² < ½'], rows)
    lines += ['**3b. σ̂²/σ² over 50-period windows (all rows of pilot_d/variance_ratio.csv)**', '']
    v = rd('d', 'variance_ratio.csv')
    rows = [[r.family, f(r.nu, '{:g}'), r.source, f'{r.n_windows:,}', f(r.mean), f(r.q50), f(r.q05), f(r.q25),
             f(r.q75), f(r.q95), f(r.share_below_half), f(r.share_above_2)] for r in v.itertuples()]
    table(lines, ['family', 'ν', 'source', 'windows', 'mean', 'median', 'q05', 'q25', 'q75', 'q95', 'share < ½',
                  'share > 2'], rows)


# ------------------------------------------------------------------------------------------- 4. Proposition 3
def prop3(lines):
    lines += ['## 4. Proposition 3 under the moment rule (27 traces: main, power law, α ∈ {0.05, 0.1, 0.3}, reps 0–2)',
              '', 'Means over the 3 reps; moment (pilot D traces) / loggain (pilot C traces, C29/C35).', '']
    sp = {p: rd(p, 'status_panel_summary.csv').groupby(['family', 'alpha']).mean(numeric_only=True) for p in 'dc'}
    lines += ['**4a. Status panel**: owner-return gap member − standalone within (market, size decile) cells, '
              'weighted by member firm-periods; by K at t (the unit size in the period, not at entry); hazard ratio = '
              'member / standalone floor-hit rate per status-period (exact, from the trace).', '']
    rows = []
    for fam in FAMS:
        for a in [0.05, 0.1, 0.3]:
            d, c = sp['d'].loc[(fam, a)], sp['c'].loc[(fam, a)]
            rows.append([fam, f'{a:g}', pair(d.diff_fe_mean, c.diff_fe_mean, '{:.5f}'),
                         pair(d.diff_fe_median, c.diff_fe_median, '{:.5f}'),
                         pair(d.diff_K2_mean, c.diff_K2_mean, '{:.5f}'), pair(d.diff_K3_4_mean, c.diff_K3_4_mean, '{:.5f}'),
                         pair(d.diff_K5p_mean, c.diff_K5p_mean, '{:.5f}'),
                         pair(d.floor_rate_member, c.floor_rate_member, '{:.4f}'),
                         pair(d.floor_rate_standalone, c.floor_rate_standalone, '{:.4f}'),
                         pair(d.floor_rate_member / d.floor_rate_standalone,
                              c.floor_rate_member / c.floor_rate_standalone, '{:.3f}')])
    table(lines, ['family', 'α', 'gap mean', 'gap median', 'gap K = 2', 'gap K = 3–4', 'gap K ≥ 5',
                  'floor rate member', 'floor rate standalone', 'hazard ratio'], rows)
    dc = {p: rd(p, 'status_panel_decomposition.csv').groupby(['family', 'alpha']).mean(numeric_only=True) for p in 'dc'}
    lines += ['**4b. Ordinary / ruin decomposition** of the total gap: A = ordinary periods, B + C = ruin protection '
              '(C35 definitions); share ordinary = A / total.', '']
    rows = []
    for fam in FAMS:
        for a in [0.05, 0.1, 0.3]:
            d, c = dc['d'].loc[(fam, a)], dc['c'].loc[(fam, a)]
            rows.append([fam, f'{a:g}', pair(d.gap_total_mean, c.gap_total_mean, '{:.5f}'),
                         pair(d.decomp_ordinary, c.decomp_ordinary, '{:.5f}'),
                         pair(d.decomp_floor_periods + d.decomp_composition, c.decomp_floor_periods + c.decomp_composition,
                              '{:.5f}'),
                         pair(d.share_ordinary, c.share_ordinary, '{:.2f}')])
    table(lines, ['family', 'α', 'total gap', 'A ordinary', 'B + C ruin', 'share ordinary'], rows)
    chk = rd('d', 'status_panel_decomposition.csv')
    lines += [f'Checks (pilot D traces): floor hits aligned with the jump in {int(chk.check_floor_jump_aligned.sum())}/27; '
              f'trace floor hits equal the pilot D result pickle in {int(chk.check_floor_matches_pickle.sum())}/27; '
              f'max |A + B + C − total| = {chk.decomp_residual.abs().max():.1e}.', '']
    es = {p: rd(p, 'event_study_summary.csv').groupby(['family', 'alpha']).mean(numeric_only=True) for p in 'dc'}
    lines += ['**4c. Event study** (difference-in-differences of log-share growth, joiner − matched standalone, '
              'window 50; owner path net of recapitalisation; medians over events, means over reps).', '']
    rows = []
    for fam in FAMS:
        for a in [0.05, 0.1, 0.3]:
            d, c = es['d'].loc[(fam, a)], es['c'].loc[(fam, a)]
            rows.append([fam, f'{a:g}', pair(d.n_matched, c.n_matched, '{:.0f}'),
                         pair(d.did_itt_median, c.did_itt_median, '{:.5f}'),
                         pair(d.did_owner_median, c.did_owner_median, '{:.5f}'),
                         pair(d.did_owner_K2_median, c.did_owner_K2_median, '{:.5f}'),
                         pair(d.did_owner_K3_4_median, c.did_owner_K3_4_median, '{:.5f}'),
                         pair(d.did_owner_K5p_median, c.did_owner_K5p_median, '{:.5f}')])
    table(lines, ['family', 'α', 'matched events', 'DiD (ITT) median', 'DiD owner median', 'owner, K = 2 at entry',
                  'owner, K = 3–4', 'owner, K ≥ 5'], rows)
    bu = {p: rd(p, 'burstiness_summary.csv').groupby(['family', 'alpha']).mean(numeric_only=True) for p in 'dc'}
    lines += ['**4d. Burstiness of floor hits within markets** (index of dispersion D of hits per market-period; '
              'binomial reference D ≈ 1 − p).', '']
    rows = []
    for fam in FAMS:
        for a in [0.05, 0.1, 0.3]:
            d, c = bu['d'].loc[(fam, a)], bu['c'].loc[(fam, a)]
            rows.append([fam, f'{a:g}', pair(d.hits_per_market_period, c.hits_per_market_period),
                         pair(d.D_median, c.D_median), f'{d.D_binomial:.3f}',
                         pair(d.share_periods_3plus, c.share_periods_3plus, '{:.4f}'),
                         pair(d.share_periods_3plus_binom, c.share_periods_3plus_binom, '{:.4f}')])
    table(lines, ['family', 'α', 'hits / market-period', 'D median', 'D binomial (moment)', 'share ≥ 3 hits',
                  'share ≥ 3, binomial'], rows)


# ------------------------------------------------------------------------------------------- 5. other blocks
def block_table(comp, lines, block, by, title, metrics, alphas=None, where=None, kstar_cols=None):
    sub = comp[(comp.block == block) & (comp.alpha > 0)]   # alpha = 0: no units under either rule
    if alphas is not None:
        sub = sub[sub.alpha.isin(alphas)]
    if where is not None:
        sub = sub[where(sub)]
    sub = sort_fam(sub, by)
    lines += [f'**{title}** (medians over 5 reps; moment / loggain; ΔK = median paired difference)', '']
    rows = []
    for r in sub.itertuples():
        row = [f'{getattr(r, b):g}' if isinstance(getattr(r, b), float) else str(getattr(r, b)) for b in by]
        if kstar_cols:
            row.append(f(kstar_cols(r), '{:.0f}'))
        row.append(pair(r.K_mean_moment_median, r.K_mean_loggain_median))
        row.append(f'{r.K_mean_diff_median:+.2f}')
        for m, fmt in metrics:
            row.append(pair(getattr(r, m + '_moment_median'), getattr(r, m + '_loggain_median'), fmt))
        rows.append(row)
    names = {'cong_capital_share_median': 'pooled share', 'hill_exponent_median': 'Hill', 'acceptance_rate': 'acceptance',
             'floor_exits_per_period': 'floor exits / period', 'alpha_adopted_median': 'adopted α (median)',
             'floor_hit_rate_member_status': 'floor-hit member (per status)',
             'floor_hit_rate_standalone_status': 'floor-hit standalone (per status)',
             'voluntary_exits_per_period': 'voluntary exits / period', 'hhi_within_median': 'HHI within'}
    table(lines, list(by) + (['K*'] if kstar_cols else []) + ['K', 'ΔK'] + [names.get(m, m) for m, _ in metrics], rows)


STD = [('cong_capital_share_median', '{:.3f}'), ('hill_exponent_median', '{:.3f}'), ('acceptance_rate', '{:.3f}'),
       ('floor_exits_per_period', '{:.1f}')]
A3 = [0.1, 0.3, 0.5]


def other_blocks(comp, lines):
    lines += ['## 5. The other blocks under the moment rule', '']
    block_table(comp, lines, 'endogenous-alpha', ['log_family', 'cost_type'], '5a. Endogenous α (start 0.1)',
                [('alpha_adopted_median', '{:.2f}'), ('cong_capital_share_median', '{:.3f}'),
                 ('hill_exponent_median', '{:.3f}'), ('voluntary_exits_per_period', '{:.2f}')])
    am, al = rd('d', 'alpha_scatter.csv'), rd('c', 'alpha_scatter.csv')
    rows = []
    for fam in FAMS:
        x, y = am[am.log_family == fam], al[al.log_family == fam]
        rows.append([fam, pair(x.alpha_tavg.mean(), y.alpha_tavg.mean()),
                     pair((x.alpha_tavg >= 0.95).mean(), (y.alpha_tavg >= 0.95).mean()),
                     pair(spearmanr(x.alpha_tavg, x.mean_iqr)[0], spearmanr(y.alpha_tavg, y.mean_iqr)[0]),
                     pair(spearmanr(x.alpha_tavg, x.sd_iqr)[0], spearmanr(y.alpha_tavg, y.sd_iqr)[0]),
                     f'{len(x)} / {len(y)}'])
    lines += ['Units alive at T, all cost forms (pilot_d/alpha_scatter.csv against pilot_c/alpha_scatter.csv):', '']
    table(lines, ['family', 'α_tavg mean', 'share α_tavg ≥ 0.95', 'Spearman ρ(α_tavg, mean IQR)',
                  'Spearman ρ(α_tavg, sd IQR)', 'units'], rows)
    block_table(comp, lines, 'equal-split', ['cost_type', 'alpha'], '5b. Equal split (normal)', STD, alphas=A3,
                kstar_cols=lambda r: kstar('normal', r.cost_type, alpha=r.alpha))
    block_table(comp, lines, 'lookback', ['log_family', 'lookback', 'alpha'], '5c. Lookback (power law)',
                STD + [('voluntary_exits_per_period', '{:.2f}')])
    block_table(comp, lines, 'search', ['log_family', 'merge_thresh', 'alpha'], '5d. Search probability p',
                STD, kstar_cols=lambda r: kstar(r.log_family, alpha=r.alpha))
    block_table(comp, lines, 'cost-level', ['cost_type', 'cost_multiplier', 'alpha'],
                '5e. Cost level (laplace, c0 × 0.5 and × 2)', STD, alphas=A3)
    block_table(comp, lines, 'cost-level-full', ['cost_type', 'cost_multiplier', 'alpha'],
                '5f. Cost level, all coefficients (laplace)', STD, alphas=A3)
    block_table(comp, lines, 'cost-level-high', ['log_family', 'cost_multiplier', 'alpha'],
                '5g. Cost level high (power law, c0 × 5, 10, 20)', STD,
                kstar_cols=lambda r: kstar(r.log_family, mult=r.cost_multiplier, alpha=r.alpha))
    block_table(comp, lines, 'floor-level', ['floor_c', 'alpha'], '5h. Floor level (laplace, power law)',
                STD + [('floor_hit_rate_member_status', '{:.4f}'), ('floor_hit_rate_standalone_status', '{:.4f}')],
                alphas=A3)
    block_table(comp, lines, 'correlation', ['alpha'], '5i. Correlation 0.3 (laplace, power law)', STD,
                alphas=[0.05, 0.1, 0.3, 0.5])
    block_table(comp, lines, 'size', ['M', 'N', 'alpha'], '5j. Economy size (laplace, power law; floor c*(N))', STD,
                kstar_cols=lambda r: kstar('laplace', alpha=r.alpha, M=r.M))


def main():
    global KS
    KS = kstar_lookup()
    comp = rd('d', 'comparison_c_vs_d.csv')
    tidy_d = rd('d', 'tidy.csv')
    lines = ['# C37 tables (generated by analytics/tables_c37.py)', '']
    main_block(comp, lines)
    homogeneous(tidy_d, comp, lines)
    acceptance_by_K(lines)
    nu_block(tidy_d, comp, lines)
    prop3(lines)
    other_blocks(comp, lines)
    out = os.path.join(ROOT, 'pilot_d', 'tables_c37.md')
    with open(out, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('wrote', out, len(lines), 'lines')


if __name__ == '__main__':
    main()
