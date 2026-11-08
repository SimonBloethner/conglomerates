#!/usr/bin/env python3
"""
C40: the C33 burn-in check (run_burnin_c33_ext.py) repeated under the moment rule, on pilot D.

Same cells, reps and seeds as C33: Laplace alpha in {0, 0.1, 0.3} and t3 alpha = 0.3, main block, power law,
proportional sharing, five reps each (20 runs), now the pilot D scenarios (decision_rule='moment'), run to
T = 16000. Shocks are drawn step by step, so each run repeats its stored pilot D result up to step 11000;
run_one checks this and stops the task if it does not hold.

The model is called as run_pilot_c.py --pilot d calls it (keyword arguments, decision rule passed through).
analyse_series, rolling_max_deviation, compute_convergence_step and the C33-ext window comparison, trend and
placebo are imported unchanged, so the tables are comparable with diagnostics/burn_in_c33_ext.md.

Decision rule, fixed before these runs (the C33 rule, with the Hill exponent added to the headline series):
- Compare the mean of each headline series over (11000, 16000] with its mean over the pilot window (8000, 11000].
  Headline series: mean K, pooled capital share, floor-hit rate standalone and member, Hill exponent.
- burn_in = 8000 stands if every difference is within 2% in every cell, and the trend of the t3 pooled capital
  share over (8000, 16000] implies a change of less than 2% over those 8000 steps.
- Voluntary exits per period are reported (both window means, the absolute difference and the % difference where
  the pilot-window mean is positive) but not tested: under the moment rule they are about 0 at alpha = 0.3 and a
  few dozen events per window at alpha = 0.1, so a 2% band on them measures Poisson noise.
- If the rule fails in a cell, burn-in is not changed in the code; the report says which series and by how much.

Built-in check: at alpha = 0 no units form under either rule, and the pilot D scenarios are the moment twins of
the pilot C ones (same seeds), so the alpha = 0 series must equal C33-ext's (diagnostics/burn_in_c33_ext.csv).

Usage:
    python run_burnin_c40.py <run_index>   # one run, 0-19 (SLURM array)
    python run_burnin_c40.py --report      # csv, figures and markdown from the saved runs
"""
import csv
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pilot_design_d
from collaborative_growth import model
from pilot_design import BURN_IN, N_REPS, T
from run_burnin_c33 import (CELLS, FINAL_WINDOW, HEADLINE, SERIES, TOLERANCE, WINDOW, Z_THRESHOLD, analyse_series,
                            cell_label, extract_series, fmt_conv)
from run_burnin_c33_ext import (MAX_TREND_CHANGE, MAX_WINDOW_DIFF, N_SURROGATES, T_EXT, placebo,
                                trend_after_burn_in, window_comparison)

RUN_DIR = Path('diagnostics/burn_in_c40_runs')
PILOT_RESULTS = Path('pilot_d/results')
SCENARIOS_JSON = Path('pilot_d/scenarios.json')
C33_EXT_CSV = Path('diagnostics/burn_in_c33_ext.csv')
CSV_PATH = Path('diagnostics/burn_in_c40.csv')
MD_PATH = Path('diagnostics/burn_in_c40.md')

HEADLINE_C40 = HEADLINE | {'hill'}  # tested against the 2% band
REPORTED = HEADLINE_C40 | {'voluntary_exits'}  # rows of the window table


def burnin_scenarios():
    """The 20 pilot D scenarios (moment rule), in CELLS then rep order; each equals its pilot_d/scenarios.json entry."""
    scenarios = pilot_design_d.generate_all_scenarios()
    with open(SCENARIOS_JSON) as f:
        stored = json.load(f)
    selected = []
    for family, alpha in CELLS:
        cell = sorted((s for s in scenarios
                       if s['block'] == 'main' and s['log_family'] == family and s['cost_type'] == 'power_law'
                       and s['alpha'] == alpha and s['decision_rule'] == 'moment'),
                      key=lambda s: s['rep'])
        assert len(cell) == N_REPS, (family, alpha, len(cell))
        for s in cell:
            assert json.loads(json.dumps(s)) == stored[s['scenario_id']], s['scenario_id']
        selected.extend(stored[s['scenario_id']] for s in cell)
    return selected


def run_scenario(scenario):
    """The model() call of run_pilot_c.run_scenario, returning the raw result list."""
    assert not scenario['proportional'], "cost as a fixed amount was removed in C34"
    return model(
        M=scenario['M'],
        N=scenario['N'],
        T=scenario['T'],
        alpha=scenario['alpha'],
        search_prob=scenario['merge_thresh'],
        window=scenario['lookback'],
        cost_form=scenario['cost_type'],
        b0=scenario['c0'],
        b1=scenario['c1'],
        b2=scenario['c2'],
        cost_as_drag=True,
        seed=scenario['seed'],
        growth_process=scenario['growth_process'],
        family=scenario['log_family'],
        nu=scenario.get('nu') or 3.0,
        mu_range=tuple(scenario['mu_range']) if scenario.get('mu_range') else (0.01, 0.1),
        iqr_range=tuple(scenario['sigma_range']),
        floor_c=scenario['floor_c'],
        cross_corr=scenario['cross_corr'],
        metric_every=scenario['metric_every'],
        sharing_rule=scenario['sharing_rule'],
        burn_in=scenario['burn_in'],
        alpha_endogenous=scenario.get('alpha_endogenous', False),
        g=scenario.get('g'),
        market_size_fixed=scenario.get('market_size_fixed', False),
        decision_rule=scenario.get('decision_rule', 'loggain'),
    )


def prefix_matches_pilot(result, scenario, results_dir=PILOT_RESULTS):
    """True if the first T steps of this run reproduce the stored pilot D result (run_burnin_c33_ext's check)."""
    path = Path(results_dir) / f"scenario_{scenario['scenario_id']:04d}.pkl.gz"
    with gzip.open(path, 'rb') as f:
        pilot = pickle.load(f)
    hp = result[-1]
    n_obs = T // scenario['metric_every']
    return (np.array_equal(hp['hill_exponent'][:n_obs], pilot['hill_exponent'], equal_nan=True)
            and np.array_equal(hp['hhi_within'][:, :n_obs], pilot['hhi_within'], equal_nan=True)
            and np.array_equal(hp['cong_capital_share'][:n_obs], pilot['cong_capital_share'], equal_nan=True)
            and np.array_equal(hp['floor_hits_by_status'][:T], pilot['floor_hits_by_status']))


def run_one(idx):
    """Run scenario idx of burnin_scenarios() to T_EXT, check the prefix, save its series."""
    scenario = dict(burnin_scenarios()[idx], T=T_EXT)
    print(f"Run {idx}: pilot D scenario {scenario['scenario_id']}, {scenario['log_family']}, "
          f"alpha={scenario['alpha']}, rep={scenario['rep']}, seed={scenario['seed']}, "
          f"rule={scenario['decision_rule']}, T={T_EXT}", flush=True)

    result = run_scenario(scenario)
    series = extract_series(result, scenario['metric_every'])
    identical = prefix_matches_pilot(result, scenario)
    print(f"First {T} steps identical to pilot D result: {identical}", flush=True)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(RUN_DIR / f'run_{idx:02d}.npz', identical=identical, scenario_id=scenario['scenario_id'], **series)
    if not identical:
        print(f"STOP: run {idx} (scenario {scenario['scenario_id']}) does not reproduce its pilot D result",
              file=sys.stderr, flush=True)
        sys.exit(2)


def load_c33_ext(metric_every):
    """C33-ext series as cells[(family, alpha)][key] -> (reps, n_obs), read back exactly."""
    df = pd.read_csv(C33_EXT_CSV, float_precision='round_trip')
    cells = {}
    for cell in CELLS:
        sub = df[(df.log_family == cell[0]) & (df.alpha == cell[1])].sort_values(['rep', 'step'])
        cells[cell] = {key: sub[key].to_numpy(dtype=float).reshape(N_REPS, T_EXT // metric_every)
                       for key, _, _ in SERIES}
    return cells


def first_entry(runs, metric_every):
    """First step at which the WINDOW-step rolling mean of the rep-mean series is within TOLERANCE of its mean over
    (BURN_IN, T_EXT]: when the series first reaches its long-run level, whatever it does afterwards (the band
    test asks instead from when on it stays there, which a noisy series fails late)."""
    with np.errstate(invalid='ignore'):
        series = np.nanmean(runs, axis=0)
    level = np.nanmean(series[BURN_IN // metric_every:])
    w = WINDOW // metric_every
    for i in range(w - 1, len(series)):
        if abs(np.nanmean(series[i - w + 1:i + 1]) - level) <= TOLERANCE * abs(level):
            return (i + 1) * metric_every
    return None


NOTE = (
    'The moment-rule economy relaxes by 8,000 periods under the C33 criterion: all 17 headline means over '
    '11,000–16,000 are within 0.54% of the pilot window (C33 under loggain: within 0.71% on its 13), and the t3 '
    'pooled-share trend implies +0.16% over 8,000 periods (loggain: −0.4%). Relaxation is not slower than under '
    'loggain. Under both rules mean K and the pooled share reach their long-run level (within 2% of the mean over '
    '8,000–16,000) at step 500–600, the first step the 500-step rolling window can measure, although the moment '
    'economy settles at larger units (K 5.4 against 4.0 at Laplace '
    'α = 0.3) and a higher pooled share (0.91 against 0.78). The C18 band steps differ cell by cell in both '
    'directions: the pooled share stays in the band from step 500 under the moment rule in every cell against '
    '7,100–13,300 under loggain, and mean K from 4,900 against 12,600 at Laplace α = 0.3 but from 8,000 against 500 '
    'at t3 α = 0.3. These steps measure how often a series near its level strays outside a 2% band, which depends '
    'on its noise relative to its level (the pooled share is higher and relatively steadier under the moment rule), '
    'not on how fast it got there; floor-hit rates give 12,700 in most cells under both rules. The Hill exponent '
    'never stays in the band at Laplace α = 0.3 under the moment rule (14,200 under loggain); its window means '
    'differ by −0.28% and its stationary surrogates pass the band test by 8,000 in 1% of draws, so this is its noise, '
    'as in C33.'
)


def window_abs(runs, steps):
    """Window means in absolute terms (for series near zero)."""
    pilot = runs[:, (steps > BURN_IN) & (steps <= T)].mean(axis=1)
    later = runs[:, steps > T].mean(axis=1)
    return pilot.mean(), later.mean()


def report():
    """Write csv, figures and markdown from the saved runs."""
    scenarios = burnin_scenarios()
    metric_every = scenarios[0]['metric_every']
    n_obs = T_EXT // metric_every
    steps = np.arange(1, n_obs + 1) * metric_every
    rng = np.random.default_rng(40)

    loaded = [np.load(RUN_DIR / f'run_{i:02d}.npz') for i in range(len(scenarios))]
    assert [int(run['scenario_id']) for run in loaded] == [s['scenario_id'] for s in scenarios]
    n_identical = sum(bool(run['identical']) for run in loaded)

    cells = {}
    for c, cell in enumerate(CELLS):
        cells[cell] = {key: np.stack([loaded[c * N_REPS + r][key] for r in range(N_REPS)])
                       for key, _, _ in SERIES}
    verdicts = {cell: {key: analyse_series(cells[cell][key], metric_every) for key, _, _ in SERIES}
                for cell in CELLS}

    c33 = load_c33_ext(metric_every)
    verdicts_c33 = {cell: {key: analyse_series(c33[cell][key], metric_every) for key, _, _ in SERIES}
                    for cell in CELLS}

    # alpha = 0: the same economy under both rules
    a0 = ('laplace', 0.0)
    a0_equal = {key: np.array_equal(cells[a0][key], c33[a0][key], equal_nan=True) for key, _, _ in SERIES}
    a0_maxdiff = max(float(np.nanmax(np.abs(cells[a0][key] - c33[a0][key]), initial=0.0)) for key, _, _ in SERIES)

    labels = {key: label for key, label, _ in SERIES}
    rows = []
    for cell in CELLS:
        for key, _, _ in SERIES:
            if key in REPORTED and verdicts[cell][key]['applicable']:
                runs = cells[cell][key]
                pm, lm = window_abs(runs, steps)
                tested = key in HEADLINE_C40
                rows.append({'cell': cell, 'key': key, 'tested': tested, 'pilot_abs': pm, 'later_abs': lm,
                             'window': window_comparison(runs, steps) if pm > 0 else None,
                             'trend': trend_after_burn_in(runs, steps) if tested else None,
                             'placebo': placebo(runs, metric_every, rng) if tested else None})

    tested_rows = [r for r in rows if r['tested']]
    failing = [r for r in tested_rows if abs(r['window']['diff']) > MAX_WINDOW_DIFF]
    t3_ccs_trend = next(r['trend'] for r in rows if r['cell'] == ('t3', 0.3) and r['key'] == 'ccs')
    keep = not failing and abs(t3_ccs_trend['implied_change']) < MAX_TREND_CHANGE
    keep_c33_set = (all(abs(r['window']['diff']) <= MAX_WINDOW_DIFF for r in tested_rows if r['key'] in HEADLINE)
                    and abs(t3_ccs_trend['implied_change']) < MAX_TREND_CHANGE)
    worst = max(tested_rows, key=lambda r: abs(r['window']['diff']))

    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['log_family', 'alpha', 'rep', 'seed', 'scenario_id', 'step'] + [key for key, _, _ in SERIES])
        for c, cell in enumerate(CELLS):
            for r in range(N_REPS):
                s = scenarios[c * N_REPS + r]
                for i, step in enumerate(steps):
                    writer.writerow([cell[0], cell[1], r, s['seed'], s['scenario_id'], step]
                                    + [cells[cell][key][r, i] for key, _, _ in SERIES])
    print(f"Wrote {CSV_PATH}")

    for cell in CELLS:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'C40 burn-in, moment rule: {cell[0]}, power_law, alpha={cell[1]:g}, T={T_EXT} '
                     f'(grey: C33-ext, loggain)', fontsize=12)
        for (key, label, _), ax in zip(SERIES, axes.flat):
            with np.errstate(invalid='ignore'):
                if verdicts_c33[cell][key]['applicable']:
                    ax.plot(steps, np.nanmean(c33[cell][key], axis=0), color='0.6', linewidth=1, label='loggain')
                if verdicts[cell][key]['applicable']:
                    ax.plot(steps, np.nanmean(cells[cell][key], axis=0), 'b-', linewidth=1, label='moment')
            ax.axvline(BURN_IN, color='k', linestyle=':', label=f'burn_in={BURN_IN}')
            ax.axvline(T, color='r', linestyle=':', label=f'pilot T={T}')
            ax.set_xlabel('Step')
            ax.set_title(label)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = f'diagnostics/figures/burnin_c40_{cell_label(*cell)}.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Wrote {fig_path}")

    def cell_name(cell):
        return f"{cell[0]}, α = {cell[1]:g}"

    lines = [
        '# C40: Burn-in under the moment rule (pilot D), T = 16000',
        '',
        'The C33-ext check (`burn_in_c33_ext.md`, loggain, pilot C) repeated on the pilot D economy: the same 20 '
        'scenarios (Laplace α ∈ {0, 0.1, 0.3} and t3 α = 0.3, main block, power law, proportional sharing, five reps, '
        f'the pilot D seeds), `decision_rule = \'moment\'`, run to T = {T_EXT}. Script `run_burnin_c40.py`.',
        '',
        f'{n_identical} of {len(scenarios)} runs reproduce their stored pilot D result '
        f'(`pilot_d/results/scenario_XXXX.pkl.gz`) up to step {T} exactly (Hill, within-market HHI, pooled capital '
        'share, floor hits by status).',
        '',
        f"α = 0 check: no units form under either rule and the pilot D scenarios are the moment twins of the pilot C "
        f"ones (same seeds), so the α = 0 series must equal C33-ext's. "
        f"{'All ' + str(len(SERIES)) + ' series are identical' if all(a0_equal.values()) else 'NOT identical: ' + ', '.join(k for k, v in a0_equal.items() if not v)}"
        f" (largest absolute difference {a0_maxdiff:.3g}).",
        '',
        '## Decision rule (fixed before the runs)',
        '',
        f'burn_in = {BURN_IN} stands if the mean of each headline series over ({T}, {T_EXT}] is within '
        f'{MAX_WINDOW_DIFF:g}% of its mean over the pilot window ({BURN_IN}, {T}] in every cell, and the trend of the '
        f't3 pooled capital share over ({BURN_IN}, {T_EXT}] implies a change of less than {MAX_TREND_CHANGE:g}% over '
        f'those {T_EXT - BURN_IN} steps. Headline series: mean K, pooled capital share, floor-hit rate standalone and '
        'member (the C33 set) and the Hill exponent. Voluntary exits per period are reported, not tested: they are '
        'about 0 at α = 0.3 and a few dozen events per window at α = 0.1, so a 2% band on them measures Poisson noise.',
        '',
        f"**Result: burn_in = {BURN_IN} {'stands' if keep else 'DOES NOT stand'}.** "
        f"{len(tested_rows)} headline means, largest difference {worst['window']['diff']:+.2f}% "
        f"({labels[worst['key']]}, {cell_name(worst['cell'])}). "
        f"t3 pooled capital-share trend: {t3_ccs_trend['slope']:+.3f}% per 1000 steps (p = {t3_ccs_trend['p']:.3f}), "
        f"implied change {t3_ccs_trend['implied_change']:+.2f}%. "
        f"On the C33 headline set alone (without Hill) the rule {'also holds' if keep_c33_set else 'fails'}.",
        '',
        '## Reading',
        '',
        NOTE,
        '',
    ]
    if failing:
        lines += ['Failing series (burn-in not changed in the code, per the brief):', '']
        for r in failing:
            lines.append(f"- {cell_name(r['cell'])}, {labels[r['key']]}: {r['window']['diff']:+.2f}% "
                         f"(95% CI [{r['window']['ci_low']:+.2f}, {r['window']['ci_high']:+.2f}])")
        lines.append('')

    lines += [
        '## Headline series',
        '',
        f'Difference: ({T}, {T_EXT}] against ({BURN_IN}, {T}], % of the pilot-window mean; 95% CI of the difference '
        f'across {N_REPS} reps, in % points. Trend: OLS over ({BURN_IN}, {T_EXT}], rep by rep. Placebo: '
        f'{N_SURROGATES} stationary surrogates block-resampled from each rep\'s post-{BURN_IN} window. Same functions '
        'as `burn_in_c33_ext.md`.',
        '',
        '| Cell | Series | Pilot window | Periods 11,000–16,000 | Difference, % of pilot mean | 95% CI of the difference, % points | Trend %/1000 | p | Implied change % | Surrogates passing | P(dev ≥ observed) |',
        '|------|--------|--------------|-----------------------|-----------------------------|------------------------------------|--------------|---|------------------|--------------------|-------------------|',
    ]
    for r in tested_rows:
        w, tr, pl = r['window'], r['trend'], r['placebo']
        lines.append(f"| {cell_name(r['cell'])} | {labels[r['key']]} | {w['pilot_mean']:.4f} | "
                     f"{w['later_mean']:.4f} | {w['diff']:+.2f} | [{w['ci_low']:+.2f}, {w['ci_high']:+.2f}] | "
                     f"{tr['slope']:+.3f} | {tr['p']:.3f} | {tr['implied_change']:+.2f} | "
                     f"{pl['pass_share']:.2f} | {pl['exceed_share']:.2f} |")

    lines += [
        '',
        '## Voluntary exits per period (reported, not tested)',
        '',
        '| Cell | Pilot window | Periods 11,000–16,000 | Difference per period | Difference, % of pilot mean | 95% CI, % points |',
        '|------|--------------|-----------------------|-----------------------|-----------------------------|------------------|',
    ]
    for cell in CELLS:
        r = next((r for r in rows if r['cell'] == cell and r['key'] == 'voluntary_exits'), None)
        if r is None:
            lines.append(f'| {cell_name(cell)} | 0 | 0 | 0 | n/a | n/a |')
            continue
        w = r['window']
        pct = 'n/a' if w is None else f"{w['diff']:+.2f}"
        ci = 'n/a' if w is None else f"[{w['ci_low']:+.2f}, {w['ci_high']:+.2f}]"
        lines.append(f"| {cell_name(cell)} | {r['pilot_abs']:.5f} | {r['later_abs']:.5f} | "
                     f"{r['later_abs'] - r['pilot_abs']:+.5f} | {pct} | {ci} |")

    lines += [
        '',
        '## Convergence steps, moment against loggain',
        '',
        f'C18 criterion on T = {T_EXT}, both rules: first step from which the {WINDOW}-step rolling mean of the '
        f'rep-mean series stays within {TOLERANCE:.0%} of its final-{FINAL_WINDOW}-step mean. Loggain: the C33-ext '
        'series (`burn_in_c33_ext.csv`), run through the same `analyse_series`. "never" means the series leaves the '
        'band again after every entry; as in C33-ext, the band test depends on run length and on how noisy the series '
        'is relative to 2%, so it is reported for completeness, and the window comparison above is the evidence.',
        '',
        'First entry: the first step at which the same rolling mean comes within 2% of the series\' mean over '
        f'({BURN_IN}, {T_EXT}], i.e. when the series first reaches its long-run level (500 is the first step the '
        'rolling window can measure).',
        '',
        '| Cell | Series | Band step, moment | Band step, loggain | First entry, moment | First entry, loggain | Moment final mean | Loggain final mean |',
        '|------|--------|-------------------|--------------------|---------------------|----------------------|-------------------|--------------------|',
    ]

    def entry(runs, verdict):
        return 'n/a' if not verdict['applicable'] else (lambda s: 'never' if s is None else str(s))(
            first_entry(runs, metric_every))

    for cell in CELLS:
        for key, label, fmt in SERIES:
            vm, vl = verdicts[cell][key], verdicts_c33[cell][key]
            if not (vm['applicable'] or vl['applicable']):
                continue
            lines.append(f"| {cell_name(cell)} | {label} | {fmt_conv(vm) if vm['applicable'] else 'n/a'} | "
                         f"{fmt_conv(vl) if vl['applicable'] else 'n/a'} | "
                         f"{entry(cells[cell][key], vm)} | {entry(c33[cell][key], vl)} | "
                         f"{format(vm['final_mean'], fmt) if vm['applicable'] else 'n/a'} | "
                         f"{format(vl['final_mean'], fmt) if vl['applicable'] else 'n/a'} |")

    lines += [
        '',
        '## Convergence results (moment rule)',
        '',
        f'C18 criterion on T = {T_EXT}: {WINDOW}-step rolling mean within {TOLERANCE:.0%} of the final-{FINAL_WINDOW}-step '
        f'mean (steps {T_EXT - FINAL_WINDOW + metric_every}-{T_EXT}). Geweke on steps {BURN_IN + metric_every}-{T_EXT} '
        f'(80 observations): first 8 against last 40, pass if |z| < {Z_THRESHOLD}.',
        '',
        'The band test depends on run length: a longer run gives a series more chances to leave the band, so '
        'series with no trend fail it later here than at T = 11000. The window comparison above is the evidence '
        'for the burn-in; this table is reported for completeness.',
        '',
    ]
    for cell in CELLS:
        lines += [
            f'### {cell[0]}, alpha={cell[1]:g}',
            '',
            f'| Metric | Convergence Step | Final Mean | Before {BURN_IN} | Max dev. | Geweke z | Geweke | Rep passes |',
            '|--------|-----------------|------------|-------------|----------|----------|--------|------------|',
        ]
        for key, label, fmt in SERIES:
            verdict = verdicts[cell][key]
            if not verdict['applicable']:
                lines.append(f'| {label} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |')
                continue
            lines.append(
                f"| {label} | {fmt_conv(verdict)} | {verdict['final_mean']:{fmt}} | "
                f"{'yes' if verdict['converged'] else 'NO'} | {verdict['max_dev']:.1%} | "
                f"{verdict['z_mean']:.2f} | {'pass' if verdict['geweke_pass'] else 'FAIL'} | "
                f"{verdict['rep_pass']}/{N_REPS} |")
        lines.append('')

    lines += [f'## Metric tables ({T + 1000}-{T_EXT}, every 1000 steps)', '', 'Means across reps at each step.', '']
    for cell in CELLS:
        lines += [
            f'### {cell[0]}, alpha={cell[1]:g}',
            '',
            '| Step | Hill | Mean K | K_eff/K | Cong Share | FHR Stand | FHR Memb | Mergers | Vol Exits | Floor Exits | HHI |',
            '|------|------|--------|---------|------------|-----------|----------|---------|-----------|-------------|-----|',
        ]
        for step in range(T + 1000, T_EXT + 1, 1000):
            idx = step // metric_every - 1
            row = [f'{step}']
            for key, _, fmt in SERIES:
                column = cells[cell][key][:, idx]
                row.append('n/a' if np.all(np.isnan(column)) else f'{np.nanmean(column):{fmt}}')
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('')

    lines += ['## Figures', '']
    for cell in CELLS:
        lines.append(f'- `figures/burnin_c40_{cell_label(*cell)}.png`: all series for {cell[0]}, α = {cell[1]:g}, '
                     'moment (blue) and loggain C33-ext (grey)')
    lines.append('')

    with open(MD_PATH, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Wrote {MD_PATH}')
    print(f'Identical to pilot D up to T={T}: {n_identical}/{len(scenarios)}')
    print(f"alpha = 0 identical to C33-ext: {all(a0_equal.values())} (max abs diff {a0_maxdiff:.3g})")
    print(f"burn_in = {BURN_IN} {'stands' if keep else 'DOES NOT stand'}; worst {worst['window']['diff']:+.2f}% "
          f"({worst['key']}, {worst['cell']}); t3 ccs implied change {t3_ccs_trend['implied_change']:+.2f}%")


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_burnin_c40.py <run_index> | --report")
        sys.exit(1)

    if sys.argv[1] == '--report':
        report()
    else:
        run_one(int(sys.argv[1]))


if __name__ == '__main__':
    main()
