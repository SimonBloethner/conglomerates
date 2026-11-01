#!/usr/bin/env python3
"""
C33: Burn-in on the frozen model (02ce733).

Replaces the C18 diagnostic (run_burnin_c14.py), which ran g=0.02, lookback=100, T=8000.
The runs here are the pilot runs: each scenario is taken from pilot_design.generate_all_scenarios()
and passed to model() as run_pilot_c.py passes it, so T=11000 and exit_review_every stays at its default.

Cells (main block, power_law, 5 reps, 20 runs):
- laplace at alpha 0, 0.1, 0.3
- t3 at alpha 0.3

Series, every metric_every steps:
- pooled Hill, mean K, K_eff/K, conglomerate capital share, within-market HHI (snapshots)
- floor-hit rate standalone and member: hits / all firm-periods, as summary['floor_hit_rate_*']
- mergers, voluntary exits, floor exits per period: mean over the metric_every steps

Convergence (run_burnin_c14.compute_convergence_step): 500-step rolling mean within 2% of the
final-2000-step mean, on the mean across reps. Geweke z (burnin_check.geweke_test) on the post-burn-in window.

Usage:
    python run_burnin_c33.py <run_index>   # one run, 0-19 (SLURM array)
    python run_burnin_c33.py --report      # csv, figures and markdown from the saved runs
"""
import csv
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from burnin_check import geweke_test
from collaborative_growth import model
from pilot_design import BURN_IN, N_REPS, T, generate_all_scenarios
from run_burnin_c14 import compute_convergence_step, extract_floor_hit_rate_by_status

RUN_DIR = Path('diagnostics/burn_in_c33_runs')

# (log_family, alpha) cells of the main block with power_law cost
CELLS = [('laplace', 0.0), ('laplace', 0.1), ('laplace', 0.3), ('t3', 0.3)]

# Key, label, table format; HEADLINE series stop the pilot if they have not converged by burn-in
SERIES = [
    ('hill', 'Pooled Hill exponent', '.4f'),
    ('mean_k', 'Mean K', '.2f'),
    ('k_eff_ratio', 'K_eff/K', '.4f'),
    ('ccs', 'Cong capital share', '.4f'),
    ('fhr_standalone', 'Floor-hit rate (standalone)', '.4f'),
    ('fhr_member', 'Floor-hit rate (member)', '.4f'),
    ('mergers', 'Mergers/period', '.4f'),
    ('voluntary_exits', 'Voluntary exits/period', '.4f'),
    ('floor_exits', 'Floor exits/period', '.4f'),
    ('hhi_within', 'Within-market HHI', '.4f'),
]
HEADLINE = {'mean_k', 'ccs', 'fhr_standalone', 'fhr_member'}

WINDOW = 500
TOLERANCE = 0.02
FINAL_WINDOW = 2000
Z_THRESHOLD = 1.96


def burnin_scenarios():
    """The 20 pilot scenarios, in CELLS then rep order."""
    scenarios = generate_all_scenarios()
    selected = []
    for family, alpha in CELLS:
        selected.extend(sorted(
            (s for s in scenarios
             if s['block'] == 'main' and s['log_family'] == family
             and s['cost_type'] == 'power_law' and s['alpha'] == alpha),
            key=lambda s: s['rep']))
    return selected


def run_scenario(scenario):
    """Run one pilot scenario; same model() call as run_pilot_c.run_scenario."""
    total_firms = scenario['M'] * scenario['N']
    params = [
        scenario['M'],
        scenario['N'],
        scenario['T'],
        scenario['alpha'],
        total_firms,
        scenario['merge_thresh'],
        4,  # K_max default
        0.0,  # minimum_benefit default
        scenario['proportional'],
        scenario['lookback'],
        scenario['cost_type'],
        scenario['c0'],
        scenario['c1'],
        scenario['c2'],
    ]

    return model(
        params,
        seed=scenario['seed'],
        growth_process=scenario['growth_process'],
        log_family=scenario['log_family'],
        mu_range=tuple(scenario['mu_range']),
        sigma_range=tuple(scenario['sigma_range']),
        floor_c=scenario['floor_c'],
        cross_corr=scenario['cross_corr'],
        metric_every=scenario['metric_every'],
        sharing_rule=scenario['sharing_rule'],
        burn_in=scenario['burn_in'],
        alpha_endogenous=scenario['alpha_endogenous'],
        g=scenario['g'],
        market_size_fixed=scenario['market_size_fixed'],
        decision_rule=scenario['decision_rule'],
    )


def block_means(per_step, metric_every):
    """Mean of a per-step series over consecutive metric_every blocks."""
    n_obs = len(per_step) // metric_every
    return np.asarray(per_step[:n_obs * metric_every], dtype=float).reshape(n_obs, metric_every).mean(axis=1)


def extract_series(result, metric_every):
    """Tracked series of one run, each of shape (n_obs,). K series are NaN at snapshots without units."""
    hp = result[-1]
    mergers_per_period = result[7]
    exits_per_period = result[9]
    n_obs = hp['steps'] // metric_every

    mean_k = np.full(n_obs, np.nan)
    k_eff_ratio = np.full(n_obs, np.nan)
    for step, congs in hp['effective_members']:
        if congs:
            idx = step // metric_every
            mean_k[idx] = np.mean([K for (_, K, _) in congs])
            k_eff_ratio[idx] = np.mean([K_eff / K for (_, K, K_eff) in congs])

    fhr_standalone, fhr_member = extract_floor_hit_rate_by_status(hp, metric_every)

    return {
        'hill': np.asarray(hp['hill_exponent'], dtype=float),
        'mean_k': mean_k,
        'k_eff_ratio': k_eff_ratio,
        'ccs': np.asarray(hp['cong_capital_share'], dtype=float),
        'fhr_standalone': fhr_standalone,
        'fhr_member': fhr_member,
        'mergers': block_means(mergers_per_period, metric_every),
        'voluntary_exits': block_means(exits_per_period - hp['floor_exits_per_period'], metric_every),
        'floor_exits': block_means(hp['floor_exits_per_period'], metric_every),
        'hhi_within': np.nanmean(hp['hhi_within'], axis=0),
    }


def matches_pilot(result, scenario):
    """True if this run reproduces the stored pilot result, snapshot by snapshot."""
    path = Path('pilot_c/results') / f"scenario_{scenario['scenario_id']:04d}.pkl.gz"
    with gzip.open(path, 'rb') as f:
        pilot = pickle.load(f)
    hp = result[-1]
    arrays_equal = all(
        np.array_equal(hp[key], pilot[key], equal_nan=True)
        for key in ['hill_exponent', 'hhi_within', 'cong_capital_share', 'floor_hits_by_status'])
    scalars_equal = all(
        np.array_equal(hp['summary'][key], pilot['summary'][key], equal_nan=True)
        for key in ['K_mean', 'mergers_per_period', 'exits_per_period', 'floor_exits_per_period'])
    return arrays_equal and scalars_equal


def run_one(idx):
    """Run scenario idx of burnin_scenarios() and save its series."""
    scenario = burnin_scenarios()[idx]
    print(f"Run {idx}: scenario {scenario['scenario_id']}, {scenario['log_family']}, "
          f"alpha={scenario['alpha']}, rep={scenario['rep']}, seed={scenario['seed']}", flush=True)

    result = run_scenario(scenario)
    series = extract_series(result, scenario['metric_every'])
    identical = matches_pilot(result, scenario)
    print(f"Identical to pilot result: {identical}", flush=True)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(RUN_DIR / f'run_{idx:02d}.npz', identical=identical, **series)


def rolling_max_deviation(series, from_step, metric_every):
    """Largest |rolling mean - final mean| / |final mean| at steps >= from_step."""
    window = WINDOW // metric_every
    final_mean = np.nanmean(series[-FINAL_WINDOW // metric_every:])
    worst = 0.0
    for i in range(from_step // metric_every - 1, len(series)):
        worst = max(worst, abs(np.nanmean(series[i - window + 1:i + 1]) - final_mean) / abs(final_mean))
    return worst


def analyse_series(runs, metric_every):
    """
    Verdict for one series in one cell; runs has shape (reps, n_obs).

    conv_step is None if the rolling mean never stays within tolerance (compute_convergence_step
    then returns the series length). applicable is False if the series is absent or identically
    zero in the cell, e.g. K and member floor hits at alpha = 0.
    """
    burn_idx = BURN_IN // metric_every
    if np.all(np.isnan(runs)) or np.nansum(np.abs(runs)) == 0:
        return {'applicable': False}

    with np.errstate(invalid='ignore'):
        mean_series = np.nanmean(runs, axis=0)
    conv_step = compute_convergence_step(mean_series, window=WINDOW, tolerance=TOLERANCE,
                                         final_window=FINAL_WINDOW, metric_every=metric_every)
    if conv_step >= len(mean_series) * metric_every:
        conv_step = None

    z_mean = geweke_test(mean_series[burn_idx:], z_threshold=Z_THRESHOLD)[0]
    rep_z = [geweke_test(run[burn_idx:], z_threshold=Z_THRESHOLD)[0] for run in runs]

    return {
        'applicable': True,
        'conv_step': conv_step,
        'converged': conv_step is not None and conv_step <= BURN_IN,
        'final_mean': np.nanmean(mean_series[-FINAL_WINDOW // metric_every:]),
        'max_dev': rolling_max_deviation(mean_series, BURN_IN, metric_every),
        'z_mean': z_mean,
        'geweke_pass': abs(z_mean) < Z_THRESHOLD,
        'rep_pass': sum(abs(z) < Z_THRESHOLD for z in rep_z),
    }


def cell_label(family, alpha):
    return f'{family}_a{alpha:g}'


def fmt_conv(verdict):
    return 'never' if verdict['conv_step'] is None else str(verdict['conv_step'])


def report():
    """Write csv, figures and markdown from the saved runs."""
    scenarios = burnin_scenarios()
    metric_every = scenarios[0]['metric_every']
    n_obs = T // metric_every
    steps_axis = np.arange(1, n_obs + 1) * metric_every

    loaded = [np.load(RUN_DIR / f'run_{i:02d}.npz') for i in range(len(scenarios))]
    n_identical = sum(bool(run['identical']) for run in loaded)

    # cells[(family, alpha)][key] has shape (reps, n_obs)
    cells = {}
    for c, cell in enumerate(CELLS):
        cells[cell] = {key: np.stack([loaded[c * N_REPS + r][key] for r in range(N_REPS)])
                       for key, _, _ in SERIES}
    verdicts = {cell: {key: analyse_series(cells[cell][key], metric_every) for key, _, _ in SERIES}
                for cell in CELLS}

    # Raw series, one row per (cell, rep, step)
    csv_path = 'diagnostics/burn_in_c33.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['log_family', 'alpha', 'rep', 'seed', 'step'] + [key for key, _, _ in SERIES])
        for c, cell in enumerate(CELLS):
            for r in range(N_REPS):
                seed = scenarios[c * N_REPS + r]['seed']
                for i, step in enumerate(steps_axis):
                    writer.writerow([cell[0], cell[1], r, seed, step]
                                    + [cells[cell][key][r, i] for key, _, _ in SERIES])
    print(f"Wrote {csv_path}")

    # One figure per cell
    for cell in CELLS:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'C33 burn-in: {cell[0]}, power_law, alpha={cell[1]:g}', fontsize=12)
        for (key, label, _), ax in zip(SERIES, axes.flat):
            verdict = verdicts[cell][key]
            if verdict['applicable']:
                with np.errstate(invalid='ignore'):
                    ax.plot(steps_axis, np.nanmean(cells[cell][key], axis=0), 'b-', linewidth=1)
                ax.axhspan(verdict['final_mean'] * (1 - TOLERANCE), verdict['final_mean'] * (1 + TOLERANCE),
                           color='g', alpha=0.15, label='final mean ±2%')
                if verdict['conv_step'] is not None:
                    ax.axvline(verdict['conv_step'], color='r', linestyle='--',
                               label=f"Conv @ {verdict['conv_step']}")
            ax.axvline(BURN_IN, color='k', linestyle=':', label=f'burn_in={BURN_IN}')
            ax.set_xlabel('Step')
            ax.set_title(label)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = f'diagnostics/figures/burnin_c33_{cell_label(*cell)}.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Wrote {fig_path}")

    failed_headline = [(cell, key) for cell in CELLS for key, _, _ in SERIES
                       if key in HEADLINE and verdicts[cell][key]['applicable']
                       and not verdicts[cell][key]['converged']]

    s0 = scenarios[0]
    lines = [
        '# C33: Burn-in on the frozen model',
        '',
        'Supersedes `burn_in_c.md` (C18). Model: commit `02ce733`.',
        '',
        'The stop below is resolved by the extension to T = 16000: see `burn_in_c33_ext.md`.',
        '',
        '## Setup',
        '',
        '**Parameters** (from `pilot_design.make_base_params()` and the main block):',
        f"- M = N = {s0['M']}",
        f"- g = {s0['g']}, lookback = {s0['lookback']}, merge_thresh = {s0['merge_thresh']}",
        f"- floor_c = {s0['floor_c']}, sigma_range = {tuple(s0['sigma_range'])}",
        f"- decision_rule = {s0['decision_rule']}, market_size_fixed = {s0['market_size_fixed']}, "
        f"sharing_rule = {s0['sharing_rule']}",
        f"- power_law cost: c0 = {s0['c0']}, c1 = {s0['c1']}",
        '- exit_review_every at its default (1), as in every pilot run',
        f"- T = {T}, metric_every = {metric_every}, {N_REPS} replications with the pilot seeds",
        '',
        f'{n_identical} of {len(scenarios)} runs reproduce the stored pilot result exactly '
        '(Hill, within-market HHI, capital share and floor hits at every snapshot; '
        'K_mean and merger and exit rates in the summary).',
        '',
        '| Cell | Scenario ids | Seeds |',
        '|------|--------------|-------|',
    ]
    for c, cell in enumerate(CELLS):
        group = scenarios[c * N_REPS:(c + 1) * N_REPS]
        lines.append(f"| {cell[0]}, α = {cell[1]:g} | {group[0]['scenario_id']}-{group[-1]['scenario_id']} | "
                     f"{group[0]['seed']}-{group[-1]['seed']} |")

    lines += [
        '',
        '## Definitions',
        '',
        f'- **Convergence**: first step from which the {WINDOW}-step rolling mean stays within '
        f'{TOLERANCE:.0%} of the final-{FINAL_WINDOW}-step mean to the end of the run, on the mean across reps '
        '(C18 criterion). "never" means no such step exists. '
        f'"Max dev." is the largest deviation of the rolling mean from the final mean at steps ≥ {BURN_IN}.',
        f'- **Geweke**: `burnin_check.geweke_test` on steps {BURN_IN + metric_every}-{T} '
        f'({(T - BURN_IN) // metric_every} observations): first 10% (3 observations) against last 50% '
        f'(15 observations), i.i.d. variance, pass if |z| < {Z_THRESHOLD}. "Rep" counts passes across {N_REPS} reps. '
        'The variance ignores autocorrelation, so the test over-rejects on persistent series, '
        f'and with {sum(v["applicable"] for c in verdicts.values() for v in c.values())} '
        'cell-level tests about 2 failures are expected at the 5% level under stationarity.',
        '- **Floor-hit rates**: hits by status divided by all firm-periods, as '
        "`summary['floor_hit_rate_*']` and `tidy.csv`; the two rates add up to the total rate.",
        '- **n/a**: the series is absent or identically zero in the cell.',
        '',
        '## Verdict',
        '',
    ]
    if failed_headline:
        lines.append('**STOP: headline series not converged by the burn-in of '
                     f'{BURN_IN}** (K, capital share, floor-hit rates):')
        lines.append('')
        for cell, key in failed_headline:
            verdict = verdicts[cell][key]
            label = next(lab for k, lab, _ in SERIES if k == key)
            lines.append(f"- {cell[0]}, α = {cell[1]:g}, {label}: convergence step {fmt_conv(verdict)}, "
                         f"max dev. {verdict['max_dev']:.1%}")
    else:
        lines.append(f'All headline series (K, capital share, floor-hit rates) converge by step {BURN_IN}.')
    lines.append('')

    others = [(cell, key) for cell in CELLS for key, _, _ in SERIES
              if key not in HEADLINE and verdicts[cell][key]['applicable']
              and not verdicts[cell][key]['converged']]
    if others:
        lines.append(f'Other series not converged by step {BURN_IN} under the 2% criterion:')
        lines.append('')
        for cell, key in others:
            verdict = verdicts[cell][key]
            label = next(lab for k, lab, _ in SERIES if k == key)
            lines.append(f"- {cell[0]}, α = {cell[1]:g}, {label}: convergence step {fmt_conv(verdict)}, "
                         f"max dev. {verdict['max_dev']:.1%}")
        lines.append('')

    failed_geweke = [(cell, key) for cell in CELLS for key, _, _ in SERIES
                     if verdicts[cell][key]['applicable'] and not verdicts[cell][key]['geweke_pass']]
    if failed_geweke:
        lines.append(f'Post-{BURN_IN} Geweke failures on the cell mean:')
        lines.append('')
        for cell, key in failed_geweke:
            verdict = verdicts[cell][key]
            label = next(lab for k, lab, _ in SERIES if k == key)
            lines.append(f"- {cell[0]}, α = {cell[1]:g}, {label}: z = {verdict['z_mean']:.2f}, "
                         f"rep passes {verdict['rep_pass']}/{N_REPS}")
        lines.append('')
    else:
        lines.append(f'All post-{BURN_IN} Geweke tests on the cell mean pass.')
        lines.append('')

    lines += ['## Convergence Results', '']
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

    lines += ['## Metric Tables (every 1000 steps)', '', 'Means across reps at each step.', '']
    for cell in CELLS:
        lines += [
            f'### {cell[0]}, alpha={cell[1]:g}',
            '',
            '| Step | Hill | Mean K | K_eff/K | Cong Share | FHR Stand | FHR Memb | Mergers | Vol Exits | Floor Exits | HHI |',
            '|------|------|--------|---------|------------|-----------|----------|---------|-----------|-------------|-----|',
        ]
        for step in range(1000, T + 1, 1000):
            idx = step // metric_every - 1
            row = [f'{step}']
            for key, _, fmt in SERIES:
                column = cells[cell][key][:, idx]
                row.append('n/a' if np.all(np.isnan(column)) else f'{np.nanmean(column):{fmt}}')
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('')

    lines += ['## Figures', '']
    for cell in CELLS:
        lines.append(f'- `figures/burnin_c33_{cell_label(*cell)}.png`: all series for {cell[0]}, α = {cell[1]:g}')
    lines.append('')

    with open('diagnostics/burn_in_c33.md', 'w') as f:
        f.write('\n'.join(lines))
    print('Wrote diagnostics/burn_in_c33.md')
    print(f'Identical to pilot: {n_identical}/{len(scenarios)}')
    print('STOP: headline series not converged' if failed_headline else 'Headline series converged')


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_burnin_c33.py <run_index> | --report")
        sys.exit(1)

    if sys.argv[1] == '--report':
        report()
    else:
        run_one(int(sys.argv[1]))


if __name__ == '__main__':
    main()
