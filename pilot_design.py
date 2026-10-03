#!/usr/bin/env python3
"""
Pilot design generator (§7).

Generates factorial design for Phase B experiments:
- Treatment factors: growth_process, pooling_rule, rho, cross_corr
- α values: [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
- Replications: configurable (default 5 per scenario)

Full factorial: 2 × 2 × 3 × 3 = 36 scenarios
With α values: 36 × 12 = 432 cells
With replications: 432 × 5 = 2160 experiments

Usage:
    # Generate design table
    python pilot_design.py --output pilot_scenarios.csv

    # Generate SLURM scripts
    python pilot_design.py --slurm --output pilot_jobs/

    # Estimate runtime
    python pilot_design.py --estimate
"""
import argparse
import itertools
import os
import numpy as np
import pandas as pd


# Treatment factors and their levels
FACTORS = {
    'growth_process': ['normal_net', 'lognormal'],
    'pooling_rule': ['ewp', 'cap'],
    'rho': ['uncorr', 'pos', 'neg'],
    'cross_corr': ['none', 'block', 'ar1'],
}

# α values from updated grid
ALPHA_VALUES = np.array([0.00, 0.02, 0.05] + list(np.arange(0.10, 0.52, 0.05)))

# Cost function levels (for extended design)
COST_LEVELS = {
    'power_law': {'c0': None, 'c1': None, 'c2': None},  # Use defaults
}


def generate_factorial_design(include_cost=False):
    """
    Generate full factorial design.

    Parameters:
    -----------
    include_cost : bool
        If True, include cost function variations (expands design)

    Returns:
    --------
    scenarios : list of dict
        Each dict contains factor levels for one scenario
    """
    # Get all factor levels
    factor_names = list(FACTORS.keys())
    factor_levels = [FACTORS[f] for f in factor_names]

    # Generate all combinations
    scenarios = []
    for i, combo in enumerate(itertools.product(*factor_levels)):
        scenario = {
            'scenario_id': i,
            'scenario_name': '_'.join(combo),
        }
        for name, level in zip(factor_names, combo):
            scenario[name] = level

        # Add cost function (default)
        scenario['cost_type'] = 'power_law'

        scenarios.append(scenario)

    return scenarios


def generate_experiment_table(scenarios, alpha_values=ALPHA_VALUES, n_reps=5):
    """
    Generate full experiment table (scenarios × α × replications).

    Parameters:
    -----------
    scenarios : list of dict
        Scenario definitions from generate_factorial_design
    alpha_values : array-like
        α values to test
    n_reps : int
        Number of replications per cell

    Returns:
    --------
    df : pd.DataFrame
        Full experiment table
    """
    rows = []
    exp_id = 0

    for scenario in scenarios:
        for alpha in alpha_values:
            for rep in range(n_reps):
                row = scenario.copy()
                row['alpha'] = alpha
                row['replication'] = rep
                row['experiment_id'] = exp_id
                rows.append(row)
                exp_id += 1

    return pd.DataFrame(rows)


def estimate_runtime(n_scenarios, n_alpha, n_reps, steps=200, seconds_per_step=0.1):
    """
    Estimate total runtime for pilot.

    Parameters:
    -----------
    n_scenarios : int
        Number of unique scenarios
    n_alpha : int
        Number of α values
    n_reps : int
        Replications per cell
    steps : int
        Simulation steps per experiment
    seconds_per_step : float
        Estimated time per simulation step

    Returns:
    --------
    estimate : dict
        Runtime estimates
    """
    n_experiments = n_scenarios * n_alpha * n_reps
    time_per_exp = steps * seconds_per_step  # seconds
    total_serial_time = n_experiments * time_per_exp

    return {
        'n_scenarios': n_scenarios,
        'n_alpha': n_alpha,
        'n_reps': n_reps,
        'n_experiments': n_experiments,
        'time_per_exp_sec': time_per_exp,
        'total_serial_hours': total_serial_time / 3600,
        'parallel_64cores_hours': total_serial_time / 3600 / 64,
        'parallel_256cores_hours': total_serial_time / 3600 / 256,
    }


def generate_slurm_scripts(scenarios, output_dir, n_reps=5, cores_per_job=64):
    """
    Generate SLURM job scripts for each scenario.

    Parameters:
    -----------
    scenarios : list of dict
        Scenario definitions
    output_dir : str
        Directory to write scripts
    n_reps : int
        Replications per scenario
    cores_per_job : int
        CPU cores per SLURM job
    """
    os.makedirs(output_dir, exist_ok=True)

    # Master submit script
    submit_lines = ['#!/bin/bash', '# Master script to submit all pilot jobs', '']

    for scenario in scenarios:
        job_name = f"pilot_{scenario['scenario_id']:03d}_{scenario['scenario_name']}"

        script = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cores_per_job}
#SBATCH --time=04:00:00
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err

# Load modules
module load python/3.13.3

# Activate environment
source $HOME/cong_env/bin/activate

# Change to working directory
cd /groups/m-larch/bt307958/conglomerates_dev

# Run experiment
python parallel_counterfactuals.py \\
    --scenario_name {scenario['scenario_name']} \\
    --growth_process {scenario['growth_process']} \\
    --pooling_rule {scenario['pooling_rule']} \\
    --rho {scenario['rho']} \\
    --cross_corr {scenario['cross_corr']} \\
    --counterfactuals {n_reps} \\
    --n_cores {cores_per_job} \\
    --output_dir results/pilot/{scenario['scenario_name']}

echo "Job completed: {job_name}"
'''

        script_path = os.path.join(output_dir, f'{job_name}.sh')
        with open(script_path, 'w') as f:
            f.write(script)

        submit_lines.append(f'sbatch {job_name}.sh')

    # Write master submit script
    submit_path = os.path.join(output_dir, 'submit_all.sh')
    with open(submit_path, 'w') as f:
        f.write('\n'.join(submit_lines))

    print(f"Generated {len(scenarios)} SLURM scripts in {output_dir}")
    print(f"Master submit script: {submit_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate pilot design for Phase B experiments'
    )
    parser.add_argument('--output', type=str, default='pilot_scenarios.csv',
                       help='Output path for scenario table or SLURM directory')
    parser.add_argument('--n_reps', type=int, default=5,
                       help='Replications per cell (default: 5)')
    parser.add_argument('--slurm', action='store_true',
                       help='Generate SLURM job scripts instead of CSV')
    parser.add_argument('--estimate', action='store_true',
                       help='Only print runtime estimates')
    parser.add_argument('--full_table', action='store_true',
                       help='Output full experiment table (scenarios × α × reps)')

    args = parser.parse_args()

    # Generate factorial design
    scenarios = generate_factorial_design()
    n_scenarios = len(scenarios)
    n_alpha = len(ALPHA_VALUES)

    print(f"Factorial Design: {' × '.join(f'{len(FACTORS[f])}' for f in FACTORS)}")
    print(f"  = {n_scenarios} scenarios")
    print(f"α values: {n_alpha} ({ALPHA_VALUES[0]:.2f} to {ALPHA_VALUES[-1]:.2f})")
    print(f"Replications: {args.n_reps}")
    print(f"Total experiments: {n_scenarios * n_alpha * args.n_reps}")
    print()

    if args.estimate:
        est = estimate_runtime(n_scenarios, n_alpha, args.n_reps)
        print("=== Runtime Estimates ===")
        print(f"Time per experiment: {est['time_per_exp_sec']:.1f} seconds")
        print(f"Total serial time: {est['total_serial_hours']:.1f} hours")
        print(f"Parallel (64 cores): {est['parallel_64cores_hours']:.1f} hours")
        print(f"Parallel (256 cores): {est['parallel_256cores_hours']:.1f} hours")
        return

    if args.slurm:
        generate_slurm_scripts(scenarios, args.output, n_reps=args.n_reps)
        return

    if args.full_table:
        df = generate_experiment_table(scenarios, n_reps=args.n_reps)
        df.to_csv(args.output, index=False)
        print(f"Full experiment table saved to {args.output}")
        print(f"Shape: {df.shape}")
    else:
        df = pd.DataFrame(scenarios)
        df.to_csv(args.output, index=False)
        print(f"Scenario table saved to {args.output}")
        print(f"Shape: {df.shape}")


if __name__ == '__main__':
    main()
