#!/usr/bin/env python3
"""
Pilot study runner (§7).

Generates factorial design for Phase B pilot experiments:
- Fixed: M=N=50, lookback=50, p=0.05, proportional=False, lognormal growth
- Treatments: sharing_rule × sigma_range × rho × cost_type
- 48 scenarios × 9 α × 5 reps = 2,160 runs
- Plus reference scenarios for Phase A comparison

Usage:
    # Generate SLURM array job
    python run_pilot.py --slurm

    # Run locally (for testing)
    python run_pilot.py --local --scenario 0

    # Estimate runtime
    python run_pilot.py --estimate
"""
import argparse
import itertools
import os
import numpy as np
import zlib
import pickle
from pathlib import Path


# ============================================================================
# PILOT DESIGN CONFIGURATION
# ============================================================================

# Fixed parameters
M = 50                      # Markets
N = 50                      # Firms per market
LOOKBACK = 50               # Lookback period
MERGE_THRESH = 0.05         # p = 0.05
PROPORTIONAL = False        # Cost branch (not proportional)
GROWTH_PROCESS = 'lognormal'  # Phase B growth
MU_RANGE = (0.01, 0.1)      # Mean growth rate bounds
CROSS_CORR = 0.0            # No cross-market correlation
T = 5000                    # Total steps
BURN_IN = 4000              # Burn-in from §6 diagnostics
N_REPS = 5                  # Replications with common random numbers

# α grid (9 values)
ALPHA_VALUES = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])

# Factorial treatment factors
FACTORS = {
    'sharing_rule': ['equal', 'proportional'],
    'sigma_range': [(0.05, 0.10), (0.10, 0.20), (0.20, 0.40)],
    'rho': [0.0, 0.3],
    'cost_type': ['linear', 'quadratic', 'exponential', 'power_law'],
}

# Reference scenario configurations
REF_PHASE_A = {
    'name_prefix': 'ref_phaseA',
    'growth_process': 'normal_net',
    'sigma_range': (0.01, 0.05),
    'sharing_rule': 'equal',
    'rho': 0.0,
    'cost_types': ['linear', 'quadratic', 'exponential', 'power_law'],
}

REF_OLD_ALPHA0 = {
    'name_prefix': 'ref_old_alpha0',
    # Uses pre-referee-fixes code at α=0 only
    'alpha_only': 0.0,
    'cost_types': ['linear', 'quadratic', 'exponential', 'power_law'],
}


def generate_scenario_name(sharing, sigma_range, rho, cost_type):
    """Generate scenario name: pilot_<sharing>_sig<lo>-<hi>_rho<rho>_<cost>"""
    sig_lo = f"{sigma_range[0]:.2f}".replace("0.", "")
    sig_hi = f"{sigma_range[1]:.2f}".replace("0.", "")
    rho_str = f"{rho:.1f}".replace(".", "")
    return f"pilot_{sharing}_sig{sig_lo}-{sig_hi}_rho{rho_str}_{cost_type}"


def generate_factorial_scenarios():
    """Generate all 48 factorial treatment combinations."""
    scenarios = []
    scenario_id = 0

    for combo in itertools.product(
        FACTORS['sharing_rule'],
        FACTORS['sigma_range'],
        FACTORS['rho'],
        FACTORS['cost_type']
    ):
        sharing_rule, sigma_range, rho, cost_type = combo
        name = generate_scenario_name(sharing_rule, sigma_range, rho, cost_type)

        scenarios.append({
            'scenario_id': scenario_id,
            'scenario_name': name,
            'sharing_rule': sharing_rule,
            'sigma_range': sigma_range,
            'rho': rho,
            'cost_type': cost_type,
            'growth_process': GROWTH_PROCESS,
            'mu_range': MU_RANGE,
            'cross_corr': CROSS_CORR,
        })
        scenario_id += 1

    return scenarios


def generate_reference_scenarios(start_id):
    """Generate reference scenarios for Phase A comparison."""
    scenarios = []
    scenario_id = start_id

    # ref_phaseA: Phase A model at pilot scale
    for cost_type in REF_PHASE_A['cost_types']:
        name = f"{REF_PHASE_A['name_prefix']}_{cost_type}"
        scenarios.append({
            'scenario_id': scenario_id,
            'scenario_name': name,
            'sharing_rule': REF_PHASE_A['sharing_rule'],
            'sigma_range': REF_PHASE_A['sigma_range'],
            'rho': REF_PHASE_A['rho'],
            'cost_type': cost_type,
            'growth_process': REF_PHASE_A['growth_process'],
            'mu_range': MU_RANGE,
            'cross_corr': CROSS_CORR,
            'is_reference': True,
        })
        scenario_id += 1

    return scenarios


def generate_all_tasks():
    """Generate all (scenario, α, rep) tasks."""
    factorial = generate_factorial_scenarios()
    reference = generate_reference_scenarios(len(factorial))
    all_scenarios = factorial + reference

    tasks = []
    task_id = 0

    for scenario in all_scenarios:
        for alpha in ALPHA_VALUES:
            for rep in range(N_REPS):
                # Common random numbers: seed depends on (scenario_name, rep), not α
                seed = zlib.crc32(f"{scenario['scenario_name']}:{rep}".encode()) & 0xFFFFFFFF
                tasks.append({
                    'task_id': task_id,
                    'scenario_id': scenario['scenario_id'],
                    'scenario_name': scenario['scenario_name'],
                    'alpha': alpha,
                    'rep': rep,
                    'seed': seed,
                    **{k: v for k, v in scenario.items()
                       if k not in ['scenario_id', 'scenario_name']},
                })
                task_id += 1

    return tasks, all_scenarios


def estimate_runtime():
    """Estimate total CPU-hours for pilot."""
    tasks, scenarios = generate_all_tasks()
    n_tasks = len(tasks)

    # Estimate: ~60 seconds per task at 50×50 scale with T=5000
    seconds_per_task = 60
    total_seconds = n_tasks * seconds_per_task
    total_hours = total_seconds / 3600

    n_factorial = len(generate_factorial_scenarios())
    n_reference = len(scenarios) - n_factorial

    print("=== Pilot Study Design ===")
    print(f"Fixed: M={M}, N={N}, T={T}, burn_in={BURN_IN}")
    print(f"α values: {len(ALPHA_VALUES)} ({ALPHA_VALUES[0]:.2f} to {ALPHA_VALUES[-1]:.2f})")
    print(f"Replications: {N_REPS}")
    print()
    print(f"Factorial scenarios: {n_factorial}")
    print(f"  sharing_rule: {FACTORS['sharing_rule']}")
    print(f"  sigma_range: {FACTORS['sigma_range']}")
    print(f"  rho: {FACTORS['rho']}")
    print(f"  cost_type: {FACTORS['cost_type']}")
    print(f"  = {len(FACTORS['sharing_rule'])} × {len(FACTORS['sigma_range'])} × "
          f"{len(FACTORS['rho'])} × {len(FACTORS['cost_type'])} = {n_factorial}")
    print()
    print(f"Reference scenarios: {n_reference}")
    print(f"  ref_phaseA: {len(REF_PHASE_A['cost_types'])} cost types")
    print()
    print(f"Total tasks: {n_tasks}")
    print()
    print("=== Runtime Estimate ===")
    print(f"Estimated time per task: {seconds_per_task}s")
    print(f"Total serial time: {total_hours:.1f} CPU-hours")
    print(f"With 64 cores: {total_hours/64:.1f} hours")
    print(f"With 128 cores: {total_hours/128:.1f} hours")


def generate_slurm_script(output_dir='pilot_jobs'):
    """Generate SLURM array job scripts (split if > 1000 tasks)."""
    tasks, scenarios = generate_all_tasks()
    n_tasks = len(tasks)
    max_array_size = 1000  # SLURM limit

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/logs', exist_ok=True)

    # Save task list as pickle for job to read
    task_file = f'{output_dir}/pilot_tasks.pkl'
    with open(task_file, 'wb') as f:
        pickle.dump({'tasks': tasks, 'scenarios': scenarios}, f)

    # Split into multiple array jobs if needed
    n_jobs = (n_tasks + max_array_size - 1) // max_array_size
    submit_lines = ['#!/bin/bash', '# Submit all pilot array jobs', '']

    for job_idx in range(n_jobs):
        start_idx = job_idx * max_array_size
        end_idx = min((job_idx + 1) * max_array_size - 1, n_tasks - 1)
        array_size = end_idx - start_idx + 1

        script = f'''#!/bin/bash
#SBATCH --job-name=pilot_{job_idx}
#SBATCH --partition=normal
#SBATCH --array=0-{array_size-1}%128
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --output={output_dir}/logs/pilot_{job_idx}_%A_%a.out
#SBATCH --error={output_dir}/logs/pilot_{job_idx}_%A_%a.err

# Load modules
module load python/3.12.4

# Change to working directory
cd /groups/m-larch/bt307958/conglomerates_dev

# Calculate actual task ID (offset by job batch)
TASK_ID=$(( {start_idx} + $SLURM_ARRAY_TASK_ID ))

# Run single task
python run_pilot.py --run-task $TASK_ID --task-file {task_file}

echo "Task $TASK_ID completed"
'''

        script_path = f'{output_dir}/submit_pilot_{job_idx}.sh'
        with open(script_path, 'w') as f:
            f.write(script)

        submit_lines.append(f'sbatch submit_pilot_{job_idx}.sh')

    # Write master submit script
    submit_all_path = f'{output_dir}/submit_all.sh'
    with open(submit_all_path, 'w') as f:
        f.write('\n'.join(submit_lines))

    print(f"Generated {n_jobs} SLURM array job(s)")
    print(f"Task file: {task_file}")
    print(f"Total tasks: {n_tasks}")
    for job_idx in range(n_jobs):
        start_idx = job_idx * max_array_size
        end_idx = min((job_idx + 1) * max_array_size - 1, n_tasks - 1)
        print(f"  Job {job_idx}: tasks {start_idx}-{end_idx}")
    print()
    print("Submit with:")
    print(f"  cd {output_dir} && bash submit_all.sh")


def run_single_task(task_id, task_file):
    """Run a single task from the task file."""
    import collaborative_growth
    from collaborative_growth import seed_numba, get_cost_function_defaults

    # Load task definition
    with open(task_file, 'rb') as f:
        data = pickle.load(f)

    tasks = data['tasks']
    task = tasks[task_id]

    print(f"Running task {task_id}: {task['scenario_name']}, α={task['alpha']:.2f}, rep={task['rep']}")

    # Set up parameters
    alpha = task['alpha']
    seed = task['seed']

    # Get cost function defaults
    cost_type = task['cost_type']
    defaults = get_cost_function_defaults(cost_type)

    # Seed RNGs
    np.random.seed(seed)
    seed_numba(seed)

    # Model parameters (14-element format)
    params = [
        M, N, T, alpha, M * N,
        MERGE_THRESH, 4, 0.85, PROPORTIONAL, LOOKBACK,
        cost_type, defaults['c0'], defaults['c1'], defaults['c2']
    ]

    # Run model
    result = collaborative_growth.model(
        params=params,
        seed=seed,
        market_corr='identity',
        growth_process=task['growth_process'],
        mu_range=task['mu_range'],
        sigma_range=task['sigma_range'],
        sharing_rule=task['sharing_rule'],
        rho=task['rho'],
        cross_corr=task['cross_corr'],
    )

    # Unpack results (13 elements with online rank stats)
    (mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares,
     gini_coefficient, avg_ranks, mergers_per_period, proposals_per_period,
     exits_per_period, rank_range, rank_std, hyperparameters) = result

    # Extract summary statistics (post burn-in)
    post_burnin = slice(BURN_IN, T)

    # Compute acceptance rate from proposals and mergers
    total_proposals = np.sum(proposals_per_period[post_burnin])
    total_mergers = np.sum(mergers_per_period[post_burnin])
    acceptance_rate = total_mergers / max(total_proposals, 1)

    # Summary statistics
    summary = {
        'task_id': task_id,
        'scenario_name': task['scenario_name'],
        'alpha': alpha,
        'rep': task['rep'],
        'seed': seed,

        # Treatment factors
        'sharing_rule': task['sharing_rule'],
        'sigma_range': task['sigma_range'],
        'rho': task['rho'],
        'cost_type': cost_type,

        # Key metrics (post burn-in averages)
        'acceptance_rate': acceptance_rate,
        'mean_size': np.mean(mean_members[post_burnin]),
        'num_cong': np.mean(num_cong[post_burnin]),
        'exits_per_period': np.mean(exits_per_period[post_burnin]),

        # Share quantiles (post burn-in, averaged over markets)
        # quantiles_shares shape: (7_quantiles, markets, steps)
        # Quantiles are: 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0
        'share_q50': np.mean(quantiles_shares[2, :, post_burnin]),  # median
        'share_q99': np.mean(quantiles_shares[5, :, post_burnin]),  # 99th
        'share_q100': np.mean(quantiles_shares[6, :, post_burnin]),  # max

        # Rank mobility (median across all firms)
        'rank_range_median': np.median(rank_range),
        'rank_std_median': np.median(rank_std),

        # Gini (post burn-in, averaged over markets)
        'gini': np.mean(gini_coefficient[:, post_burnin]),

        # Store hyperparameters for verification
        'hyperparameters': hyperparameters,
    }

    # Save result
    results_dir = Path('results/pilot')
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / f"{task['scenario_name']}_alpha{alpha:.2f}_rep{task['rep']:02d}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(summary, f)

    print(f"Saved: {output_file}")
    print(f"  acceptance_rate={acceptance_rate:.4f}, mean_size={summary['mean_size']:.2f}, "
          f"num_cong={summary['num_cong']:.1f}")

    return summary


def run_local_scenario(scenario_id):
    """Run a single scenario locally (all α values, all reps)."""
    tasks, scenarios = generate_all_tasks()

    # Filter tasks for this scenario
    scenario_tasks = [t for t in tasks if t['scenario_id'] == scenario_id]

    if not scenario_tasks:
        print(f"No tasks found for scenario_id={scenario_id}")
        return

    print(f"Running scenario {scenario_id}: {scenario_tasks[0]['scenario_name']}")
    print(f"Tasks: {len(scenario_tasks)}")

    # Save task file temporarily
    task_file = '/tmp/pilot_tasks_local.pkl'
    with open(task_file, 'wb') as f:
        pickle.dump({'tasks': tasks, 'scenarios': scenarios}, f)

    # Run all tasks for this scenario
    for task in scenario_tasks:
        run_single_task(task['task_id'], task_file)


def main():
    parser = argparse.ArgumentParser(description='Pilot study runner')
    parser.add_argument('--estimate', action='store_true',
                       help='Print runtime estimate')
    parser.add_argument('--slurm', action='store_true',
                       help='Generate SLURM array job script')
    parser.add_argument('--local', action='store_true',
                       help='Run locally (requires --scenario)')
    parser.add_argument('--scenario', type=int, default=None,
                       help='Scenario ID to run locally')
    parser.add_argument('--run-task', type=int, default=None,
                       help='Run single task by ID (used by SLURM)')
    parser.add_argument('--task-file', type=str, default=None,
                       help='Path to task file (used by SLURM)')
    parser.add_argument('--output-dir', type=str, default='pilot_jobs',
                       help='Output directory for SLURM scripts')

    args = parser.parse_args()

    if args.estimate:
        estimate_runtime()
    elif args.slurm:
        generate_slurm_script(args.output_dir)
    elif args.run_task is not None:
        if args.task_file is None:
            print("Error: --task-file required with --run-task")
            return
        run_single_task(args.run_task, args.task_file)
    elif args.local:
        if args.scenario is None:
            print("Error: --scenario required with --local")
            return
        run_local_scenario(args.scenario)
    else:
        # Default: print design summary
        tasks, scenarios = generate_all_tasks()
        print(f"Pilot study: {len(tasks)} tasks across {len(scenarios)} scenarios")
        print(f"Use --estimate, --slurm, or --local --scenario N")


if __name__ == '__main__':
    main()
