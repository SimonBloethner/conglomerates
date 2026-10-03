#!/usr/bin/env python3
"""
Generate scenario-based parametrizations for robustness analysis.
Changes only one parameter at a time from baseline to avoid interaction effects.
"""

import os
import itertools
from collaborative_growth import get_cost_function_defaults

class RobustnessScenarioGenerator:
    def __init__(self):
        # Economic parameter baselines (same for all cost functions)
        self.economic_baseline = {
            'markets': 100,
            'firms_per_market': 100, 
            'merge_thresh': 0.05,
            'lookback': 50,
            'steps': 10000,  # Keep fixed for computational efficiency
            'proportional': False,  # Keep fixed
        }
        
        # Economic parameter variations
        self.economic_variations = {
            'markets': [50, 100, 150],
            'firms_per_market': [50, 100, 150],
            'merge_thresh': [0.01, 0.05, 0.1], 
            'lookback': [10, 50, 100],
        }
        
        # Cost function parameter variations using MULTIPLICATIVE PERTURBATIONS
        # of calibrated defaults (all converge to ~0.0017 at size=40)
        # Level parameters: c0 (and c1 for linear/quadratic) × {0.5, 1, 2}
        # Shape parameters: c1 (exp/power_law) or c2 (quadratic) × {0.8, 1, 1.25}
        self.level_multipliers = [0.5, 1.0, 2.0]
        self.shape_multipliers = [0.8, 1.0, 1.25]
        
        # Build variations dynamically from calibrated defaults
        self.cost_variations = {}
        for cost_type in ['linear', 'quadratic', 'exponential', 'power_law']:
            defaults = get_cost_function_defaults(cost_type)
            
            if cost_type == 'linear':
                # Level: c0, c1; no shape parameter
                self.cost_variations[cost_type] = {
                    'c0': [defaults['c0'] * m for m in self.level_multipliers],
                    'c1': [defaults['c1'] * m for m in self.level_multipliers],
                    'c2': [defaults['c2']]  # Not used
                }
            elif cost_type == 'quadratic':
                # Level: c0, c1; Shape: c2
                self.cost_variations[cost_type] = {
                    'c0': [defaults['c0'] * m for m in self.level_multipliers],
                    'c1': [defaults['c1'] * m for m in self.level_multipliers],
                    'c2': [defaults['c2'] * m for m in self.shape_multipliers]
                }
            elif cost_type == 'exponential':
                # Level: c0; Shape: c1
                self.cost_variations[cost_type] = {
                    'c0': [defaults['c0'] * m for m in self.level_multipliers],
                    'c1': [defaults['c1'] * m for m in self.shape_multipliers],
                    'c2': [defaults['c2']]  # Not used
                }
            elif cost_type == 'power_law':
                # Level: c0; Shape: c1
                self.cost_variations[cost_type] = {
                    'c0': [defaults['c0'] * m for m in self.level_multipliers],
                    'c1': [defaults['c1'] * m for m in self.shape_multipliers],
                    'c2': [defaults['c2']]  # Not used
                }
    
    def get_baseline_config(self, cost_type):
        """Get baseline configuration using CALIBRATED DEFAULTS"""
        # Use calibrated defaults (converge to ~0.0017 at size=40)
        defaults = get_cost_function_defaults(cost_type)
        
        baseline = self.economic_baseline.copy()
        baseline.update({
            'cost_type': cost_type,
            'c0': defaults['c0'],  # Calibrated default
            'c1': defaults['c1'],  # Calibrated default
            'c2': defaults['c2'],  # Calibrated default
        })
        
        return baseline
    
    def generate_scenarios(self, cost_type):
        """Generate all scenarios for a specific cost function"""
        baseline = self.get_baseline_config(cost_type)
        scenarios = []
        
        # Scenario 0: Pure baseline
        scenarios.append(baseline.copy())
        
        # Economic parameter scenarios  
        for param, values in self.economic_variations.items():
            for value in values:
                if value != baseline[param]:  # Skip if same as baseline
                    scenario = baseline.copy()
                    scenario[param] = value
                    scenario['scenario_name'] = f"{cost_type}_vary_{param}_{value}"
                    scenarios.append(scenario)
        
        # Cost parameter scenarios
        cost_vars = self.cost_variations[cost_type]
        for param in ['c0', 'c1', 'c2']:
            if param in cost_vars and len(cost_vars[param]) > 1:
                for value in cost_vars[param]:
                    if value != baseline[param]:  # Skip if same as baseline  
                        scenario = baseline.copy()
                        scenario[param] = value
                        scenario['scenario_name'] = f"{cost_type}_vary_{param}_{value}"
                        scenarios.append(scenario)
        
        # Add scenario names for baseline
        scenarios[0]['scenario_name'] = f"{cost_type}_baseline"
        
        return scenarios
    
    def generate_all_scenarios(self):
        """Generate scenarios for all cost functions"""
        all_scenarios = {}
        
        for cost_type in ['linear', 'quadratic', 'exponential', 'power_law']:
            scenarios = self.generate_scenarios(cost_type)
            all_scenarios[cost_type] = scenarios
            
            print(f"{cost_type}: {len(scenarios)} scenarios generated")
            
        return all_scenarios
    
    def create_slurm_array_script(self, scenario, counterfactuals=50, chunk_size=2, throttle=10):
        """Create SLURM job array script for a scenario"""
        scenario_name = scenario['scenario_name']
        n_chunks = counterfactuals // chunk_size

        # OPTIMIZED: Use 10 cores based on validated baseline configuration
        # Validated: 10 cores, 10 experiments in 12 min (50-core had severe overhead)
        # Measured: ~220MB per worker baseline, up to 700MB for large scenarios
        n_cores = 10
        memory = "10G"  # Safe: measured up to 7GB for large scenarios (150 markets/firms)

        # Runtime varies significantly by scenario size:
        # Baseline (100×100): 2 exp × 26 shares × 10 min/share = ~50 min
        # Large (150×150): 2 exp × 26 shares × 75 min/share = ~6.5 hours
        # Use 24-hour limit to handle worst case with buffer
        time_limit = "24:00:00"  # 24 hours: max allowed, handles worst case with safety margin

        script_content = f"""#!/bin/bash
#SBATCH --job-name={scenario_name}
#SBATCH --array=0-{n_chunks-1}%{throttle}
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task={n_cores}
#SBATCH --mem={memory}
#SBATCH --time={time_limit}
#SBATCH --output=/groups/m-larch/bt307958/conglomerates/robustness_slurm_logs/{scenario_name}_%a.out
#SBATCH --error=/groups/m-larch/bt307958/conglomerates/robustness_slurm_logs/{scenario_name}_%a.err

# Load required modules
module load python/3.13.3

# Change to the correct directory
cd /groups/m-larch/bt307958/conglomerates

# Activate environment
source cong_env/bin/activate

# Calculate chunk range from array task ID
CHUNK_ID=$SLURM_ARRAY_TASK_ID
CHUNK_START=$((CHUNK_ID * {chunk_size}))
CHUNK_END=$((CHUNK_START + {chunk_size}))

echo "Starting scenario: {scenario_name}, array task $CHUNK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "Configuration: {n_cores} cores, {memory} memory, experiments $CHUNK_START-$((CHUNK_END-1))"

# Run the counterfactual analysis with scenario-specific parameters and chunking
$VIRTUAL_ENV/bin/python parallel_counterfactuals.py \\
    --markets {scenario['markets']} \\
    --firms_per_market {scenario['firms_per_market']} \\
    --steps {scenario['steps']} \\
    --merge_thresh {scenario['merge_thresh']} \\
    --lookback {scenario['lookback']} \\
    --cost_type {scenario['cost_type']} \\
    --c0 {scenario['c0']} \\
    --c1 {scenario['c1']} \\
    --c2 {scenario['c2']} \\
    --counterfactuals {counterfactuals} \\
    --chunk ${{CHUNK_START}}_${{CHUNK_END}} \\
    --n_cores {n_cores} \\
    {"--proportional" if scenario['proportional'] else ""} \\
    --scenario_name {scenario_name}

echo "Scenario {scenario_name}, array task $CHUNK_ID completed at: $(date)"
"""

        return script_content
    
    def setup_directories(self):
        """Create necessary directories for robustness analysis"""
        directories = [
            'robustness_slurm_logs', 
            'robustness_slurm_scripts',
            'robustness_results'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def generate_submission_scripts(self, scenarios, output_dir='robustness_slurm_scripts',
                                   counterfactuals=50, chunk_size=2, throttle=10):
        """Generate all SLURM job array submission scripts"""
        self.setup_directories()
        script_files = []

        for cost_type, cost_scenarios in scenarios.items():
            for scenario in cost_scenarios:
                scenario_name = scenario['scenario_name']

                # Create one SLURM array script per scenario
                script_content = self.create_slurm_array_script(scenario, counterfactuals,
                                                               chunk_size, throttle)
                script_file = f"{output_dir}/submit_{scenario_name}.sh"

                with open(script_file, 'w') as f:
                    f.write(script_content)

                # Make executable
                os.chmod(script_file, 0o755)
                script_files.append(script_file)

                print(f"Created: {script_file}")

        return script_files
    
    def create_master_submit_script(self, script_files, output_file='submit_all_robustness.sh'):
        """Create master script to submit all scenarios"""
        
        master_content = f"""#!/bin/bash
# Master script to submit all robustness scenarios
# Generated {len(script_files)} scenarios

echo "Submitting {len(script_files)} robustness analysis scenarios..."

"""
        
        for script_file in script_files:
            master_content += f"echo \"Submitting {script_file}...\"\n"
            master_content += f"sbatch {script_file}\n"
            master_content += f"sleep 1  # Brief pause between submissions\n\n"
        
        master_content += f"""
echo "All {len(script_files)} scenarios submitted!"
echo "Monitor progress with: squeue -u $USER"
echo "Check logs in: robustness_slurm_logs/"
"""
        
        with open(output_file, 'w') as f:
            f.write(master_content)
        
        os.chmod(output_file, 0o755)
        print(f"Created master submission script: {output_file}")
        
        return output_file


def generate_cost_table():
    """Generate robustness_cost_table.md showing Phi(size) for all scenarios"""
    from collaborative_growth import management_cost_function, get_cost_function_defaults
    
    sizes = [2, 5, 10, 20, 40]
    generator = RobustnessScenarioGenerator()
    scenarios = generator.generate_all_scenarios()
    
    lines = []
    lines.append("# Robustness Cost Table")
    lines.append("")
    lines.append("Management cost Φ(size) for each scenario at sizes {2, 5, 10, 20, 40}.")
    lines.append("Calibrated defaults converge to ~0.0017 at size=40.")
    lines.append("")
    
    for cost_type in ['linear', 'quadratic', 'exponential', 'power_law']:
        lines.append(f"## {cost_type.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Scenario | Φ(2) | Φ(5) | Φ(10) | Φ(20) | Φ(40) |")
        lines.append("|----------|------|------|-------|-------|-------|")
        
        for scenario in scenarios[cost_type]:
            name = scenario['scenario_name']
            costs = []
            for size in sizes:
                cost = management_cost_function(size, cost_type, 
                                               scenario['c0'], scenario['c1'], scenario['c2'])
                costs.append(f"{cost:.6f}")
            lines.append(f"| {name} | {' | '.join(costs)} |")
        
        lines.append("")
    
    with open("robustness_cost_table.md", "w") as f:
        f.write("\n".join(lines))
    
    print("Generated robustness_cost_table.md")


def main():
    """Generate all robustness scenarios and SLURM job array scripts"""
    generator = RobustnessScenarioGenerator()

    # Job array configuration
    counterfactuals = 50
    chunk_size = 2  # Smaller chunks for better load balancing
    throttle = 10   # Max 10 array tasks running per scenario
    n_chunks = counterfactuals // chunk_size

    # Generate scenarios
    print("Generating robustness scenarios...")
    scenarios = generator.generate_all_scenarios()

    # Calculate totals
    total_scenarios = sum(len(cost_scenarios) for cost_scenarios in scenarios.values())
    total_array_tasks = total_scenarios * n_chunks
    max_concurrent = total_scenarios * throttle

    print(f"\nJob Array Configuration:")
    print(f"  Total scenarios: {total_scenarios}")
    print(f"  Counterfactuals per scenario: {counterfactuals}")
    print(f"  Chunk size: {chunk_size} experiments per task")
    print(f"  Array tasks per scenario: {n_chunks}")
    print(f"  Throttle: {throttle} concurrent tasks per scenario")
    print(f"\nQueue Summary:")
    print(f"  Jobs in queue: {total_scenarios} (one array per scenario)")
    print(f"  Total array tasks: {total_array_tasks}")
    print(f"  Max concurrent tasks: {max_concurrent}")
    print(f"\nExpected Runtime per Task:")
    print(f"  Baseline (100×100): ~50 min")
    print(f"  Large (150×150): ~6.5 hours")

    # Generate SLURM scripts
    print("\nGenerating SLURM job array scripts...")
    script_files = generator.generate_submission_scripts(scenarios,
                                                         counterfactuals=counterfactuals,
                                                         chunk_size=chunk_size,
                                                         throttle=throttle)

    # Create master submission script
    master_script = generator.create_master_submit_script(script_files)

    print(f"\nSetup complete!")
    print(f"Generated {len(script_files)} job array scripts")
    print(f"To submit all scenarios: ./{master_script}")
    print(f"To submit individual scenarios: sbatch robustness_slurm_scripts/submit_<scenario_name>.sh")


if __name__ == "__main__":
    main()