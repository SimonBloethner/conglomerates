#!/usr/bin/env python3
"""
Example script to run robustness analysis with scenario-based parametrizations.

This script demonstrates how to use the scenario-based parametrization system
to run controlled robustness checks.
"""

import subprocess
import os
import time
from generate_robustness_scenarios import RobustnessScenarioGenerator

def run_local_robustness_test():
    """Run a small subset of scenarios locally for testing"""
    
    generator = RobustnessScenarioGenerator()
    
    # Generate just a few test scenarios
    power_law_scenarios = generator.generate_scenarios('power_law')
    
    # Take first 3 scenarios for quick test
    test_scenarios = power_law_scenarios[:3]
    
    print("Running local robustness test with 3 scenarios...")
    
    for scenario in test_scenarios:
        print(f"\nRunning scenario: {scenario['scenario_name']}")
        
        # Build command
        cmd = [
            'python', 'parallel_counterfactuals.py',
            '--markets', str(scenario['markets']),
            '--firms_per_market', str(scenario['firms_per_market']),
            '--steps', str(scenario['steps']),
            '--merge_thresh', str(scenario['merge_thresh']),
            '--lookback', str(scenario['lookback']),
            '--cost_type', scenario['cost_type'],
            '--c0', str(scenario['c0']),
            '--c1', str(scenario['c1']),
            '--c2', str(scenario['c2']),
            '--scenario_name', scenario['scenario_name'],
            '--counterfactuals', '5',  # Small number for testing
            '--n_cores', '2',  # Use fewer cores for testing
            '--share', '0.1'  # Test with single share
        ]
        
        if scenario['proportional']:
            cmd.append('--proportional')
        
        # Run command
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
            
            if result.returncode == 0:
                print(f"✅ {scenario['scenario_name']} completed successfully")
            else:
                print(f"❌ {scenario['scenario_name']} failed:")
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {scenario['scenario_name']} timed out (>5 minutes)")
        except Exception as e:
            print(f"❌ {scenario['scenario_name']} error: {e}")
    
    print("\nLocal test completed!")
    print("Check robustness_results/ directory for results")

def setup_slurm_robustness():
    """Set up SLURM submission for full robustness analysis"""
    
    print("Setting up SLURM robustness analysis...")
    
    # Generate all scenarios and scripts
    subprocess.run(['python', 'generate_robustness_scenarios.py'])
    
    print("\nSLurm setup completed!")
    print("To submit all jobs: ./submit_all_robustness.sh")
    print("To submit individual cost functions:")
    print("  sbatch robustness_slurm_scripts/submit_linear_*.sh")
    print("  sbatch robustness_slurm_scripts/submit_quadratic_*.sh")
    print("  etc.")

def analyze_robustness_results(plot_only=False):
    """Analyze completed robustness results with controlled regressions"""

    if plot_only:
        print("Regenerating plots from existing results...")
    else:
        print("Analyzing robustness results...")

        # First merge individual scenario results
        robustness_dirs = [d for d in os.listdir('robustness_results')
                          if os.path.isdir(f'robustness_results/{d}')]

        print(f"Found {len(robustness_dirs)} robustness scenarios")

        # Merge results for each scenario
        for scenario_dir in robustness_dirs:
            print(f"Merging {scenario_dir}...")
            cmd = [
                'python', 'merge_counterfactual_results.py',
                '--results_dir', f'robustness_results/{scenario_dir}',
                '--output', 'counterfactual_results_final.pkl'
            ]

            subprocess.run(cmd)

    # Run controlled analysis across all scenarios
    print("\nRunning cross-parametrization analysis with controls...")
    cmd = [
        'python', 'compare_parametrizations.py',
        '--results_dir', 'results',  # Will find robustness_* directories
        '--dashboard_dir', 'robustness_sensitivity_plots'
    ]

    if plot_only:
        cmd.append('--plot_only')

    subprocess.run(cmd)

    print("\nRobustness analysis completed!")
    print("Check robustness_sensitivity_plots/ for controlled regression results")

def main():
    """Main robustness analysis workflow"""
    import argparse

    parser = argparse.ArgumentParser(description='Robustness analysis workflow')
    parser.add_argument('command', choices=['test', 'setup', 'analyze', 'status'],
                       help='Command to run: test (local), setup (SLURM), analyze (results), status (check progress)')
    parser.add_argument('--plot_only', action='store_true',
                       help='Only regenerate plots from existing results (skip merging and analysis)')

    args = parser.parse_args()

    if args.command == 'test':
        run_local_robustness_test()
    elif args.command == 'setup':
        setup_slurm_robustness()
    elif args.command == 'analyze':
        if not args.plot_only:
            # Run status check before analysis
            print("Checking robustness analysis status before proceeding...")
            subprocess.run(['python', 'robustness_status_report.py'])
            print("\nProceeding with analysis...")
        analyze_robustness_results(plot_only=args.plot_only)
    elif args.command == 'status':
        subprocess.run(['python', 'robustness_status_report.py'])

if __name__ == "__main__":
    main()