#!/usr/bin/env python3
"""
Generate comprehensive status report for robustness analysis.
Checks job completion, file integrity, and identifies problems.
"""

import os
import glob
import pickle
import subprocess
from pathlib import Path
from generate_robustness_scenarios import RobustnessScenarioGenerator

class RobustnessStatusReport:
    def __init__(self):
        self.generator = RobustnessScenarioGenerator()
        self.all_scenarios = self.generator.generate_all_scenarios()
        
    def check_slurm_jobs(self):
        """Check SLURM job completion status"""
        print("=" * 60)
        print("SLURM JOB STATUS")
        print("=" * 60)
        
        try:
            # Get completed jobs from sacct
            result = subprocess.run(['sacct', '-u', os.getenv('USER'), '--format=JobName,State,ExitCode', 
                                   '--noheader', '--parsable2'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                print("Warning: Could not fetch SLURM job history")
                return {}
            
            job_status = {}
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        job_name, state, exit_code = parts[0], parts[1], parts[2]
                        
                        # Filter for robustness jobs
                        if any(cost in job_name for cost in ['linear', 'quadratic', 'exponential', 'power_law']):
                            job_status[job_name] = {'state': state, 'exit_code': exit_code}
            
            # Summarize job status
            completed = sum(1 for status in job_status.values() if status['state'] == 'COMPLETED')
            failed = sum(1 for status in job_status.values() if status['state'] == 'FAILED')
            
            print(f"Jobs found in SLURM history: {len(job_status)}")
            print(f"✅ Completed: {completed}")
            print(f"❌ Failed: {failed}")
            print(f"❓ Other states: {len(job_status) - completed - failed}")
            
            # Show failed jobs
            if failed > 0:
                print("\nFAILED JOBS:")
                for job_name, status in job_status.items():
                    if status['state'] == 'FAILED':
                        print(f"  ❌ {job_name} (exit code: {status['exit_code']})")
            
            return job_status
            
        except Exception as e:
            print(f"Error checking SLURM jobs: {e}")
            return {}
    
    def check_scenario_results(self):
        """Check if all scenario results exist and are complete"""
        print("\n" + "=" * 60)
        print("SCENARIO RESULTS STATUS")
        print("=" * 60)
        
        expected_scenarios = []
        for cost_type, scenarios in self.all_scenarios.items():
            for scenario in scenarios:
                expected_scenarios.append(scenario['scenario_name'])
        
        print(f"Expected scenarios: {len(expected_scenarios)}")
        
        # Check robustness_results directories
        results_found = 0
        results_missing = []
        results_complete = 0
        results_incomplete = []
        
        for scenario_name in expected_scenarios:
            result_dir = Path(f'robustness_results/{scenario_name}')
            
            if result_dir.exists():
                results_found += 1

                # Check if results are complete (25 chunk files expected with chunk_size=2)
                chunk_files = list(result_dir.glob('all_shares_chunk_*.pkl'))
                expected_chunks = 25  # 50 experiments / 2 per chunk = 25 chunks

                if len(chunk_files) >= expected_chunks:
                    results_complete += 1
                else:
                    results_incomplete.append((scenario_name, len(chunk_files), expected_chunks))
            else:
                results_missing.append(scenario_name)
        
        print(f"✅ Result directories found: {results_found}/{len(expected_scenarios)}")
        print(f"✅ Complete scenarios: {results_complete}")
        print(f"⚠️  Incomplete scenarios: {len(results_incomplete)}")
        print(f"❌ Missing scenarios: {len(results_missing)}")
        
        # Show details for incomplete/missing
        if results_incomplete:
            print("\nINCOMPLETE SCENARIOS:")
            for scenario, found, expected in results_incomplete:
                print(f"  ⚠️  {scenario}: {found}/{expected} chunk files")
        
        if results_missing:
            print("\nMISSING SCENARIOS:")
            for scenario in results_missing[:10]:  # Show first 10
                print(f"  ❌ {scenario}")
            if len(results_missing) > 10:
                print(f"  ... and {len(results_missing) - 10} more")
        
        return {
            'total': len(expected_scenarios),
            'found': results_found,
            'complete': results_complete,
            'incomplete': results_incomplete,
            'missing': results_missing
        }
    
    def check_merged_results(self):
        """Check if scenarios have been merged"""
        print("\n" + "=" * 60)
        print("MERGED RESULTS STATUS") 
        print("=" * 60)
        
        results_dir = Path('results')
        merged_found = 0
        merged_scenarios = []
        
        if results_dir.exists():
            for merged_dir in results_dir.glob('robustness_*'):
                if merged_dir.is_dir():
                    final_file = merged_dir / 'counterfactual_results_final.pkl'
                    if final_file.exists():
                        merged_found += 1
                        merged_scenarios.append(merged_dir.name)
                        
                        # Check file size as sanity check
                        file_size = final_file.stat().st_size / (1024 * 1024)  # MB
                        if file_size < 1:  # Less than 1 MB might be incomplete
                            print(f"  ⚠️  {merged_dir.name}: Small file size ({file_size:.1f} MB)")
        
        print(f"✅ Merged scenarios found: {merged_found}")
        
        return merged_found, merged_scenarios
    
    def check_log_files(self):
        """Check SLURM log files for errors"""
        print("\n" + "=" * 60)
        print("LOG FILE ANALYSIS")
        print("=" * 60)
        
        log_dir = Path('robustness_slurm_logs')
        errors_found = []
        
        if not log_dir.exists():
            print("❌ Log directory not found")
            return []
        
        error_files = list(log_dir.glob('*.err'))
        out_files = list(log_dir.glob('*.out'))
        
        print(f"Log files found: {len(error_files)} .err, {len(out_files)} .out")
        
        # Check error files for non-empty content
        for err_file in error_files:
            if err_file.stat().st_size > 0:  # Non-empty error file
                with open(err_file, 'r') as f:
                    content = f.read()
                    if 'Error' in content or 'Failed' in content or 'Exception' in content:
                        scenario_name = err_file.stem
                        errors_found.append((scenario_name, err_file))
        
        if errors_found:
            print(f"❌ Scenarios with errors: {len(errors_found)}")
            for scenario, err_file in errors_found[:5]:  # Show first 5
                print(f"  ❌ {scenario} (see {err_file})")
            if len(errors_found) > 5:
                print(f"  ... and {len(errors_found) - 5} more")
        else:
            print("✅ No obvious errors found in log files")
        
        return errors_found
    
    def check_data_integrity(self, sample_size=5):
        """Check a sample of result files for data integrity"""
        print("\n" + "=" * 60)
        print("DATA INTEGRITY CHECK")
        print("=" * 60)
        
        # Get a sample of merged result files
        results_dir = Path('results')
        sample_files = []
        
        if results_dir.exists():
            merged_dirs = list(results_dir.glob('robustness_*'))
            import random
            sample_dirs = random.sample(merged_dirs, min(sample_size, len(merged_dirs)))
            
            for sample_dir in sample_dirs:
                final_file = sample_dir / 'counterfactual_results_final.pkl'
                if final_file.exists():
                    sample_files.append(final_file)
        
        print(f"Checking {len(sample_files)} sample files...")
        
        integrity_issues = []
        
        for pkl_file in sample_files:
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Basic integrity checks
                if not isinstance(data, dict):
                    integrity_issues.append(f"{pkl_file.parent.name}: Not a dictionary")
                    continue
                
                if len(data) == 0:
                    integrity_issues.append(f"{pkl_file.parent.name}: Empty data")
                    continue
                
                # Check if we have expected alpha values
                expected_alphas = 26  # 0.00 to 0.50
                if len(data) < expected_alphas * 0.8:  # Allow some tolerance
                    integrity_issues.append(f"{pkl_file.parent.name}: Only {len(data)} alpha values")
                
                # Check if hyperparameters exist
                first_alpha = next(iter(data.values()))
                if 'hyperparameters' not in first_alpha:
                    integrity_issues.append(f"{pkl_file.parent.name}: Missing hyperparameters")
                
                print(f"  ✅ {pkl_file.parent.name}: OK ({len(data)} alpha values)")
                
            except Exception as e:
                integrity_issues.append(f"{pkl_file.parent.name}: Error loading - {e}")
        
        if integrity_issues:
            print("\nDATA ISSUES FOUND:")
            for issue in integrity_issues:
                print(f"  ⚠️  {issue}")
        
        return integrity_issues
    
    def generate_report(self):
        """Generate comprehensive status report"""
        print("ROBUSTNESS ANALYSIS STATUS REPORT")
        print("Generated:", __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Run all checks
        slurm_status = self.check_slurm_jobs()
        scenario_status = self.check_scenario_results()
        merged_count, merged_scenarios = self.check_merged_results()
        log_errors = self.check_log_files()
        data_issues = self.check_data_integrity()
        
        # Overall summary
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)
        
        total_scenarios = scenario_status['total']
        complete_scenarios = scenario_status['complete']
        
        completion_rate = (complete_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0
        
        print(f"Total scenarios expected: {total_scenarios}")
        print(f"Scenarios completed: {complete_scenarios}")
        print(f"Completion rate: {completion_rate:.1f}%")
        print(f"Scenarios merged: {merged_count}")
        print(f"Scenarios with errors: {len(log_errors)}")
        print(f"Data integrity issues: {len(data_issues)}")
        
        # Status indicator
        if completion_rate >= 95 and len(log_errors) == 0 and len(data_issues) == 0:
            print("\n🎉 STATUS: ALL GOOD - Ready for analysis!")
        elif completion_rate >= 80:
            print("\n⚠️  STATUS: MOSTLY COMPLETE - Some issues to investigate")
        else:
            print("\n❌ STATUS: INCOMPLETE - Significant issues found")
        
        print("\nNext steps:")
        if completion_rate < 100:
            print("  1. Check failed jobs and resubmit if needed")
            print("  2. Investigate missing scenarios")
        if merged_count < complete_scenarios:
            print("  3. Run merge process: python run_robustness_analysis.py analyze")
        if completion_rate >= 80:
            print("  4. Run controlled analysis: python compare_parametrizations.py --results_dir results")

def main():
    reporter = RobustnessStatusReport()
    reporter.generate_report()

if __name__ == "__main__":
    main()