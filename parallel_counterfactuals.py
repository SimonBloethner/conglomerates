#!/usr/bin/env python3
"""
Parallelized version of counterfactuals.py for cluster computation
VERSION: 2024-10-30-DEBUG with np.RankWarning fix and polynomial debugging
"""

import numpy as np
import multiprocessing as mp
from functools import partial
import pickle
import argparse
import os
from tqdm import tqdm
import warnings
import collaborative_growth
from collaborative_growth import seed_numba
import signal
import time
import subprocess
import zlib

def log_memory_usage(label):
    """Log current memory usage using ps command"""
    try:
        import os
        pid = os.getpid()
        result = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], 
                              capture_output=True, text=True)
        memory_kb = int(result.stdout.strip())
        memory_gb = memory_kb / 1024 / 1024
        print(f"MEMORY_LOG {label}: {memory_gb:.2f}GB", flush=True)
    except Exception as e:
        print(f"MEMORY_LOG {label}: Unable to measure ({e})", flush=True)

class ParallelCounterfactualRunner:
    def __init__(self, n_cores=64, counterfactuals=50):
        """
        Initialize parallel counterfactual runner
        
        Args:
            n_cores: Number of CPU cores to use
            counterfactuals: Number of replications per share value
        """
        self.n_cores = n_cores
        self.counterfactuals = counterfactuals
        
        # Fixed parameters (matching original counterfactuals.py)
        self.markets = 100
        self.firms_per_market = 100
        self.total_firms = self.markets * self.firms_per_market
        self.steps = 10000
        self.merge_thresh = 0.05
        self.comparison = 4
        self.break_thresh = 0.85
        self.lookback = 50
        self.ramp = 10
        self.proportional = False
        self.market_corr = 'identity'  # 'identity' (paper baseline) or 'random'

        # Growth process parameters (Phase B §1)
        self.growth_process = 'normal_net'  # 'normal_net' (Phase A) or 'lognormal'
        self.mu_range = (0.01, 0.1)  # Mean growth rate bounds
        self.sigma_range = (0.01, 0.05)  # Volatility bounds

        # Share values to test (α grid) - 9-point grid
        self.shares = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
        
        # Cost function parameters
        self.cost_type = 'power_law'  # Default cost function type
        self.c0 = None  # Will use cost-function-specific defaults
        self.c1 = None
        self.c2 = None
        
        # Backward compatibility - Power law parameters
        self.b0 = 0.00001  # Deprecated
        self.b1 = 1.2      # Deprecated
        
        # Timeout settings
        self.experiment_timeout = 3600 * 3  # 75 minutes per experiment (increased to allow reaching post-processing)
    
    def timeout_handler(self, signum, frame):
        """Handle timeout signal"""
        raise TimeoutError("Experiment exceeded timeout limit")
        
    def run_single_experiment(self, params_and_seed):
        """Run a single experiment with given share value and seed"""
        share, experiment_id, seed = params_and_seed
        
        # Set timeout protection
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.experiment_timeout)
        
        # Log experiment start
        start_time = time.time()
        print(f"Starting: share={share:.2f}, exp={experiment_id}, seed={seed}", flush=True)
        
        # Set random seed for reproducibility (both NumPy and Numba RNGs)
        np.random.seed(seed)
        seed_numba(seed)
        
        # Get cost function defaults if parameters not specified
        from collaborative_growth import get_cost_function_defaults
        defaults = get_cost_function_defaults(self.cost_type)
        
        c0 = self.c0 if self.c0 is not None else defaults['c0']
        c1 = self.c1 if self.c1 is not None else defaults['c1']
        c2 = self.c2 if self.c2 is not None else defaults['c2']
        
        # Model parameters (14-parameter format with cost function support)
        params = [self.markets, self.firms_per_market, self.steps, share, 
                 self.total_firms, self.merge_thresh, self.comparison, 
                 self.break_thresh, self.proportional, self.lookback, 
                 self.cost_type, c0, c1, c2]
        
        # DEBUG: Print parameters being passed to model
        print(f"DEBUG PARAMS: share={share}, cost_type={self.cost_type}, c0={c0}, c1={c1}, c2={c2}")
        try:
            # Run the model with growth process, sharing rule, and correlation params
            res = collaborative_growth.model(
                params=params, seed=seed, market_corr=self.market_corr,
                growth_process=self.growth_process,
                mu_range=self.mu_range,
                sigma_range=self.sigma_range,
                sharing_rule=self.sharing_rule,
                rho=self.rho,
                cross_corr=self.cross_corr
            )
            # Updated for new model output (13 elements: online rank stats replace full array)
            mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, \
            gini_coefficient, avg_ranks, mergers_per_period, proposals_per_period, \
            exits_per_period, rank_range, rank_std, hyperparameters = res

            # Extract only essential data for aggregation (much smaller memory footprint)
            results = {
                'alpha': share,
                'rep': experiment_id,
                'seed': seed,  # crc32 seed for reproducibility

                # Store hyperparameters BEFORE they get deleted
                'hyperparameters': hyperparameters.copy() if hyperparameters else None,
                'gini_coefficient': gini_coefficient.mean(axis=0),  # Average over markets

                # Time series data (keep these for averaging)
                'mean_members': mean_members,
                'num_cong': num_cong,

                # Pre-computed quantiles (already computed by model, average across markets)
                # quantiles_shares shape: (7_quantiles, markets, steps)
                'market_share_quantiles': quantiles_shares.mean(axis=1),  # Average over markets -> (7_quantiles, steps)
                'gini_quantiles': np.quantile(gini_coefficient, q=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1], axis=0).T,

                # Size-market share relationship analysis
                'temporal_poly_estimates': self._calculate_temporal_polynomial_estimates(avg_shares),
                'panel_poly_estimates': self._calculate_panel_polynomial_estimates(avg_shares),

                # Merger and exit frequency per period
                'mergers_per_period': mergers_per_period,
                'proposals_per_period': proposals_per_period,
                'exits_per_period': exits_per_period,

                # Online rank mobility statistics (shape: markets, firms_per_market)
                'rank_range': rank_range,  # max - min rank per firm
                'rank_std': rank_std,      # std dev of rank per firm
            }

            # Explicitly delete large arrays to free memory immediately
            del res, mean_members, quantiles_members, avg_shares, quantiles_shares, proposals_per_period
            del gini_coefficient, avg_ranks, mergers_per_period, exits_per_period, rank_range, rank_std, hyperparameters
            
            # Log successful completion
            runtime = time.time() - start_time
            print(f"Completed: share={share:.2f}, exp={experiment_id}, runtime={runtime:.1f}s", flush=True)
            
            return results
            
        except TimeoutError:
            runtime = time.time() - start_time
            print(f"TIMEOUT: share={share:.2f}, exp={experiment_id}, seed={seed}, runtime={runtime:.1f}s", flush=True)
            return None
        except Exception as e:
            runtime = time.time() - start_time
            print(f"ERROR: share={share:.2f}, exp={experiment_id}, seed={seed}, runtime={runtime:.1f}s, error={str(e)}", flush=True)
            return None
        finally:
            # Always cancel the timeout alarm
            signal.alarm(0)
    
    def run_single_experiment_to_disk(self, params_and_seed):
        """Run single experiment and save result directly to disk.

        Each task is (α, rep, seed) and produces one pickle file.
        """
        import pickle
        import os

        alpha, rep, seed = params_and_seed

        # Run the experiment
        result = self.run_single_experiment(params_and_seed)

        if result is not None:
            # Save result directly to disk - one pickle per (α, rep)
            filename = f'{self.results_dir}/alpha_{alpha:.2f}_rep_{rep:03d}.pkl'
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(result, f)
                print(f"Saved: alpha={alpha:.2f}, rep={rep}, seed={seed} to {filename}", flush=True)

                # Immediate memory cleanup after saving
                del result
                import gc
                gc.collect()

                return (alpha, rep, seed)  # Return full tuple for tracking
            except Exception as e:
                print(f"Failed to save alpha={alpha:.2f}, rep={rep}: {e}", flush=True)
                return None
        else:
            print(f"Experiment failed: alpha={alpha:.2f}, rep={rep}", flush=True)
            return None
    
    def _calculate_temporal_polynomial_estimates(self, avg_shares):
        """
        Calculate polynomial estimates every 10th timestep.
        This gives us temporal evolution of the size-market share relationship.
        """
        # Initialize estimates array with FULL steps size, like original (first ramp entries stay NaN)
        estimates = np.full([self.steps, 3], np.nan)

        successful_fits = 0
        empty_steps = 0
        insufficient_points = 0
        failed_fits = 0

        for step in range(self.ramp, self.steps, 10):  # Every 10th step
            try:
                # Check if we have data for this step
                if len(avg_shares[step]) == 0:
                    empty_steps += 1
                    continue
                    
                # Need at least 3 points for quadratic fit
                if avg_shares[step].shape[0] < 3:
                    insufficient_points += 1
                    continue
                
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')  # Ignore all warnings from polyfit
                    # Convert to float64 for polyfit (linalg doesn't support float128)
                    x_data = np.array(avg_shares[step][:, 0], dtype=np.float64)
                    y_data = np.array(avg_shares[step][:, 1], dtype=np.float64)
                    coeffs = np.polyfit(x_data, y_data, 2)
                # Store at index (step - ramp) but array is now full size
                estimates[step - self.ramp, :] = coeffs
                successful_fits += 1
            except Exception as e:
                # Add debugging info
                print(f"Warning: Polynomial fit failed at step {step}: {e}")
                failed_fits += 1
                continue
        
        return estimates

    def _calculate_panel_polynomial_estimates(self, avg_shares):
        """
        Calculate panel (pooled across timesteps) polynomial estimates.
        This pools all conglomerates across all timesteps within a single realization.
        Returns a single set of coefficients [β₀, β₁, β₂] for this realization.
        """
        # Pool all data across timesteps
        all_sizes = []
        all_market_shares = []

        for step in range(self.ramp, self.steps):
            if len(avg_shares[step]) > 0:
                all_sizes.extend(avg_shares[step][:, 0])
                all_market_shares.extend(avg_shares[step][:, 1])

        # Need at least 3 points for quadratic fit
        if len(all_sizes) < 3:
            return np.array([np.nan, np.nan, np.nan])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                coeffs = np.polyfit(all_sizes, all_market_shares, 2)
            return coeffs
        except Exception as e:
            print(f"Warning: Panel polynomial fit failed: {e}")
            return np.array([np.nan, np.nan, np.nan])

    def _aggregate_poly_estimates(self, results):
        """Aggregate polynomial estimates with debugging"""
        if not results:
            print("AGG_DEBUG: No results to aggregate!")
            return np.full([self.steps, 3], np.nan)
            
        share_value = results[0]['share']
        poly_arrays = [r['poly_estimates'] for r in results]
        
        # Check how many experiments have valid data
        valid_experiments = 0
        total_valid_coeffs = 0
        zero_coeff_experiments = []
        
        for i, poly_est in enumerate(poly_arrays):
            valid_coeffs = np.sum(~np.isnan(poly_est[:, 0]))  # Count non-NaN rows
            total_valid_coeffs += valid_coeffs
            if valid_coeffs > 0:
                valid_experiments += 1
            else:
                zero_coeff_experiments.append(i)
        
        print(f"AGG_DEBUG: share={share_value}, total_exp={len(poly_arrays)}, valid_exp={valid_experiments}, avg_coeffs={total_valid_coeffs / len(poly_arrays):.1f}, zero_exp={len(zero_coeff_experiments)}")
        
        # Perform the aggregation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = np.nanmean(poly_arrays, axis=0)
            
            if w:
                print(f"AGG_WARNING: {len(w)} warnings during aggregation for share={share_value}")
        
        final_valid = np.sum(~np.isnan(result[:, 0]))
        print(f"AGG_RESULT: share={share_value}, final_valid_coeffs={final_valid}")

        return result

    def run_share_experiments(self, share_value, chunk_start=None, chunk_end=None):
        """Run all experiments for a specific share value"""
        print(f"Running experiments for share = {share_value:.2f}")
        
        # Create experiment parameters
        experiments = list(range(self.counterfactuals))
        if chunk_start is not None and chunk_end is not None:
            experiments = experiments[chunk_start:chunk_end]
            print(f"Processing experiments {chunk_start}-{chunk_end} for share {share_value}")
        
        total_experiments = len(experiments)
        print(f"COUNTER: Starting {total_experiments} experiments for share {share_value:.2f}")
        
        # Generate seeds for reproducibility - COMMON RANDOM NUMBERS across alpha
        # Seed depends only on (scenario_name, exp_id), NOT on share_value
        # This enables paired comparisons: same exp_id sees same shocks at all alpha
        # Use zlib.crc32 for deterministic hashing (Python's hash() is salted per-process)
        scenario_name = getattr(self, 'scenario_name', None) or 'default'
        experiment_params = []
        for exp_id in experiments:
            seed = zlib.crc32(f"{scenario_name}:{exp_id}".encode()) & 0xFFFFFFFF
            experiment_params.append((share_value, exp_id, seed))
        
        # Run experiments in parallel with disk-based results
        # Force worker cleanup after each experiment to prevent memory accumulation
        log_memory_usage(f"BEFORE_POOL_{share_value:.2f}")
        
        with mp.Pool(processes=self.n_cores, maxtasksperchild=1) as pool:
            experiment_ids = list(tqdm(
                pool.imap(self.run_single_experiment_to_disk, experiment_params),
                total=len(experiment_params),
                desc=f"Share {share_value:.2f}"
            ))
        
        log_memory_usage(f"AFTER_POOL_{share_value:.2f}")
        
        # Load results from disk sequentially with immediate memory cleanup
        log_memory_usage(f"BEFORE_LOADING_{share_value:.2f}")
        successful_results = []
        failed_experiments = 0
        
        for exp_id in experiment_ids:
            if exp_id is not None:
                filename = f'{self.results_dir}/share_{share_value:.2f}_exp_{exp_id}.pkl'
                try:
                    with open(filename, 'rb') as f:
                        result = pickle.load(f)
                        successful_results.append(result)
                    
                    # Immediate memory cleanup after each experiment
                    import gc
                    gc.collect()
                    
                except Exception as e:
                    print(f"Failed to load experiment {exp_id}: {e}")
                    failed_experiments += 1
            else:
                failed_experiments += 1
        
        log_memory_usage(f"AFTER_LOADING_{share_value:.2f}")
        
        print(f"COUNTER: Share {share_value:.2f} completed: {len(successful_results)}/{total_experiments} successful, {failed_experiments} failed")
        
        # Clean up individual experiment files after loading (optional - saves disk space)
        for exp_id in experiment_ids:
            if exp_id is not None:
                filename = f'{self.results_dir}/share_{share_value:.2f}_exp_{exp_id}.pkl'
                try:
                    os.remove(filename)
                except:
                    pass  # Ignore cleanup errors
        
        return successful_results
    
    def aggregate_share_results(self, results):
        """Aggregate results across experiments for a single share value"""
        if not results:
            return None
            
        share_value = results[0]['share']
        n_experiments = len(results)
        
        # Get hyperparameters from first successful experiment (they should be the same for all)
        # Use the actual hyperparameters from the model results instead of overriding them
        if results and 'hyperparameters' in results[0]:
            hyperparameters = results[0]['hyperparameters'].copy()
            print(f"DEBUG AGG: Using hyperparameters from model results: cost_type={hyperparameters.get('cost_type')}")
        else:
            # Fallback to constructed hyperparameters if model results don't have them
            hyperparameters = {
                'markets': self.markets,
                'firms_per_market': self.firms_per_market,
                'steps': self.steps,
                'merge_thresh': self.merge_thresh,
                'comparison': self.comparison,
                'break_thresh': self.break_thresh,
                'proportional': self.proportional,
                'lookback': self.lookback,
                'cost_type': self.cost_type,
                'c0': self.c0,
                'c1': self.c1,
                'c2': self.c2,
                'scenario_name': getattr(self, 'scenario_name', None)
            }
            print(f"DEBUG AGG: Using fallback hyperparameters: cost_type={hyperparameters.get('cost_type')}")
        
        # Initialize aggregated arrays
        aggregated = {
            'share': share_value,
            'n_experiments': n_experiments,
            'hyperparameters': hyperparameters,

            # Preserve individual results for paired analysis (§5)
            'individual_results': results,

            # Time series averages
            'mean_members_avg': np.mean([r['mean_members'] for r in results], axis=0),
            'num_cong_avg': np.mean([r['num_cong'] for r in results], axis=0),
            'gini_avg': np.mean([r['gini_coefficient'] for r in results], axis=0),

            # Quantile averages
            'market_share_quantiles_avg': np.mean([r['market_share_quantiles'] for r in results], axis=0),
            'gini_quantiles_avg': np.mean([r['gini_quantiles'] for r in results], axis=0),

            # Size-market share relationship (temporal and panel regression coefficients)
            'temporal_poly_estimates_avg': np.nanmean([r['temporal_poly_estimates'] for r in results], axis=0),
            'temporal_poly_estimates_std': np.nanstd([r['temporal_poly_estimates'] for r in results], axis=0),
            'panel_poly_estimates_all': np.array([r['panel_poly_estimates'] for r in results]),  # shape: (n_realizations, 3)

            # Merger and exit frequency analysis
            'mergers_per_period_avg': np.mean([r['mergers_per_period'] for r in results], axis=0),
            'mergers_per_period_std': np.std([r['mergers_per_period'] for r in results], axis=0),
            'mergers_per_period_all': np.array([r['mergers_per_period'] for r in results]),

            # Proposals frequency analysis
            'proposals_per_period_avg': np.mean([r['proposals_per_period'] for r in results], axis=0),
            'proposals_per_period_std': np.std([r['proposals_per_period'] for r in results], axis=0),
            'proposals_per_period_all': np.array([r['proposals_per_period'] for r in results]),

            'exits_per_period_avg': np.mean([r['exits_per_period'] for r in results], axis=0),
            'exits_per_period_std': np.std([r['exits_per_period'] for r in results], axis=0),
            'exits_per_period_all': np.array([r['exits_per_period'] for r in results])
        }
        
        return aggregated

    def save_results(self, aggregated_results, filename_prefix='counterfactual_results'):
        """Save aggregated results"""
        # Save as pickle for complete data
        with open(f'{filename_prefix}.pkl', 'wb') as f:
            pickle.dump(aggregated_results, f)
        
        print(f"Results saved to {filename_prefix}.pkl")
        
        return aggregated_results

def main():
    """Main function for parallel counterfactual analysis"""
    import time
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Run parallel counterfactual analysis')
    parser.add_argument('--share', type=float, default=None,
                       help='Specific share value to process (if None, process all)')
    parser.add_argument('--alpha_list', type=str, default=None,
                       help='Comma-separated list of alpha values (e.g., "0.0,0.1")')
    parser.add_argument('--chunk', type=str, default=None,
                       help='Experiment chunk to process (format: start_end, e.g., 0_25)')
    parser.add_argument('--n_cores', type=int, default=64,
                       help='Number of CPU cores to use (default: 64)')
    parser.add_argument('--counterfactuals', type=int, default=50,
                       help='Number of replications per share value (default: 50)')
    
    # Hyperparameter arguments
    parser.add_argument('--markets', type=int, default=100,
                       help='Number of markets (default: 100)')
    parser.add_argument('--firms_per_market', type=int, default=100,
                       help='Number of firms per market (default: 100)')
    parser.add_argument('--steps', type=int, default=10000,
                       help='Number of simulation steps (default: 10000)')
    parser.add_argument('--merge_thresh', type=float, default=0.05,
                       help='Merger threshold (default: 0.05)')
    parser.add_argument('--comparison', type=int, default=4,
                       help='Comparison parameter (default: 4)')
    parser.add_argument('--break_thresh', type=float, default=0.85,
                       help='Breakup threshold (default: 0.85)')
    parser.add_argument('--lookback', type=int, default=50,
                       help='Lookback period for exit decisions (default: 50)')
    parser.add_argument('--proportional', action='store_true',
                       help='Use proportional sharing (default: False)')
    parser.add_argument('--market_corr', type=str, default='identity',
                       choices=['identity', 'random'],
                       help='Market correlation type (default: identity = uncorrelated)')
    # Growth process parameters (Phase B §1)
    parser.add_argument('--growth_process', type=str, default='normal_net',
                       choices=['normal_net', 'lognormal'],
                       help='Growth process type (default: normal_net = Phase A)')
    parser.add_argument('--mu_range', type=float, nargs=2, default=[0.01, 0.1],
                       metavar=('LO', 'HI'),
                       help='Mean growth rate bounds (default: 0.01 0.1)')
    parser.add_argument('--sigma_range', type=float, nargs=2, default=[0.01, 0.05],
                       metavar=('LO', 'HI'),
                       help='Volatility bounds (default: 0.01 0.05)')
    # §2 Sharing rule
    parser.add_argument('--sharing_rule', type=str, default='equal',
                       choices=['equal', 'proportional'],
                       help='Sharing rule: equal (Phase A default), proportional (capital-weighted)')
    # §3 Correlation structure
    parser.add_argument('--rho', type=float, default=0.0,
                       help='Within-market correlation coefficient (default: 0.0)')
    parser.add_argument('--cross_corr', type=float, default=0.0,
                       help='Cross-market correlation coefficient (default: 0.0)')
    parser.add_argument('--cost_type', type=str, default='power_law',
                       choices=['linear', 'quadratic', 'exponential', 'power_law'],
                       help='Management cost function type (default: power_law)')
    parser.add_argument('--c0', type=float, default=None,
                       help='Base cost parameter c0 (default: cost-function-specific)')
    parser.add_argument('--c1', type=float, default=None,
                       help='Scaling cost parameter c1 (default: cost-function-specific)')
    parser.add_argument('--c2', type=float, default=None,
                       help='Quadratic cost parameter c2 (default: cost-function-specific)')
    # Backward compatibility
    parser.add_argument('--b0', type=float, default=0.00001,
                       help='DEPRECATED: Use --c0 instead. Power law parameter b0 (default: 0.00001)')
    parser.add_argument('--b1', type=float, default=1.2,
                       help='DEPRECATED: Use --c1 instead. Power law parameter b1 (default: 1.2)')
    
    # Robustness analysis scenario naming
    parser.add_argument('--scenario_name', type=str, default=None,
                       help='Name for robustness analysis scenario')

    args = parser.parse_args()
    
    # DEBUG: Print actual parsed arguments
    print(f"DEBUG ARGS: cost_type={args.cost_type}, c0={args.c0}, c1={args.c1}, c2={args.c2}")
    print(f"DEBUG ARGS: Full command line args: {args}")
    
    # Parse chunk parameters
    chunk_start, chunk_end = None, None
    if args.chunk:
        try:
            chunk_start, chunk_end = map(int, args.chunk.split('_'))
        except ValueError:
            print("Error: --chunk must be in format start_end (e.g., 0_25)")
            return
    
    # Initialize runner with command-line hyperparameters
    runner = ParallelCounterfactualRunner(n_cores=args.n_cores, 
                                        counterfactuals=args.counterfactuals)
    
    # Override default hyperparameters with command-line arguments
    runner.markets = args.markets
    runner.firms_per_market = args.firms_per_market
    runner.total_firms = runner.markets * runner.firms_per_market
    runner.steps = args.steps
    runner.merge_thresh = args.merge_thresh
    runner.comparison = args.comparison
    runner.break_thresh = args.break_thresh
    runner.lookback = args.lookback
    runner.proportional = args.proportional
    runner.market_corr = args.market_corr

    # Growth process parameters (Phase B §1)
    runner.growth_process = args.growth_process
    runner.mu_range = tuple(args.mu_range)
    runner.sigma_range = tuple(args.sigma_range)

    # Sharing rule parameters
    runner.sharing_rule = args.sharing_rule

    # Correlation structure parameters
    runner.rho = args.rho
    runner.cross_corr = args.cross_corr

    # Cost function parameters
    runner.cost_type = args.cost_type
    runner.c0 = args.c0
    runner.c1 = args.c1  
    runner.c2 = args.c2
    
    # Backward compatibility
    runner.b0 = args.b0
    runner.b1 = args.b1
    
    # Robustness scenario naming
    runner.scenario_name = args.scenario_name

    # Create results directory (organized by scenario if provided)
    results_dir = 'results'
    if args.scenario_name:
        results_dir = f'results/{args.scenario_name}'
    os.makedirs(results_dir, exist_ok=True)

    # Set results directory in runner
    runner.results_dir = results_dir

    # Determine which α values to process
    if args.alpha_list is not None:
        alpha_values = [float(a) for a in args.alpha_list.split(',')]
    elif args.share is not None:
        alpha_values = [args.share]
    else:
        alpha_values = runner.shares

    # Generate all (α, rep, seed) tasks upfront
    scenario_name = args.scenario_name or 'default'
    all_tasks = []
    for alpha in alpha_values:
        for rep in range(runner.counterfactuals):
            # crc32 seed: depends on (scenario, rep), NOT on alpha (common random numbers)
            seed = zlib.crc32(f"{scenario_name}:{rep}".encode()) & 0xFFFFFFFF
            all_tasks.append((alpha, rep, seed))

    n_tasks = len(all_tasks)
    print(f"\n=== Task Overview ===")
    print(f"α values: {len(alpha_values)} ({alpha_values[0]:.2f} to {alpha_values[-1]:.2f})")
    print(f"Replications: {runner.counterfactuals}")
    print(f"Total tasks: {n_tasks}")
    print(f"Output directory: {results_dir}")
    print()

    # Run all tasks in one pool
    log_memory_usage("BEFORE_POOL")

    with mp.Pool(processes=runner.n_cores, maxtasksperchild=1) as pool:
        completed = list(tqdm(
            pool.imap(runner.run_single_experiment_to_disk, all_tasks),
            total=n_tasks,
            desc="Running experiments"
        ))

    log_memory_usage("AFTER_POOL")

    # Summary
    successful = [t for t in completed if t is not None]
    failed = n_tasks - len(successful)
    print(f"\n=== Summary ===")
    print(f"Completed: {len(successful)}/{n_tasks}")
    if failed > 0:
        print(f"Failed: {failed}")
    
    # Calculate and display runtime
    end_time = time.time()
    runtime = end_time - start_time
    hours = int(runtime // 3600)
    minutes = int((runtime % 3600) // 60)
    seconds = runtime % 60
    
    print(f"Counterfactual analysis complete!")
    print(f"Total runtime: {hours:02d}h {minutes:02d}m {seconds:05.2f}s ({runtime:.2f} seconds)")

if __name__ == "__main__":
    main()