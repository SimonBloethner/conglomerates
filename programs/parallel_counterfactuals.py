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
from sklearn.cluster import DBSCAN
import signal
import time

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
        
        # Share values to test
        self.shares = np.arange(0, 0.52, 0.02)
        
        # Power law parameters (use optimal values from hyperparameter validation)
        self.b0 = 0.00001  # Adjust based on your optimal parameters
        self.b1 = 1.2      # Adjust based on your optimal parameters
        
        # Timeout settings
        self.experiment_timeout = 4000  # ~67 minutes per experiment
    
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
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Model parameters (including power law parameters from hyperparameter validation)
        params = [self.markets, self.firms_per_market, self.steps, share, 
                 self.total_firms, self.merge_thresh, self.comparison, 
                 self.break_thresh, self.proportional, self.lookback, 
                 self.b0, self.b1]
        
        try:
            # Run the model
            res = collaborative_growth.model(params=params)
            mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, \
            max_shares, market_share, hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks, mergers_per_period = res
            
            # Extract only essential data for aggregation (much smaller memory footprint)
            results = {
                'share': share,
                'experiment_id': experiment_id,
                
                # Time series data (keep these for averaging)
                'mean_members': mean_members,
                'num_cong': num_cong,
                'gini_coefficient': gini_coefficient,
                
                # Pre-computed quantiles (smaller than raw data)
                'market_share_quantiles': np.quantile(market_share, q=[0.5, 0.9, 0.99, 1], axis=2).mean(axis=1),
                'gini_quantiles': np.quantile(gini_coefficient, q=[0.1, 0.25, 0.5, 0.75, 0.9], axis=0).T,
                
                # Polynomial estimates (computed like in counterfactuals.py)
                'poly_estimates': self._calculate_polynomial_estimates(avg_shares, share),
                
                # Mobility metrics (pre-computed to avoid storing full ranks)
                'rank_ranges': self._calculate_single_run_mobility(ranks),
                'rank_std': self._calculate_single_run_std(ranks),
                
                # Merger frequency per period
                'mergers_per_period': mergers_per_period,
                
                # DBSCAN merger clusters
                'merger_clusters': self.detect_merger_clusters_dbscan(mergers_per_period)
            }
            
            # Explicitly delete large arrays to free memory immediately
            del res, mean_members, quantiles_members, avg_shares, quantiles_shares
            del max_shares, market_share, hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks, mergers_per_period
            
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
    
    def _calculate_polynomial_estimates(self, avg_shares, share_value):
        """Calculate polynomial estimates exactly like in counterfactuals.py"""
        # Initialize estimates array with FULL steps size, like original (first ramp entries stay NaN)
        estimates = np.full([self.steps, 3], np.nan)
        
        successful_fits = 0
        empty_steps = 0
        insufficient_points = 0
        failed_fits = 0
        
        for step in range(self.ramp, self.steps):
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
        
        # Always show polynomial estimation summary  
        print(f"POLY_DEBUG: share={share_value}, successful_fits={successful_fits}, empty_steps={empty_steps}, insufficient_points={insufficient_points}, failed_fits={failed_fits}")
                
        return estimates
    
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
    
    def _calculate_single_run_mobility(self, ranks):
        """Calculate mobility metrics for a single run (to avoid storing full ranks)"""
        # ranks shape: (steps, markets, firms)
        min_ranks = np.min(ranks, axis=0)  # min over time for each firm
        max_ranks = np.max(ranks, axis=0)  # max over time for each firm
        return max_ranks - min_ranks  # range for each firm
    
    def _calculate_single_run_std(self, ranks):
        """Calculate rank standard deviation for a single run"""
        return np.std(ranks, axis=0)  # std over time for each firm
    
    def detect_merger_clusters_dbscan(self, merger_time_series, exclude_early=50, eps=10, min_samples=3):
        """Use DBSCAN to identify merger wave clusters"""
        if np.sum(merger_time_series) == 0:  # No mergers
            return []
        
        # Exclude early periods to avoid initialization noise
        if len(merger_time_series) <= exclude_early:
            return []
        
        clean_series = merger_time_series[exclude_early:]
        
        if np.sum(clean_series) == 0:
            return []
        
        # Find periods with above-median activity
        threshold = np.median(clean_series[clean_series > 0]) if np.any(clean_series > 0) else 0
        
        if threshold <= 0:
            return []
        
        # Create feature matrix: [time_index, merger_intensity] for active periods
        active_periods = []
        period_indices = []
        
        for t, activity in enumerate(clean_series):
            if activity > threshold:
                # Scale time and intensity for clustering
                scaled_time = (t + exclude_early) / len(merger_time_series)  # Normalize time to [0,1]
                scaled_intensity = activity / np.max(clean_series)  # Normalize intensity to [0,1]
                active_periods.append([scaled_time, scaled_intensity])
                period_indices.append(t + exclude_early)
        
        if len(active_periods) < min_samples:
            return []
        
        # Apply DBSCAN clustering
        active_periods = np.array(active_periods)
        
        # Scale eps appropriately for normalized features
        normalized_eps = eps / len(merger_time_series)  # eps in terms of fraction of total time
        
        dbscan = DBSCAN(eps=normalized_eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(active_periods)
        
        # Convert clusters back to merger wave format
        clusters = []
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise points (not part of any cluster)
                continue
            
            # Get periods belonging to this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_periods = [period_indices[i] for i in range(len(period_indices)) if cluster_mask[i]]
            cluster_intensities = [merger_time_series[p] for p in cluster_periods]
            
            if len(cluster_periods) >= min_samples:
                peak_idx = np.argmax(cluster_intensities)
                peak_period = cluster_periods[peak_idx]
                
                clusters.append({
                    'cluster_id': cluster_id,
                    'periods': sorted(cluster_periods),
                    'start_period': min(cluster_periods),
                    'end_period': max(cluster_periods),
                    'peak_period': peak_period,
                    'peak_intensity': merger_time_series[peak_period],
                    'total_mergers': sum(cluster_intensities),
                    'duration': max(cluster_periods) - min(cluster_periods) + 1,
                    'n_active_periods': len(cluster_periods),
                    'avg_intensity': np.mean(cluster_intensities),
                    'threshold_used': threshold
                })
        
        return clusters
    
    def run_share_experiments(self, share_value, chunk_start=None, chunk_end=None):
        """Run all experiments for a specific share value"""
        print(f"Running experiments for share = {share_value:.2f}")
        
        # Create experiment parameters
        experiments = list(range(self.counterfactuals))
        if chunk_start is not None and chunk_end is not None:
            experiments = experiments[chunk_start:chunk_end]
            print(f"Processing experiments {chunk_start}-{chunk_end} for share {share_value}")
        
        # Generate seeds for reproducibility
        base_seed = int(share_value * 10000) % 10000
        experiment_params = []
        for exp_id in experiments:
            seed = base_seed + exp_id * 1000
            experiment_params.append((share_value, exp_id, seed))
        
        # Run experiments in parallel
        with mp.Pool(processes=self.n_cores) as pool:
            results = list(tqdm(
                pool.imap(self.run_single_experiment, experiment_params),
                total=len(experiment_params),
                desc=f"Share {share_value:.2f}"
            ))
        
        # Filter out failed experiments
        results = [r for r in results if r is not None]
        
        return results
    
    def aggregate_share_results(self, results):
        """Aggregate results across experiments for a single share value"""
        if not results:
            return None
            
        share_value = results[0]['share']
        n_experiments = len(results)
        
        # Initialize aggregated arrays
        aggregated = {
            'share': share_value,
            'n_experiments': n_experiments,
            
            # Time series averages
            'mean_members_avg': np.mean([r['mean_members'] for r in results], axis=0),
            'num_cong_avg': np.mean([r['num_cong'] for r in results], axis=0),
            'gini_coefficient_avg': np.mean([r['gini_coefficient'] for r in results], axis=0),
            
            # Quantile averages
            'market_share_quantiles_avg': np.mean([r['market_share_quantiles'] for r in results], axis=0),
            'gini_quantiles_avg': np.mean([r['gini_quantiles'] for r in results], axis=0),
            
            # Polynomial estimates (with debugging)
            'poly_estimates_avg': self._aggregate_poly_estimates(results),
            
            # Mobility analysis (concatenate pre-computed metrics)
            'rank_ranges': np.array([r['rank_ranges'] for r in results]),
            'rank_std': np.array([r['rank_std'] for r in results]),
            
            # Merger frequency analysis
            'mergers_per_period_avg': np.mean([r['mergers_per_period'] for r in results], axis=0),
            'mergers_per_period_std': np.std([r['mergers_per_period'] for r in results], axis=0),
            'mergers_per_period_all': np.array([r['mergers_per_period'] for r in results]),
            
            # Merger cluster analysis
            'merger_clusters_all': [r['merger_clusters'] for r in results],
            'n_clusters_per_experiment': [len(r['merger_clusters']) for r in results],
            'avg_clusters_per_experiment': np.mean([len(r['merger_clusters']) for r in results])
        }
        
        return aggregated
    
    def _calculate_rank_mobility(self, ranks_list):
        """Calculate rank mobility metrics"""
        # Stack all experiments
        ranks_array = np.array(ranks_list)  # shape: (experiments, steps, markets, firms)
        
        # Calculate min/max ranks over time for each firm
        min_ranks = np.min(ranks_array, axis=1)  # shape: (experiments, markets, firms)
        max_ranks = np.max(ranks_array, axis=1)  # shape: (experiments, markets, firms)
        
        # Range of ranks (mobility measure)
        rank_ranges = max_ranks - min_ranks
        
        return rank_ranges
    
    def _calculate_rank_std(self, ranks_list):
        """Calculate standard deviation of ranks"""
        ranks_array = np.array(ranks_list)
        return np.std(ranks_array, axis=1)  # std over time for each experiment
    
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
    parser.add_argument('--chunk', type=str, default=None,
                       help='Experiment chunk to process (format: start_end, e.g., 0_25)')
    parser.add_argument('--n_cores', type=int, default=64,
                       help='Number of CPU cores to use (default: 64)')
    parser.add_argument('--counterfactuals', type=int, default=50,
                       help='Number of replications per share value (default: 50)')
    
    args = parser.parse_args()
    
    # Parse chunk parameters
    chunk_start, chunk_end = None, None
    if args.chunk:
        try:
            chunk_start, chunk_end = map(int, args.chunk.split('_'))
        except ValueError:
            print("Error: --chunk must be in format start_end (e.g., 0_25)")
            return
    
    # Initialize runner
    runner = ParallelCounterfactualRunner(n_cores=args.n_cores, 
                                        counterfactuals=args.counterfactuals)
    
    # Create results directory
    os.makedirs('counterfactual_results', exist_ok=True)
    
    if args.share is not None:
        # Process single share value
        results = runner.run_share_experiments(args.share, chunk_start, chunk_end)
        aggregated = runner.aggregate_share_results(results)
        
        # Save with share-specific filename
        filename = f'counterfactual_results/share_{args.share:.2f}'
        if args.chunk:
            filename += f'_chunk_{args.chunk}'
        
        runner.save_results(aggregated, filename)
        
    else:
        # Process all share values
        all_results = {}
        
        for share in runner.shares:
            print(f"\nProcessing share = {share:.2f}")
            results = runner.run_share_experiments(share, chunk_start, chunk_end)
            aggregated = runner.aggregate_share_results(results)
            all_results[share] = aggregated
        
        # Save combined results
        filename = 'counterfactual_results/all_shares'
        if args.chunk:
            filename += f'_chunk_{args.chunk}'
            
        runner.save_results(all_results, filename)
    
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