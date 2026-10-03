#!/usr/bin/env python3
"""
Merge and aggregate counterfactual results from parallel computation
"""

import os
import pickle
import numpy as np
from pathlib import Path
import argparse

class CounterfactualMerger:
    def __init__(self, results_dir='counterfactual_results'):
        self.results_dir = Path(results_dir)
        self.shares = np.arange(0, 0.52, 0.02)

        # Detect merge mode: baseline (per-share chunks) vs robustness (all-shares chunks)
        self.merge_mode = self._detect_merge_mode()

    def _detect_merge_mode(self):
        """Detect whether we have per-share chunks or all-shares chunks"""
        # Check for all_shares_chunk_*.pkl files (robustness mode)
        all_shares_chunks = list(self.results_dir.glob('all_shares_chunk_*.pkl'))
        if all_shares_chunks:
            return 'all_shares'

        # Check for share_*_chunk_*.pkl files (baseline mode)
        share_chunks = list(self.results_dir.glob('share_*_chunk_*.pkl'))
        if share_chunks:
            return 'per_share'

        # Default to per_share mode
        return 'per_share'

    def merge_all_shares_chunks(self):
        """Merge all_shares_chunk_*.pkl files (robustness mode)"""
        print("Detected robustness mode: merging all_shares_chunk_*.pkl files")

        chunk_files = sorted(list(self.results_dir.glob('all_shares_chunk_*.pkl')))

        if not chunk_files:
            print("  No all_shares_chunk_*.pkl files found")
            return None

        print(f"Found {len(chunk_files)} chunk files to merge")

        # Load all chunks
        all_chunk_data = []
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                    if chunk_data:
                        all_chunk_data.append(chunk_data)
                print(f"  Loaded {chunk_file.name}")
            except Exception as e:
                print(f"  Error loading {chunk_file.name}: {e}")

        if not all_chunk_data:
            return None

        # Merge chunks - each chunk is a dict with share values as keys
        # We need to merge results for each share across chunks
        merged_results = {}

        for share in self.shares:
            # Collect data for this share from all chunks
            share_chunks = []
            for chunk_data in all_chunk_data:
                if share in chunk_data:
                    share_chunks.append(chunk_data[share])

            if share_chunks:
                # Merge chunks for this share
                merged_results[share] = self._merge_aggregated_chunks(share_chunks, share)
                print(f"  Merged {len(share_chunks)} chunks for share {share:.2f}")

        return merged_results

    def merge_chunks_for_share(self, share_value):
        """Merge all chunks for a specific share value (baseline mode)"""
        print(f"Merging chunks for share {share_value:.2f}")

        all_experiments = []
        chunk_files = list(self.results_dir.glob(f'share_{share_value:.2f}_chunk_*.pkl'))

        if not chunk_files:
            print(f"  No chunk files found for share {share_value:.2f}")
            return None

        for chunk_file in sorted(chunk_files):
            try:
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                    if chunk_data:  # Make sure it's not None
                        all_experiments.append(chunk_data)
                print(f"  Loaded {chunk_file.name}")
            except Exception as e:
                print(f"  Error loading {chunk_file.name}: {e}")

        if not all_experiments:
            return None

        # Aggregate across all chunks for this share
        return self._aggregate_experiments(all_experiments, share_value)
    
    def _aggregate_experiments(self, experiment_chunks, share_value):
        """Aggregate experiment results across chunks"""
        # Extract hyperparameters from first experiment (they should be the same across all)
        hyperparameters = None
        
        # Flatten all experiments from all chunks
        all_experiments = []
        for chunk in experiment_chunks:
            if isinstance(chunk, dict) and 'n_experiments' in chunk:
                # This is already aggregated chunk data - we need the raw experiments
                # For now, just use the aggregated data
                all_experiments.append(chunk)
            elif isinstance(chunk, list):
                # Raw experiment list
                all_experiments.extend(chunk)
        
        if not all_experiments:
            return None
        
        # If we have aggregated chunks, we need to re-aggregate
        # For simplicity, let's assume we're working with pre-aggregated chunks
        if isinstance(all_experiments[0], dict) and 'mean_members_avg' in all_experiments[0]:
            return self._merge_aggregated_chunks(all_experiments, share_value)
        else:
            return self._aggregate_raw_experiments(all_experiments, share_value)
    
    def _merge_aggregated_chunks(self, aggregated_chunks, share_value):
        """Merge pre-aggregated chunk results"""
        total_experiments = sum(chunk['n_experiments'] for chunk in aggregated_chunks)
        
        # Weight averages by number of experiments
        weights = [chunk['n_experiments'] / total_experiments for chunk in aggregated_chunks]
        
        # Extract hyperparameters from first chunk (should be same for all)
        hyperparameters = aggregated_chunks[0].get('hyperparameters', {})
        
        merged = {
            'share': share_value,
            'n_experiments': total_experiments,
            'hyperparameters': hyperparameters,
            'mean_members_avg': np.average([chunk['mean_members_avg'] for chunk in aggregated_chunks], 
                                         weights=weights, axis=0),
            'num_cong_avg': np.average([chunk['num_cong_avg'] for chunk in aggregated_chunks], 
                                     weights=weights, axis=0),
            'market_share_quantiles_avg': np.average([chunk['market_share_quantiles_avg'] for chunk in aggregated_chunks], 
                                                    weights=weights, axis=0),
            'gini_quantiles_avg': np.average([chunk['gini_quantiles_avg'] for chunk in aggregated_chunks],
                                           weights=weights, axis=0),

            # Size-market share relationship (temporal and panel)
            'temporal_poly_estimates_avg': np.average([chunk['temporal_poly_estimates_avg'] for chunk in aggregated_chunks],
                                                     weights=weights, axis=0),
            'temporal_poly_estimates_std': np.average([chunk['temporal_poly_estimates_std'] for chunk in aggregated_chunks],
                                                     weights=weights, axis=0),
        }
        
        # For rank mobility, we need to concatenate arrays
        all_rank_ranges = []
        all_rank_std = []
        for chunk in aggregated_chunks:
            if 'rank_ranges' in chunk:
                all_rank_ranges.append(chunk['rank_ranges'])
            if 'rank_std' in chunk:
                all_rank_std.append(chunk['rank_std'])
        
        if all_rank_ranges:
            merged['rank_ranges'] = np.concatenate(all_rank_ranges, axis=0)
        if all_rank_std:
            merged['rank_std'] = np.concatenate(all_rank_std, axis=0)
        
        # Panel polynomial estimates - concatenate across chunks
        if 'panel_poly_estimates_all' in aggregated_chunks[0]:
            all_panel_estimates = []
            for chunk in aggregated_chunks:
                if 'panel_poly_estimates_all' in chunk:
                    all_panel_estimates.append(chunk['panel_poly_estimates_all'])

            if all_panel_estimates:
                merged['panel_poly_estimates_all'] = np.concatenate(all_panel_estimates, axis=0)

        # Merge merger frequency data
        if 'mergers_per_period_avg' in aggregated_chunks[0]:
            merged['mergers_per_period_avg'] = np.average([chunk['mergers_per_period_avg'] for chunk in aggregated_chunks],
                                                        weights=weights, axis=0)
            merged['mergers_per_period_std'] = np.average([chunk['mergers_per_period_std'] for chunk in aggregated_chunks],
                                                        weights=weights, axis=0)

            # Concatenate all individual merger data
            all_merger_data = []
            for chunk in aggregated_chunks:
                if 'mergers_per_period_all' in chunk:
                    all_merger_data.append(chunk['mergers_per_period_all'])

            if all_merger_data:
                merged['mergers_per_period_all'] = np.concatenate(all_merger_data, axis=0)

        # Merge exit frequency data (same structure as mergers)
        if 'exits_per_period_avg' in aggregated_chunks[0]:
            merged['exits_per_period_avg'] = np.average([chunk['exits_per_period_avg'] for chunk in aggregated_chunks],
                                                       weights=weights, axis=0)
            merged['exits_per_period_std'] = np.average([chunk['exits_per_period_std'] for chunk in aggregated_chunks],
                                                       weights=weights, axis=0)

            # Concatenate all individual exit data
            all_exit_data = []
            for chunk in aggregated_chunks:
                if 'exits_per_period_all' in chunk:
                    all_exit_data.append(chunk['exits_per_period_all'])

            if all_exit_data:
                merged['exits_per_period_all'] = np.concatenate(all_exit_data, axis=0)
        
        # Merge cluster data
        if 'merger_clusters_all' in aggregated_chunks[0]:
            all_cluster_data = []
            all_cluster_counts = []
            
            for chunk in aggregated_chunks:
                if 'merger_clusters_all' in chunk:
                    all_cluster_data.extend(chunk['merger_clusters_all'])
                    all_cluster_counts.extend(chunk['n_clusters_per_experiment'])
            
            if all_cluster_data:
                merged['merger_clusters_all'] = all_cluster_data
                merged['n_clusters_per_experiment'] = all_cluster_counts
                merged['avg_clusters_per_experiment'] = np.mean(all_cluster_counts)
        
        return merged
    
    def _aggregate_raw_experiments(self, experiments, share_value):
        """Aggregate raw experiment results"""
        # This would be more complex - for now, assume we have aggregated chunks
        raise NotImplementedError("Raw experiment aggregation not implemented yet")
    
    def merge_all_shares(self):
        """Merge results for all share values"""
        print(f"Merge mode: {self.merge_mode}")

        if self.merge_mode == 'all_shares':
            # Robustness mode: merge all_shares_chunk_*.pkl files
            return self.merge_all_shares_chunks()
        else:
            # Baseline mode: merge per-share chunk files
            merged_results = {}

            for share in self.shares:
                result = self.merge_chunks_for_share(share)
                if result:
                    merged_results[share] = result
                else:
                    print(f"Warning: No valid results for share {share:.2f}")

            return merged_results
    
    def save_merged_results(self, merged_results, filename='counterfactual_results_final.pkl'):
        """Save the final merged results in a parameterized directory"""
        
        # Extract hyperparameters from first result to create directory name
        if not merged_results:
            print("No merged results to save!")
            return merged_results
        
        first_result = next(iter(merged_results.values()))
        hyperparams = first_result.get('hyperparameters', {})
        
        # Build directory name from hyperparameters (matching run_parallel_counterfactuals.py format)
        dir_name = self._build_param_directory_name(hyperparams)
        
        # Create parameterized results directory
        param_results_dir = Path('results') / dir_name
        param_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save merged results in the parameterized directory
        output_path = param_results_dir / filename
        with open(output_path, 'wb') as f:
            pickle.dump(merged_results, f)
        
        print(f"Final merged results saved to {output_path}")
        
        return merged_results
    
    def _build_param_directory_name(self, hyperparams):
        """Build directory name from hyperparameters matching the hyperparam_str format"""

        # Check if this is a robustness scenario by looking at results_dir path
        # If results_dir is robustness_results/scenario_name, extract scenario_name
        results_dir_str = str(self.results_dir)
        if 'robustness_results' in results_dir_str:
            # Extract scenario name from path like 'robustness_results/exponential_baseline'
            scenario_name = self.results_dir.name
            return f"robustness_{scenario_name}"

        # Check if scenario_name exists in hyperparameters (legacy support)
        scenario_name = hyperparams.get('scenario_name', None)
        if scenario_name:
            # For robustness scenarios, use the scenario name as directory
            return f"robustness_{scenario_name}"
        
        # Extract parameters with defaults (original behavior)
        markets = hyperparams.get('markets', 100)
        firms_per_market = hyperparams.get('firms_per_market', 100)
        steps = hyperparams.get('steps', 10000)
        merge_thresh = hyperparams.get('merge_thresh', 0.05)
        comparison = hyperparams.get('comparison', 4)
        break_thresh = hyperparams.get('break_thresh', 0.85)
        lookback = hyperparams.get('lookback', 50)
        
        # Handle both old and new cost parameter formats
        if 'cost_type' in hyperparams:
            cost_type = hyperparams.get('cost_type', 'power_law')
            c0 = hyperparams.get('c0', 0.00001)
            c1 = hyperparams.get('c1', 1.2)
            c2 = hyperparams.get('c2', 0.001)
            cost_str = f"cost_type_{cost_type}_c0_{c0}_c1_{c1}_c2_{c2}"
        else:
            # Backward compatibility - old b0, b1 format
            b0 = hyperparams.get('b0', 0.00001)
            b1 = hyperparams.get('b1', 1.2)
            cost_str = f"cost_type_power_law_c0_{b0}_c1_{b1}_c2_0.001"
        
        proportional = hyperparams.get('proportional', False)
        proportional_str = "_proportional" if proportional else ""
        
        # Build directory name matching hyperparam_str format
        dir_name = f"markets_{markets}_firms_per_market_{firms_per_market}_steps_{steps}_merge_thresh_{merge_thresh}_comparison_{comparison}_break_thresh_{break_thresh}_lookback_{lookback}_{cost_str}{proportional_str}"
        
        return dir_name
    

def main():
    parser = argparse.ArgumentParser(description='Merge counterfactual results')
    parser.add_argument('--results_dir', type=str, default='counterfactual_results',
                       help='Directory containing chunk results')
    parser.add_argument('--output', type=str, default='counterfactual_results_final.pkl',
                       help='Output filename for merged results')
    
    args = parser.parse_args()
    
    merger = CounterfactualMerger(args.results_dir)
    
    print("Merging counterfactual results...")
    merged_results = merger.merge_all_shares()
    
    if merged_results:
        merger.save_merged_results(merged_results, args.output)
        print(f"Successfully merged results for {len(merged_results)} share values")
    else:
        print("No results to merge!")

if __name__ == "__main__":
    main()
