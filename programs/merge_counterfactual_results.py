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
        
    def merge_chunks_for_share(self, share_value):
        """Merge all chunks for a specific share value"""
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
        
        merged = {
            'share': share_value,
            'n_experiments': total_experiments,
            'mean_members_avg': np.average([chunk['mean_members_avg'] for chunk in aggregated_chunks], 
                                         weights=weights, axis=0),
            'num_cong_avg': np.average([chunk['num_cong_avg'] for chunk in aggregated_chunks], 
                                     weights=weights, axis=0),
            'gini_coefficient_avg': np.average([chunk['gini_coefficient_avg'] for chunk in aggregated_chunks], 
                                             weights=weights, axis=0),
            'market_share_quantiles_avg': np.average([chunk['market_share_quantiles_avg'] for chunk in aggregated_chunks], 
                                                    weights=weights, axis=0),
            'gini_quantiles_avg': np.average([chunk['gini_quantiles_avg'] for chunk in aggregated_chunks], 
                                           weights=weights, axis=0),
            'poly_estimates_avg': np.average([chunk['poly_estimates_avg'] for chunk in aggregated_chunks], 
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
        merged_results = {}
        
        for share in self.shares:
            result = self.merge_chunks_for_share(share)
            if result:
                merged_results[share] = result
            else:
                print(f"Warning: No valid results for share {share:.2f}")
        
        return merged_results
    
    def save_merged_results(self, merged_results, filename='counterfactual_results_final.pkl'):
        """Save the final merged results"""
        output_path = self.results_dir / filename
        
        with open(output_path, 'wb') as f:
            pickle.dump(merged_results, f)
        
        print(f"Final merged results saved to {output_path}")
        
        # Also save in format compatible with original counterfactuals.py
        self._save_original_format(merged_results)
        
        return merged_results
    
    def _save_original_format(self, merged_results):
        """Save in format matching original counterfactuals.py output"""
        if not merged_results:
            return
            
        n_shares = len(merged_results)
        
        # Get dimensions from first result
        first_result = next(iter(merged_results.values()))
        steps = len(first_result['mean_members_avg'])
        
        # Initialize arrays like original script
        mean_quantiles = np.empty(shape=(n_shares, 4, steps + 1))
        mean_members_ = np.empty(shape=(n_shares, steps))
        mean_conglomerates = np.empty(shape=(n_shares, steps))
        mean_gini = np.empty(shape=(n_shares, 5, steps + 1))
        mean_ests = np.empty(shape=(n_shares, steps, 3))
        
        # Fill arrays
        for i, (share, result) in enumerate(sorted(merged_results.items())):
            mean_quantiles[i, :, :] = result['market_share_quantiles_avg']
            mean_members_[i, :] = result['mean_members_avg']
            mean_conglomerates[i, :] = result['num_cong_avg']
            mean_gini[i, :, :] = result['gini_quantiles_avg'].T
            mean_ests[i, :, :] = result['poly_estimates_avg']
        
        # Save arrays
        output_dir = self.results_dir / 'original_format'
        output_dir.mkdir(exist_ok=True)
        
        np.save(output_dir / 'mean_quantiles.npy', mean_quantiles)
        np.save(output_dir / 'mean_members_.npy', mean_members_)
        np.save(output_dir / 'mean_conglomerates.npy', mean_conglomerates)
        np.save(output_dir / 'mean_gini.npy', mean_gini)
        np.save(output_dir / 'mean_ests.npy', mean_ests)
        
        # Save share values
        np.save(output_dir / 'shares.npy', np.array(list(merged_results.keys())))
        
        print(f"Results saved in original format to {output_dir}/")

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