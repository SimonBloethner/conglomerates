#!/usr/bin/env python3
"""
Merge hyperparameter validation results from multiple chunk files
"""

import pickle
import json
import glob
from pathlib import Path
import argparse

def merge_chunk_results(results_dir='results', output_prefix='hyperparameter_results_merged'):
    """Merge all chunk files in results directory"""
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: {results_dir} directory not found")
        return False
    
    # Find all chunk files
    chunk_files = list(results_path.glob('hyperparameter_results_chunk_*.pkl'))
    
    if not chunk_files:
        print(f"No chunk files found in {results_dir}")
        return False
    
    print(f"Found {len(chunk_files)} chunk files:")
    for file in sorted(chunk_files):
        print(f"  - {file.name}")
    
    # Merge all results
    all_results = []
    
    for chunk_file in sorted(chunk_files):
        print(f"Loading {chunk_file.name}...")
        try:
            with open(chunk_file, 'rb') as f:
                chunk_results = pickle.load(f)
                all_results.extend(chunk_results)
                print(f"  Added {len(chunk_results)} parameter sets")
        except Exception as e:
            print(f"  Error loading {chunk_file}: {e}")
    
    if not all_results:
        print("No results to merge")
        return False
    
    print(f"\nTotal merged results: {len(all_results)} parameter sets")
    
    # Save merged results
    merged_pkl = results_path / f'{output_prefix}.pkl'
    merged_json = results_path / f'{output_prefix}.json'
    
    print(f"Saving merged results to {merged_pkl}")
    with open(merged_pkl, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"Saving merged results to {merged_json}")
    with open(merged_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Basic statistics
    print(f"\nMerged Results Summary:")
    print(f"  Total parameter combinations: {len(all_results)}")
    
    if all_results:
        # Count replications
        total_sims = sum(result.get('n_replications', 0) for result in all_results)
        print(f"  Total simulations: {total_sims}")
        
        # Find best/worst results for key metrics
        metrics_of_interest = [
            ('final_gini_mean_mean', 'Final Gini (lower=better)'),
            ('rank_mobility_range_mean', 'Rank Mobility (higher=better)'),
            ('mean_conglomerate_size_mean', 'Mean Conglomerate Size')
        ]
        
        for metric_key, metric_name in metrics_of_interest:
            if metric_key in all_results[0]:
                values = [r.get(metric_key, 0) for r in all_results]
                print(f"  {metric_name}: min={min(values):.3f}, max={max(values):.3f}, mean={sum(values)/len(values):.3f}")
    
    print(f"\nMerging complete! Files saved to {results_path}/")
    return True

def main():
    parser = argparse.ArgumentParser(description='Merge hyperparameter validation chunk results')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing chunk result files (default: results)')
    parser.add_argument('--output', type=str, default='hyperparameter_results_merged',
                       help='Output filename prefix (default: hyperparameter_results_merged)')
    
    args = parser.parse_args()
    
    success = merge_chunk_results(args.results_dir, args.output)
    
    if success:
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("1. Run analysis: python3 analyze_hyperparameter_results.py")
        print("2. Check figures/ directory for all visualizations")
        print("3. Use optimal parameters from figures/optimal_parameters.csv")
        print("="*60)
    
if __name__ == "__main__":
    main()