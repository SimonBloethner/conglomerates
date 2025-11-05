import numpy as np
import multiprocessing as mp
from itertools import product
import pickle
import json
import time
from tqdm import tqdm
import collaborative_growth
from functools import partial
import gc

class HyperparameterValidator:
    def __init__(self, n_cores=64, n_replications=50):
        """
        Initialize hyperparameter validation framework
        
        Args:
            n_cores: Number of CPU cores to use
            n_replications: Number of replications per parameter set
        """
        self.n_cores = n_cores
        self.n_replications = n_replications
        self.results = {}
        
    def define_parameter_grid(self, chunk_start=None, chunk_end=None):
        """Define the hyperparameter grid to search over"""
        parameter_grid = {
            'b0': [0.000001, 0.000005, 0.00001, 0.00005],  # Power law base (gives 0.0005%-3.15% costs)
            'b1': [1.05, 1.1, 1.2, 1.4],  # Power law exponent (mildly convex: b1 > 1)
            'merge_thresh': [0.01, 0.03, 0.05, 0.07, 0.1],  # Merging probability
            'markets': [50, 100, 150, 200],  # Number of markets
            'firms_per_market': [50, 100, 150],  # Market size
            'share': [0.3, 0.5, 0.7, 0.9],  # Pooling rate
        }
        
        # Generate all combinations
        keys = list(parameter_grid.keys())
        values = list(parameter_grid.values())
        combinations = list(product(*values))
        
        parameter_sets = [dict(zip(keys, combo)) for combo in combinations]
        
        # Randomize parameter order to distribute computational load evenly
        np.random.shuffle(parameter_sets)
        
        # Apply chunking if specified
        if chunk_start is not None and chunk_end is not None:
            parameter_sets = parameter_sets[chunk_start:chunk_end]
            print(f"Processing chunk {chunk_start}:{chunk_end}")
        
        print(f"Generated {len(parameter_sets)} parameter combinations")
        print(f"Total simulations: {len(parameter_sets) * self.n_replications}")
        
        return parameter_sets
    
    def run_single_replication(self, params_and_seed):
        """Run a single model replication with given parameters"""
        params, seed = params_and_seed
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Extract parameters
        b0 = params['b0']
        b1 = params['b1'] 
        merge_thresh = params['merge_thresh']
        markets = params['markets']
        firms_per_market = params['firms_per_market']
        share = params['share']
        
        # Fixed parameters (can be made configurable)
        steps = 1000
        total_firms = markets * firms_per_market
        comparison = 2
        break_thresh = 0.85
        lookback = 50
        proportional = False
        
        model_params = [markets, firms_per_market, steps, share, total_firms, 
                       merge_thresh, comparison, break_thresh, proportional, lookback, b0, b1]
        
        try:
            # Run the model
            results = collaborative_growth.model(model_params)
            
            # Extract key metrics
            mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, \
            max_shares, market_share, hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks = results
            
            # Calculate paper-relevant summary statistics
            summary_stats = {
                # Market share distribution metrics
                'final_gini_mean': float(np.mean(gini_coefficient[:, -1])),
                'final_gini_std': float(np.std(gini_coefficient[:, -1])),
                'max_market_share_mean': float(np.mean(max_shares[:, -1])),
                'max_market_share_std': float(np.std(max_shares[:, -1])),
                'q90_market_share': float(np.mean(quantiles_shares[1, :, -1])),  # 90th percentile
                'q99_market_share': float(np.mean(quantiles_shares[4, :, -1])),  # 99th percentile
                
                # Conglomerate formation metrics
                'mean_conglomerate_size': float(np.mean(mean_members[-100:])),  # Last 100 periods
                'final_conglomerate_size': float(mean_members[-1]),
                'num_conglomerates_final': int(num_cong[-1]),
                'conglomerate_stability': self._calculate_stability(mean_members),
                
                # Mobility metrics
                'rank_mobility_range': self._calculate_rank_mobility(ranks),
                'rank_volatility': self._calculate_rank_volatility(ranks),
                
                # Temporal dynamics
                'gini_convergence_time': self._calculate_convergence_time(gini_coefficient),
                'share_inequality_trend': self._calculate_inequality_trend(quantiles_shares),
                
                'parameters': params,
                'seed': seed
            }
            
            # Clean up large arrays to prevent memory accumulation in multiprocessing
            del results, mean_members, quantiles_members, avg_shares, quantiles_shares
            del max_shares, market_share, hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks
            gc.collect()
            
            return summary_stats
            
        except Exception as e:
            print(f"Error in replication with params {params}, seed {seed}: {str(e)}")
            return None
    
    def _calculate_convergence_time(self, metric_array, threshold=0.01):
        """Calculate time to convergence based on metric stability"""
        try:
            # Calculate rolling standard deviation across markets
            window = 50
            rolling_std = np.array([np.std(metric_array[:, i:i+window]) for i in range(metric_array.shape[1] - window)])
            
            # Find first time it goes below threshold and stays there
            stable_periods = rolling_std < threshold
            if np.any(stable_periods):
                return int(np.argmax(stable_periods))
            else:
                return metric_array.shape[1]  # Never converged
        except Exception:
            return metric_array.shape[1]  # Default to full length
    
    def _calculate_stability(self, time_series):
        """Calculate stability of conglomerate formation"""
        try:
            # Calculate coefficient of variation over last 100 periods
            last_periods = time_series[-100:] if len(time_series) >= 100 else time_series
            return float(np.std(last_periods) / np.mean(last_periods)) if np.mean(last_periods) > 0 else 1.0
        except Exception:
            return 1.0
    
    def _calculate_rank_mobility(self, ranks):
        """Calculate rank mobility (range of ranks per firm)"""
        try:
            # ranks shape should be (steps, markets, firms_per_market)
            if len(ranks.shape) != 3:
                return 0.0
            
            steps, markets, firms_per_market = ranks.shape
            rank_ranges = []
            
            for market in range(markets):
                for firm in range(firms_per_market):
                    firm_ranks = ranks[:, market, firm]  # All time periods for this firm
                    rank_ranges.append(np.max(firm_ranks) - np.min(firm_ranks))
            
            return float(np.mean(rank_ranges))
        except Exception as e:
            print(f"ERROR in _calculate_rank_mobility: {str(e)}")
            return 0.0
    
    def _calculate_rank_volatility(self, ranks):
        """Calculate rank volatility (standard deviation of ranks per firm)"""
        try:
            # ranks shape should be (steps, markets, firms_per_market)
            if len(ranks.shape) != 3:
                return 0.0
                
            steps, markets, firms_per_market = ranks.shape
            rank_volatilities = []
            
            for market in range(markets):
                for firm in range(firms_per_market):
                    firm_ranks = ranks[:, market, firm]  # All time periods for this firm
                    rank_volatilities.append(np.std(firm_ranks))
            
            return float(np.mean(rank_volatilities))
        except Exception as e:
            print(f"ERROR in _calculate_rank_volatility: {str(e)}")
            return 0.0
    
    def _calculate_inequality_trend(self, quantiles_shares):
        """Calculate trend in inequality over time"""
        try:
            # Use difference between 99th and 10th percentile as inequality measure
            inequality_over_time = quantiles_shares[4, :, :] - quantiles_shares[0, :, :]  # 99th - 10th percentile
            mean_inequality = np.mean(inequality_over_time, axis=0)  # Average across markets
            
            # Calculate trend (slope of inequality over time)
            time_points = np.arange(len(mean_inequality))
            slope = np.polyfit(time_points, mean_inequality, 1)[0]
            return float(slope)
        except Exception:
            return 0.0
    
    def run_parameter_set(self, param_set):
        """Run all replications for a single parameter set"""
        print(f"Processing parameter set: {param_set}")
        
        # Create seeds for all replications
        base_seed = hash(str(param_set)) % 10000
        seeds = [base_seed + i for i in range(self.n_replications)]
        
        # Create parameter-seed pairs
        param_seed_pairs = [(param_set, seed) for seed in seeds]
        
        # Run replications sequentially for this parameter set
        replication_results = []
        for param_seed_pair in param_seed_pairs:
            result = self.run_single_replication(param_seed_pair)
            if result is not None:
                replication_results.append(result)
        
        # Aggregate results across replications
        if replication_results:
            aggregated = self._aggregate_replications(replication_results, param_set)
            return aggregated
        else:
            return None
    
    def _aggregate_replications(self, replication_results, param_set):
        """Aggregate statistics across replications"""
        metrics = ['final_gini_mean', 'final_gini_std', 'max_market_share_mean', 'max_market_share_std',
                  'q90_market_share', 'q99_market_share', 'mean_conglomerate_size', 'final_conglomerate_size',
                  'num_conglomerates_final', 'conglomerate_stability', 'rank_mobility_range',
                  'rank_volatility', 'gini_convergence_time', 'share_inequality_trend']
        
        aggregated = {'parameters': param_set, 'n_replications': len(replication_results)}
        
        for metric in metrics:
            values = [result[metric] for result in replication_results if metric in result]
            if values:
                aggregated[f'{metric}_mean'] = float(np.mean(values))
                aggregated[f'{metric}_std'] = float(np.std(values))
                aggregated[f'{metric}_ci_lower'] = float(np.percentile(values, 2.5))
                aggregated[f'{metric}_ci_upper'] = float(np.percentile(values, 97.5))
        
        return aggregated
    
    def run_validation(self, parameter_sets=None, save_results=True, filename_prefix='hyperparameter_results'):
        """Run the full hyperparameter validation"""
        if parameter_sets is None:
            parameter_sets = self.define_parameter_grid()
        
        print(f"Starting hyperparameter validation with {self.n_cores} cores")
        print(f"Processing {len(parameter_sets)} parameter combinations")
        print(f"Total individual simulations: {len(parameter_sets) * self.n_replications}")
        
        # Create all parameter-seed combinations for parallel processing
        all_param_seed_pairs = []
        for param_set in parameter_sets:
            base_seed = hash(str(param_set)) % 10000
            seeds = [base_seed + i for i in range(self.n_replications)]
            for seed in seeds:
                all_param_seed_pairs.append((param_set, seed))
        
        start_time = time.time()
        
        # Parallelize at replication level for maximum efficiency
        with mp.Pool(processes=self.n_cores) as pool:
            replication_results = list(tqdm(
                pool.imap(self.run_single_replication, all_param_seed_pairs),
                total=len(all_param_seed_pairs),
                desc="Individual simulations"
            ))
        
        # Filter out None results
        replication_results = [r for r in replication_results if r is not None]
        
        # Group results by parameter set
        results_by_params = {}
        for result in replication_results:
            param_key = str(result['parameters'])
            if param_key not in results_by_params:
                results_by_params[param_key] = []
            results_by_params[param_key].append(result)
        
        # Aggregate results for each parameter set
        aggregated_results = []
        for param_key, param_results in results_by_params.items():
            if param_results:
                # Extract parameter set from first result
                param_set = param_results[0]['parameters']
                aggregated = self._aggregate_replications(param_results, param_set)
                aggregated_results.append(aggregated)
        
        end_time = time.time()
        
        print(f"Validation completed in {end_time - start_time:.2f} seconds")
        print(f"Successfully processed {len(aggregated_results)}/{len(parameter_sets)} parameter sets")
        print(f"Average time per simulation: {(end_time - start_time) / len(replication_results):.2f} seconds")
        
        self.results = aggregated_results
        
        if save_results:
            # Save as pickle for complete data
            with open(f'{filename_prefix}.pkl', 'wb') as f:
                pickle.dump(aggregated_results, f)
            
            # Save as JSON for easy reading
            with open(f'{filename_prefix}.json', 'w') as f:
                json.dump(aggregated_results, f, indent=2)
            
            print(f"Results saved to {filename_prefix}.pkl and {filename_prefix}.json")
        
        return aggregated_results

def main():
    """Main function to run hyperparameter validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter validation')
    parser.add_argument('--chunk', type=str, default=None, 
                       help='Parameter chunk to process (format: start_end, e.g., 0_960)')
    parser.add_argument('--n_cores', type=int, default=64, 
                       help='Number of CPU cores to use (default: 64)')
    parser.add_argument('--n_replications', type=int, default=50, 
                       help='Number of replications per parameter set (default: 50)')
    
    args = parser.parse_args()
    
    # Parse chunk parameters
    chunk_start, chunk_end = None, None
    if args.chunk:
        try:
            chunk_start, chunk_end = map(int, args.chunk.split('_'))
        except ValueError:
            print("Error: --chunk must be in format start_end (e.g., 0_960)")
            return
    
    # Initialize validator
    validator = HyperparameterValidator(n_cores=args.n_cores, n_replications=args.n_replications)
    
    # Get parameter sets (with chunking if specified)
    parameter_sets = validator.define_parameter_grid(chunk_start, chunk_end)
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    # Set output filename based on chunk
    if args.chunk:
        filename_prefix = f'results/hyperparameter_results_chunk_{args.chunk}'
    else:
        filename_prefix = 'results/hyperparameter_results'
    
    # Run validation
    results = validator.run_validation(parameter_sets, filename_prefix=filename_prefix)
    
    print("Validation complete!")
    
    # Basic analysis
    if results:
        print(f"Lowest inequality parameters: {min(results, key=lambda x: x.get('final_gini_mean_mean', float('inf')))['parameters']}")
        print(f"Highest mobility parameters: {max(results, key=lambda x: x.get('rank_mobility_range_mean', 0))['parameters']}")


if __name__ == "__main__":
    main()