#!/usr/bin/env python3
"""
Comprehensive analysis and visualization of hyperparameter validation results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import pickle
import json
from pathlib import Path
import argparse
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Set style for academic plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HyperparameterAnalyzer:
    def __init__(self, results_file=None):
        """Initialize analyzer with results data"""
        if results_file is None:
            results_file = 'results/hyperparameter_results_merged.pkl'
        
        self.results_file = Path(results_file)
        self.results = []
        self.df = None
        
        # Ensure figures directory exists
        self.figures_dir = Path('figures')
        self.figures_dir.mkdir(exist_ok=True)
        
    def load_results(self):
        """Load and process results data"""
        print(f"Loading results from {self.results_file}")
        
        if not self.results_file.exists():
            print(f"Error: {self.results_file} not found")
            print("Make sure to merge results first using run_parallel_validation.py")
            return False
        
        with open(self.results_file, 'rb') as f:
            self.results = pickle.load(f)
        
        print(f"Loaded {len(self.results)} parameter combinations")
        
        # Convert to DataFrame for easier analysis
        self._create_dataframe()
        return True
    
    def _create_dataframe(self):
        """Convert results to pandas DataFrame"""
        rows = []
        for result in self.results:
            row = result['parameters'].copy()
            
            # Add all metrics with their means, stds, and confidence intervals
            for key, value in result.items():
                if key not in ['parameters', 'n_replications']:
                    row[key] = value
            
            rows.append(row)
        
        self.df = pd.DataFrame(rows)
        print(f"Created DataFrame with {len(self.df)} rows and {len(self.df.columns)} columns")
        
        # Print column names for reference
        print("Available metrics:")
        metric_cols = [col for col in self.df.columns if '_mean' in col]
        for col in sorted(metric_cols):
            print(f"  - {col}")
    
    def create_parameter_heatmaps(self):
        """Create heatmaps showing parameter effects on key metrics"""
        print("Creating parameter effect heatmaps...")
        
        key_metrics = [
            ('final_gini_mean_mean', 'Final Gini Coefficient'),
            ('rank_mobility_range_mean', 'Rank Mobility Range'), 
            ('mean_conglomerate_size_mean', 'Mean Conglomerate Size'),
            ('num_conglomerates_final_mean', 'Number of Conglomerates')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (metric, title) in enumerate(key_metrics):
            # Create pivot table for heatmap
            # Focus on most important parameter pairs
            pivot = self.df.pivot_table(values=metric, 
                                      index='b0', 
                                      columns='b1', 
                                      aggfunc='mean')
            
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                       ax=axes[idx], cbar_kws={'label': title})
            axes[idx].set_title(f'{title} vs Power Law Parameters')
            axes[idx].set_xlabel('Power Law Parameter b1')
            axes[idx].set_ylabel('Power Law Parameter b0')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'parameter_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / 'parameter_heatmaps.eps', format='eps', bbox_inches='tight')
        plt.show()
    
    def create_pooling_rate_analysis(self):
        """Analyze effects of different pooling rates (share parameter)"""
        print("Analyzing pooling rate effects...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Gini coefficient vs pooling rate
        self._plot_metric_vs_share(axes[0,0], 'final_gini_mean_mean', 'Final Gini Coefficient')
        
        # 2. Mobility vs pooling rate  
        self._plot_metric_vs_share(axes[0,1], 'rank_mobility_range_mean', 'Rank Mobility Range')
        
        # 3. Conglomerate size vs pooling rate
        self._plot_metric_vs_share(axes[1,0], 'mean_conglomerate_size_mean', 'Mean Conglomerate Size')
        
        # 4. Number of conglomerates vs pooling rate
        self._plot_metric_vs_share(axes[1,1], 'num_conglomerates_final_mean', 'Number of Conglomerates')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'pooling_rate_effects.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / 'pooling_rate_effects.eps', format='eps', bbox_inches='tight')
        plt.show()
    
    def _plot_metric_vs_share(self, ax, metric, title):
        """Helper function to plot metric vs share parameter"""
        # Group by share and calculate statistics
        grouped = self.df.groupby('share')[metric].agg(['mean', 'std']).reset_index()
        
        ax.errorbar(grouped['share'], grouped['mean'], yerr=grouped['std'], 
                   marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
        ax.set_xlabel('Pooling Rate (α)')
        ax.set_ylabel(title)
        ax.set_title(f'{title} vs Pooling Rate')
        ax.grid(True, alpha=0.3)
    
    def create_market_structure_analysis(self):
        """Analyze effects of market structure parameters"""
        print("Analyzing market structure effects...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Markets vs key metrics
        self._plot_market_effects(axes[0,0], 'markets', 'final_gini_mean_mean', 
                                'Number of Markets', 'Final Gini Coefficient')
        
        self._plot_market_effects(axes[0,1], 'firms_per_market', 'final_gini_mean_mean',
                                'Firms per Market', 'Final Gini Coefficient')
        
        self._plot_market_effects(axes[1,0], 'markets', 'rank_mobility_range_mean',
                                'Number of Markets', 'Rank Mobility Range')
        
        self._plot_market_effects(axes[1,1], 'firms_per_market', 'rank_mobility_range_mean',
                                'Firms per Market', 'Rank Mobility Range')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'market_structure_effects.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / 'market_structure_effects.eps', format='eps', bbox_inches='tight')
        plt.show()
    
    def _plot_market_effects(self, ax, param_col, metric_col, param_title, metric_title):
        """Helper function to plot market parameter effects"""
        grouped = self.df.groupby(param_col)[metric_col].agg(['mean', 'std']).reset_index()
        
        ax.errorbar(grouped[param_col], grouped['mean'], yerr=grouped['std'],
                   marker='s', capsize=5, capthick=2, linewidth=2, markersize=8)
        ax.set_xlabel(param_title)
        ax.set_ylabel(metric_title)
        ax.set_title(f'{metric_title} vs {param_title}')
        ax.grid(True, alpha=0.3)
    
    def create_power_law_vs_logistic_comparison(self):
        """Compare power law cost function with logistic (your key research question)"""
        print("Comparing power law vs logistic cost functions...")
        
        # For this comparison, we need to identify which parameter sets represent
        # different cost function behaviors
        
        # Instead of scatter plots, create binned heatmaps that will show patterns clearly
        metrics = [
            ('final_gini_mean_mean', 'Final Gini Coefficient'),
            ('rank_mobility_range_mean', 'Rank Mobility Range'),
            ('mean_conglomerate_size_mean', 'Mean Conglomerate Size'),
            ('num_conglomerates_final_mean', 'Number of Conglomerates'),
            ('conglomerate_stability_mean', 'Conglomerate Stability'),
            ('gini_convergence_time_mean', 'Convergence Time')
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            # Create a pivot table for heatmap - this will show patterns clearly
            # Use all parameter combinations
            pivot_data = self.df.pivot_table(values=metric, 
                                           index='b1', 
                                           columns='b0', 
                                           aggfunc='mean')
            
            # Create heatmap
            im = sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', 
                           ax=ax, cbar_kws={'label': title})
            
            ax.set_title(title)
            ax.set_xlabel('Power Law Parameter b₀')
            ax.set_ylabel('Power Law Parameter b₁')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'power_law_parameter_effects.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / 'power_law_parameter_effects.eps', format='eps', bbox_inches='tight')
        plt.show()
    
    def analyze_parameter_importance_and_interactions(self):
        """Analyze which parameters matter and their interaction effects"""
        print("Analyzing parameter importance and interactions...")
        
        param_cols = ['b0', 'b1', 'share', 'merge_thresh', 'markets', 'firms_per_market']
        key_metrics = ['final_gini_mean_mean', 'rank_mobility_range_mean', 
                      'mean_conglomerate_size_mean', 'num_conglomerates_final_mean']
        
        # 1. Calculate effect sizes for each parameter
        self._calculate_main_effects(param_cols, key_metrics)
        
        # 2. Test conditional importance (interactions)
        self._test_conditional_importance(param_cols, key_metrics)
        
        # 3. Identify irrelevant parameters
        self._identify_irrelevant_parameters(param_cols, key_metrics)
    
    def _calculate_main_effects(self, param_cols, metrics):
        """Calculate main effect size for each parameter"""
        print("\n=== MAIN PARAMETER EFFECTS ===")
        
        results = []
        
        for metric in metrics:
            print(f"\nMetric: {metric.replace('_mean', '')}")
            print("-" * 50)
            
            for param in param_cols:
                # Calculate range of metric values across parameter range
                param_groups = self.df.groupby(param)[metric]
                min_val = param_groups.mean().min()
                max_val = param_groups.mean().max()
                effect_size = max_val - min_val
                
                # Calculate as percentage of total metric range
                total_range = self.df[metric].max() - self.df[metric].min()
                effect_pct = (effect_size / total_range) * 100
                
                results.append({
                    'metric': metric,
                    'parameter': param,
                    'effect_size': effect_size,
                    'effect_percentage': effect_pct
                })
                
                print(f"{param:15s}: {effect_size:.4f} ({effect_pct:.1f}% of total range)")
        
        # Create visualization
        results_df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            metric_data = results_df[results_df['metric'] == metric]
            
            bars = axes[idx].bar(metric_data['parameter'], metric_data['effect_percentage'])
            axes[idx].set_title(f'Parameter Effects on {metric.replace("_mean", "")}')
            axes[idx].set_ylabel('Effect Size (% of total range)')
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Color bars by effect size
            max_effect = metric_data['effect_percentage'].max()
            for bar, effect in zip(bars, metric_data['effect_percentage']):
                if effect < max_effect * 0.1:  # Less than 10% of max effect
                    bar.set_color('red')  # Irrelevant
                elif effect < max_effect * 0.3:
                    bar.set_color('orange')  # Weak
                else:
                    bar.set_color('green')  # Important
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'parameter_main_effects.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _test_conditional_importance(self, param_cols, metrics):
        """Test if parameter importance depends on other parameters"""
        print("\n=== CONDITIONAL PARAMETER IMPORTANCE ===")
        
        # Example: Does b0 effect depend on firms_per_market?
        key_interactions = [
            ('b0', 'firms_per_market'),
            ('b1', 'markets'),
            ('share', 'merge_thresh'),
            ('b0', 'b1')
        ]
        
        for metric in metrics[:2]:  # Test on key metrics
            print(f"\nMetric: {metric.replace('_mean', '')}")
            print("-" * 50)
            
            for param1, param2 in key_interactions:
                # Split param2 into low/high groups
                param2_median = self.df[param2].median()
                low_group = self.df[self.df[param2] <= param2_median]
                high_group = self.df[self.df[param2] > param2_median]
                
                # Calculate param1 effect in each group
                low_effect = self._calculate_parameter_effect(low_group, param1, metric)
                high_effect = self._calculate_parameter_effect(high_group, param1, metric)
                
                interaction_strength = abs(high_effect - low_effect)
                
                print(f"{param1} effect when {param2} is low:  {low_effect:.4f}")
                print(f"{param1} effect when {param2} is high: {high_effect:.4f}")
                print(f"Interaction strength: {interaction_strength:.4f}")
                
                if interaction_strength > 0.01:  # Threshold for meaningful interaction
                    print("  *** STRONG INTERACTION DETECTED ***")
                print()
    
    def _calculate_parameter_effect(self, data, parameter, metric):
        """Calculate effect size of parameter on metric in given data subset"""
        param_groups = data.groupby(parameter)[metric]
        return param_groups.mean().max() - param_groups.mean().min()
    
    def _identify_irrelevant_parameters(self, param_cols, metrics):
        """Identify parameters that have minimal impact across all metrics"""
        print("\n=== PARAMETER RELEVANCE SUMMARY ===")
        
        # Calculate average effect size across all metrics for each parameter
        param_importance = {}
        
        for param in param_cols:
            total_effect = 0
            for metric in metrics:
                effect = self._calculate_parameter_effect(self.df, param, metric)
                total_range = self.df[metric].max() - self.df[metric].min()
                normalized_effect = effect / total_range
                total_effect += normalized_effect
            
            avg_effect = total_effect / len(metrics)
            param_importance[param] = avg_effect
        
        # Sort by importance
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("Parameter ranking (average normalized effect across all metrics):")
        print("-" * 60)
        for param, importance in sorted_params:
            status = "CRITICAL" if importance > 0.3 else "MODERATE" if importance > 0.1 else "WEAK"
            print(f"{param:15s}: {importance:.4f} ({status})")
        
        # Identify candidates for removal
        irrelevant_threshold = 0.05
        irrelevant_params = [param for param, imp in param_importance.items() if imp < irrelevant_threshold]
        
        if irrelevant_params:
            print(f"\nParameters with minimal impact (< {irrelevant_threshold:.2f}):")
            for param in irrelevant_params:
                print(f"  - {param}")
            print("\nConsider removing these parameters to simplify the model.")
        else:
            print(f"\nAll parameters have meaningful impact (> {irrelevant_threshold:.2f})")
    
    def create_all_parameter_interaction_heatmaps(self):
        """Create heatmaps for all important parameter pairs"""
        print("Creating heatmaps for all parameter interactions...")
        
        # Define ALL possible parameter pairs (15 unique combinations)
        param_pairs = [
            ('b0', 'b1'),
            ('b0', 'share'),
            ('b0', 'merge_thresh'),
            ('b0', 'markets'),
            ('b0', 'firms_per_market'),
            ('b1', 'share'),
            ('b1', 'merge_thresh'),
            ('b1', 'markets'),
            ('b1', 'firms_per_market'),
            ('share', 'merge_thresh'),
            ('share', 'markets'),
            ('share', 'firms_per_market'),
            ('merge_thresh', 'markets'),
            ('merge_thresh', 'firms_per_market'),
            ('markets', 'firms_per_market')
        ]
        
        # Key metrics to analyze
        key_metrics = [
            ('final_gini_mean_mean', 'Final Gini Coefficient'),
            ('rank_mobility_range_mean', 'Rank Mobility Range'),
            ('mean_conglomerate_size_mean', 'Mean Conglomerate Size'),
            ('num_conglomerates_final_mean', 'Number of Conglomerates')
        ]
        
        # Create separate figure for each metric
        for metric_col, metric_title in key_metrics:
            print(f"Creating interaction heatmaps for {metric_title}...")
            
            # Calculate grid size for subplots
            n_pairs = len(param_pairs)
            n_cols = 3
            n_rows = (n_pairs + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
            
            # Handle case where we have only one row
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            axes = axes.flatten()
            
            for idx, (param1, param2) in enumerate(param_pairs):
                if idx >= len(axes):
                    break
                
                ax = axes[idx]
                
                # Create pivot table for heatmap
                try:
                    pivot_data = self.df.pivot_table(values=metric_col, 
                                                   index=param2, 
                                                   columns=param1, 
                                                   aggfunc='mean')
                    
                    # Create heatmap
                    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', 
                               ax=ax, cbar_kws={'label': metric_title})
                    
                    ax.set_title(f'{param1} vs {param2}')
                    ax.set_xlabel(param1)
                    ax.set_ylabel(param2)
                    
                except Exception as e:
                    # If pivot fails (e.g., not enough unique combinations), show error
                    ax.text(0.5, 0.5, f'No data for\n{param1} vs {param2}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{param1} vs {param2} (No Data)')
            
            # Hide unused subplots
            for idx in range(len(param_pairs), len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle(f'Parameter Interactions: {metric_title}', fontsize=16, y=0.98)
            plt.tight_layout()
            
            # Save with metric-specific filename
            filename = f'parameter_interactions_{metric_col.replace("_mean", "")}'
            plt.savefig(self.figures_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.figures_dir / f'{filename}.eps', format='eps', bbox_inches='tight')
            plt.show()
    
    def create_optimal_parameter_analysis(self):
        """Find and visualize optimal parameter combinations"""
        print("Finding optimal parameter combinations...")
        
        # Define optimization criteria (you can adjust these based on your paper's goals)
        criteria = {
            'Low Inequality': ('final_gini_mean_mean', 'minimize'),
            'High Mobility': ('rank_mobility_range_mean', 'maximize'), 
            'Stable Conglomerates': ('conglomerate_stability_mean', 'minimize'),
            'Fast Convergence': ('gini_convergence_time_mean', 'minimize')
        }
        
        results_summary = []
        
        for criterion_name, (metric, direction) in criteria.items():
            if direction == 'minimize':
                best_idx = self.df[metric].idxmin()
            else:
                best_idx = self.df[metric].idxmax()
            
            best_params = self.df.loc[best_idx]
            results_summary.append({
                'Criterion': criterion_name,
                'Metric Value': best_params[metric],
                'b0': best_params['b0'],
                'b1': best_params['b1'],
                'share': best_params['share'],
                'merge_thresh': best_params['merge_thresh'],
                'markets': best_params['markets'],
                'firms_per_market': best_params['firms_per_market']
            })
        
        # Create summary table
        summary_df = pd.DataFrame(results_summary)
        print("\nOptimal Parameter Combinations:")
        print("=" * 80)
        print(summary_df.to_string(index=False, float_format='%.3f'))
        
        # Save to file
        summary_df.to_csv(self.figures_dir / 'optimal_parameters.csv', index=False)
        
        return summary_df
    
    def create_parameter_correlation_analysis(self):
        """Analyze correlations between parameters and outcomes"""
        print("Analyzing parameter correlations...")
        
        # Select key columns for correlation analysis
        param_cols = ['b0', 'b1', 'share', 'merge_thresh', 'markets', 'firms_per_market']
        metric_cols = [col for col in self.df.columns if '_mean' in col and 'ci' not in col]
        
        # Create correlation matrix
        corr_data = self.df[param_cols + metric_cols]
        correlation_matrix = corr_data.corr()
        
        # Focus on correlations between parameters and metrics
        param_metric_corr = correlation_matrix.loc[param_cols, metric_cols]
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(param_metric_corr, annot=True, fmt='.3f', cmap='RdBu_r',
                   center=0, square=False)
        plt.title('Parameter-Metric Correlations')
        plt.xlabel('Metrics')
        plt.ylabel('Parameters')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(self.figures_dir / 'parameter_correlations.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / 'parameter_correlations.eps', format='eps', bbox_inches='tight')
        plt.show()
    
    def create_statistical_significance_tests(self):
        """Perform statistical significance tests between parameter groups"""
        print("Performing statistical significance tests...")
        
        # Test if power law parameters have significant effects
        sig_tests = []
        
        # Group by b0 values and test differences in key metrics
        for metric in ['final_gini_mean_mean', 'rank_mobility_range_mean']:
            groups = [group[metric].values for name, group in self.df.groupby('b0')]
            f_stat, p_value = stats.f_oneway(*groups)
            
            sig_tests.append({
                'Test': f'b0 effect on {metric}',
                'F-statistic': f_stat,
                'p-value': p_value,
                'Significant': p_value < 0.05
            })
        
        # Group by b1 values and test differences
        for metric in ['final_gini_mean_mean', 'rank_mobility_range_mean']:
            groups = [group[metric].values for name, group in self.df.groupby('b1')]
            f_stat, p_value = stats.f_oneway(*groups)
            
            sig_tests.append({
                'Test': f'b1 effect on {metric}',
                'F-statistic': f_stat,
                'p-value': p_value,
                'Significant': p_value < 0.05
            })
        
        # Create summary
        sig_df = pd.DataFrame(sig_tests)
        print("\nStatistical Significance Tests:")
        print("=" * 60)
        print(sig_df.to_string(index=False, float_format='%.6f'))
        
        # Save results
        sig_df.to_csv(self.figures_dir / 'significance_tests.csv', index=False)
        
        return sig_df
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Running complete hyperparameter analysis...")
        print("=" * 60)
        
        if not self.load_results():
            return False
        
        # Run all analysis components
        self.create_parameter_heatmaps()
        self.create_pooling_rate_analysis() 
        self.create_market_structure_analysis()
        self.create_power_law_vs_logistic_comparison()
        optimal_params = self.create_optimal_parameter_analysis()
        self.create_parameter_correlation_analysis()
        sig_tests = self.create_statistical_significance_tests()
        
        # NEW: Parameter importance and interaction analysis
        self.analyze_parameter_importance_and_interactions()
        
        # NEW: Create heatmaps for all parameter interactions
        self.create_all_parameter_interaction_heatmaps()
        
        print(f"\nAnalysis complete! All figures saved to {self.figures_dir}/")
        print(f"Key results saved to CSV files in {self.figures_dir}/")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter validation results')
    parser.add_argument('--results', type=str, default=None,
                       help='Path to results file (default: results/hyperparameter_results_merged.pkl)')
    
    args = parser.parse_args()
    
    analyzer = HyperparameterAnalyzer(args.results)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()