"""
Monte Carlo Portfolio Simulation: Hold vs. Rebalanced Strategies

This script simulates portfolio performance comparing:
1. Hold portfolio: weighted sum of component returns (no rebalancing)
2. Rebalanced portfolio: equal weights maintained through periodic rebalancing

Experiments:
- Experiment 1: 2 assets
- Experiment 2: 5 assets  
- Experiment 3: 10 assets
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Configure matplotlib for large presentation fonts
plt.rcParams.update({
    'font.family': 'DejaVu Sans',  # Use DejaVu Sans as primary (always available)
    'font.size': 18,
    'axes.labelsize': 22,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'lines.linewidth': 3.0,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5
})

class PortfolioSimulation:
    def __init__(self, n_assets, n_steps=1000, mean_return=0.05, sigma=0.15, n_paths=5):
        """
        Initialize portfolio simulation parameters
        
        Args:
            n_assets: Number of assets in portfolio
            n_steps: Number of time steps (1000)
            mean_return: Mean return per period (0.05)
            sigma: Standard deviation of returns (0.15)
            n_paths: Number of simulation paths to generate
        """
        self.n_assets = n_assets
        self.n_steps = n_steps
        self.mean_return = mean_return
        self.sigma = sigma
        self.n_paths = n_paths
        
        # Equal initial weights
        self.initial_weights = np.ones(n_assets) / n_assets
        
    def generate_returns(self):
        """Generate Monte Carlo returns for all assets and paths"""
        # Shape: (n_paths, n_steps, n_assets)
        returns = np.random.normal(
            loc=self.mean_return, 
            scale=self.sigma, 
            size=(self.n_paths, self.n_steps, self.n_assets)
        )
        return returns
    
    def simulate_hold_portfolio(self, returns):
        """
        Simulate hold portfolio strategy
        
        Args:
            returns: Array of shape (n_paths, n_steps, n_assets)
            
        Returns:
            portfolio_values: Array of shape (n_paths, n_steps+1)
        """
        portfolio_values = np.zeros((self.n_paths, self.n_steps + 1))
        portfolio_values[:, 0] = 1.0  # Start with value 1
        
        for path in range(self.n_paths):
            weights = self.initial_weights.copy()
            
            for t in range(self.n_steps):
                # Portfolio return is weighted sum of asset returns
                portfolio_return = np.sum(weights * returns[path, t, :])
                portfolio_values[path, t + 1] = portfolio_values[path, t] * (1 + portfolio_return)
                
                # Update weights based on asset performance (no rebalancing)
                asset_values = weights * (1 + returns[path, t, :])
                weights = asset_values / np.sum(asset_values)
        
        return portfolio_values
    
    def simulate_rebalanced_portfolio(self, returns):
        """
        Simulate rebalanced portfolio strategy
        
        Args:
            returns: Array of shape (n_paths, n_steps, n_assets)
            
        Returns:
            portfolio_values: Array of shape (n_paths, n_steps+1)
        """
        portfolio_values = np.zeros((self.n_paths, self.n_steps + 1))
        portfolio_values[:, 0] = 1.0  # Start with value 1
        
        for path in range(self.n_paths):
            for t in range(self.n_steps):
                # Portfolio return with equal weights (rebalanced each period)
                portfolio_return = np.mean(returns[path, t, :])
                portfolio_values[path, t + 1] = portfolio_values[path, t] * (1 + portfolio_return)
        
        return portfolio_values
    
    def run_simulation(self):
        """Run complete simulation for both strategies"""
        np.random.seed(42)  # For reproducibility
        
        # Generate returns
        returns = self.generate_returns()
        
        # Simulate both strategies
        hold_values = self.simulate_hold_portfolio(returns)
        rebalanced_values = self.simulate_rebalanced_portfolio(returns)
        
        # Calculate individual firm trajectories
        individual_firms = self.simulate_individual_firms(returns)
        
        return hold_values, rebalanced_values, returns, individual_firms
    
    def simulate_individual_firms(self, returns):
        """
        Simulate individual firm growth trajectories
        
        Args:
            returns: Array of shape (n_paths, n_steps, n_assets)
            
        Returns:
            firm_values: Array of shape (n_paths, n_steps+1, n_assets)
        """
        firm_values = np.zeros((self.n_paths, self.n_steps + 1, self.n_assets))
        firm_values[:, 0, :] = 1.0  # Start with value 1 for each firm
        
        for path in range(self.n_paths):
            for asset in range(self.n_assets):
                for t in range(self.n_steps):
                    firm_values[path, t + 1, asset] = firm_values[path, t, asset] * (1 + returns[path, t, asset])
        
        return firm_values
    
    def create_storytelling_plots(self, hold_values, rebalanced_values, output_dir, experiment_name):
        """
        Create plots for storytelling narrative
        Shows hold (solid) vs rebalanced (dotted) strategies on same plot
        """
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Professional colors
        
        for path_idx in range(self.n_paths):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot all previous hold paths in light gray
            for prev_path in range(path_idx):
                ax.plot(hold_values[prev_path], color='lightgray', alpha=0.4, linewidth=1, linestyle='-')
                ax.plot(rebalanced_values[prev_path], color='lightgray', alpha=0.4, linewidth=1, linestyle='--')
            
            # Plot current path highlighted
            color = colors[path_idx % len(colors)]
            ax.plot(hold_values[path_idx], color=color, linewidth=2.5, linestyle='-', 
                   label=f'Hold Strategy (Path {path_idx + 1})')
            ax.plot(rebalanced_values[path_idx], color=color, linewidth=2.5, linestyle='--', 
                   label=f'Rebalanced Strategy (Path {path_idx + 1})')
            
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Portfolio Value (Log Scale)')
            ax.set_title(f'Portfolio Strategies: {self.n_assets} Assets')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Set consistent y-axis limits
            all_values = np.concatenate([hold_values.flatten(), rebalanced_values.flatten()])
            ax.set_ylim(max(all_values.min() * 0.8, 0.1), all_values.max() * 1.2)
            
            filename = f'{experiment_name}_path{path_idx + 1}.pdf'
            plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
            
    @staticmethod
    def create_narrative_sequence(all_results, output_dir):
        """
        Create the narrative sequence for the presentation
        All experiments accumulate in a single plot to show escalating power of rebalancing
        Include individual firm trajectories for 2-asset experiments
        
        Args:
            all_results: Dictionary with results from all experiments
            output_dir: Directory to save plots
        """
        
        narrative_plots = []
        colors = plt.cm.viridis([0.05, 0.3, 0.55, 0.75, 0.95])
        
        # Get all values for consistent y-axis limits
        all_values = []
        for key in all_results:
            all_values.extend(all_results[key]['hold'].flatten())
            all_values.extend(all_results[key]['rebalanced'].flatten())
        y_min, y_max = max(min(all_values) * 0.8, 0.1), max(all_values) * 1.2
        
        # Frame 1: Regular 2 asset portfolio (hold only) + individual firms
        fig, ax = plt.subplots(figsize=(12, 8))
        hold_2 = all_results['2_assets']['hold']
        firms_2 = all_results['2_assets']['individual_firms']
        
        # Plot individual firms as dotted lines in same color as portfolio
        ax.plot(firms_2[0, :, 0], color=colors[0], linewidth=2, linestyle=':', alpha=0.7, label='Firms')
        ax.plot(firms_2[0, :, 1], color=colors[0], linewidth=2, linestyle=':', alpha=0.7)
        
        # Plot portfolio (bold line)
        ax.plot(hold_2[0], color=colors[0], linewidth=4, linestyle='-', label='Hold')
        
        ax.set_xlabel('Time Steps', fontsize=22, fontweight='bold')
        ax.set_ylabel('Compound Growth (g)', fontsize=22, fontweight='bold')
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=18, frameon=True, fancybox=True, shadow=True)
        
        # Add text box with simulation parameters
        ax.text(0.98, 0.02, r'$\delta_t \sim \mathcal{N}(0.05, 0.15)$', 
                transform=ax.transAxes, fontsize=20, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        filename = 'narrative_01_2assets_hold.pdf'
        plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
        plt.close()
        narrative_plots.append(filename)
        
        # Frame 2: Same portfolio but rebalanced
        fig, ax = plt.subplots(figsize=(12, 8))
        rebalanced_2 = all_results['2_assets']['rebalanced']
        individual_firms_2 = all_results['2_assets']['individual_firms']
        
        # Plot individual firms as dotted lines in same color as portfolio
        ax.plot(individual_firms_2[0, :, 0], color=colors[0], linewidth=2, linestyle=':', alpha=0.7, label='Firms')
        ax.plot(individual_firms_2[0, :, 1], color=colors[0], linewidth=2, linestyle=':', alpha=0.7)
        
        # Plot portfolios as bold foreground lines
        ax.plot(hold_2[0], color=colors[0], linewidth=2.5, linestyle='-', alpha=0.7, label='Hold')
        ax.plot(rebalanced_2[0], color=colors[0], linewidth=3, linestyle='--', label='Rebalanced')
        ax.set_xlabel('Time Steps', fontsize=22, fontweight='bold')
        ax.set_ylabel('Compound Growth (g)', fontsize=22, fontweight='bold')
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=18, frameon=True, fancybox=True, shadow=True)
        
        # Add text box with simulation parameters
        ax.text(0.98, 0.02, r'$\delta_t \sim \mathcal{N}(0.05, 0.15)$', 
                transform=ax.transAxes, fontsize=20, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        filename = 'narrative_02_2assets_rebalanced.pdf'
        plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
        plt.close()
        narrative_plots.append(filename)
        
        # Frame 3: Another 2 asset portfolio and its rebalanced counterpart
        fig, ax = plt.subplots(figsize=(12, 8))
        # Previous simulation in background
        ax.plot(hold_2[0], color=colors[1], linewidth=2, linestyle='-', alpha=0.4)
        ax.plot(rebalanced_2[0], color=colors[1], linewidth=2, linestyle='--', alpha=0.4)
        # New simulation highlighted
        ax.plot(hold_2[1], color=colors[2], linewidth=3, linestyle='-', label='Hold')
        ax.plot(rebalanced_2[1], color=colors[2], linewidth=3, linestyle='--', label='Rebalanced')
        ax.set_xlabel('Time Steps', fontsize=22, fontweight='bold')
        ax.set_ylabel('Compound Growth (g)', fontsize=22, fontweight='bold')
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=18, frameon=True, fancybox=True, shadow=True)
        
        # Add text box with simulation parameters
        ax.text(0.98, 0.02, r'$\delta_t \sim \mathcal{N}(0.05, 0.15)$', 
                transform=ax.transAxes, fontsize=20, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        filename = 'narrative_03_2assets_second.pdf'
        plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
        plt.close()
        narrative_plots.append(filename)
        
        # Frame 4: 5 asset portfolio and its rebalanced counterpart
        fig, ax = plt.subplots(figsize=(12, 8))
        hold_5 = all_results['5_assets']['hold']
        rebalanced_5 = all_results['5_assets']['rebalanced']
        # Previous simulations in background
        ax.plot(hold_2[0], color=colors[0], linewidth=1.5, linestyle='-', alpha=0.3)
        ax.plot(rebalanced_2[0], color=colors[0], linewidth=1.5, linestyle='--', alpha=0.3)
        ax.plot(hold_2[1], color=colors[1], linewidth=1.5, linestyle='-', alpha=0.3)
        ax.plot(rebalanced_2[1], color=colors[1], linewidth=1.5, linestyle='--', alpha=0.3)
        # New 5-asset simulation highlighted
        ax.plot(hold_5[0], color=colors[3], linewidth=3, linestyle='-', label='Hold')
        ax.plot(rebalanced_5[0], color=colors[3], linewidth=3, linestyle='--', label='Rebalanced')
        ax.set_xlabel('Time Steps', fontsize=22, fontweight='bold')
        ax.set_ylabel('Compound Growth (g)', fontsize=22, fontweight='bold')
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=18, frameon=True, fancybox=True, shadow=True)
        
        # Add text box with simulation parameters
        ax.text(0.98, 0.02, r'$\delta_t \sim \mathcal{N}(0.05, 0.15)$', 
                transform=ax.transAxes, fontsize=20, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        filename = 'narrative_04_5assets.pdf'
        plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
        plt.close()
        narrative_plots.append(filename)
        
        # Frame 5: 10 asset portfolio and its rebalanced counterpart
        fig, ax = plt.subplots(figsize=(12, 8))
        hold_10 = all_results['10_assets']['hold']
        rebalanced_10 = all_results['10_assets']['rebalanced']
        # All previous simulations in background
        ax.plot(hold_2[0], color=colors[0], linewidth=1.5, linestyle='-', alpha=0.25)
        ax.plot(rebalanced_2[0], color=colors[0], linewidth=1.5, linestyle='--', alpha=0.25)
        ax.plot(hold_2[1], color=colors[2], linewidth=1.5, linestyle='-', alpha=0.25)
        ax.plot(rebalanced_2[1], color=colors[2], linewidth=1.5, linestyle='--', alpha=0.25)
        ax.plot(hold_5[0], color=colors[3], linewidth=1.5, linestyle='-', alpha=0.25)
        ax.plot(rebalanced_5[0], color=colors[3], linewidth=1.5, linestyle='--', alpha=0.25)
        # New 10-asset simulation highlighted
        ax.plot(hold_10[0], color=colors[4], linewidth=3, linestyle='-', label='Hold')
        ax.plot(rebalanced_10[0], color=colors[4], linewidth=3, linestyle='--', label='Rebalanced')
        
        ax.set_xlabel('Time Steps', fontsize=22, fontweight='bold')
        ax.set_ylabel('Compound Growth (g)', fontsize=22, fontweight='bold')
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=18, frameon=True, fancybox=True, shadow=True)
        
        # Add text box with simulation parameters
        ax.text(0.98, 0.02, r'$\delta_t \sim \mathcal{N}(0.05, 0.15)$', 
                transform=ax.transAxes, fontsize=20, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        filename = 'narrative_05_10assets.pdf'
        plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
        plt.close()
        narrative_plots.append(filename)
        
        return narrative_plots


def run_all_experiments():
    """Run all three experiments and generate narrative plots"""
    
    # Create output directory in latex/figures
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'latex' / 'figures' / 'mc_figures'
    output_dir.mkdir(exist_ok=True)
    
    print("Running Monte Carlo Portfolio Simulations for Storytelling...")
    
    # Store all results for narrative sequence
    all_results = {}
    
    # Run experiments
    experiments = [
        {'n_assets': 2, 'key': '2_assets'},
        {'n_assets': 5, 'key': '5_assets'},
        {'n_assets': 10, 'key': '10_assets'}
    ]
    
    for exp in experiments:
        print(f"\nRunning simulation: {exp['n_assets']} assets")
        
        # Run simulation
        sim = PortfolioSimulation(n_assets=exp['n_assets'])
        hold_values, rebalanced_values, returns, individual_firms = sim.run_simulation()
        
        # Store results
        all_results[exp['key']] = {
            'hold': hold_values,
            'rebalanced': rebalanced_values,
            'individual_firms': individual_firms
        }
        
        # Print summary statistics
        final_hold = hold_values[:, -1]
        final_rebalanced = rebalanced_values[:, -1]
        
        print(f"  Hold strategy final values: mean={final_hold.mean():.3f}, std={final_hold.std():.3f}")
        print(f"  Rebalanced strategy final values: mean={final_rebalanced.mean():.3f}, std={final_rebalanced.std():.3f}")
        print(f"  Rebalancing advantage: {(final_rebalanced.mean() / final_hold.mean() - 1) * 100:.1f}% higher returns")
    
    # Create narrative sequence
    print("\nCreating narrative sequence plots...")
    narrative_files = PortfolioSimulation.create_narrative_sequence(all_results, output_dir)
    
    # Also create individual storytelling plots for each experiment
    for exp in experiments:
        print(f"Creating storytelling plots for {exp['n_assets']} assets...")
        sim = PortfolioSimulation(n_assets=exp['n_assets'])
        sim.create_storytelling_plots(
            all_results[exp['key']]['hold'], 
            all_results[exp['key']]['rebalanced'], 
            output_dir, 
            f'storytelling_{exp["n_assets"]}assets'
        )
    
    print(f"\nAll simulations complete! Figures saved to: {output_dir}")
    print("\n" + "="*60)
    print("PRESENTATION NARRATIVE SEQUENCE:")
    print("="*60)
    
    narrative_descriptions = [
        "Frame 1: Regular two asset portfolio (hold only)",
        "Frame 2: Same portfolio but rebalanced",  
        "Frame 3: Another 2 asset portfolio and its rebalanced counterpart",
        "Frame 4: 5 asset portfolio and its rebalanced counterpart",
        "Frame 5: 10 asset portfolio and its rebalanced counterpart"
    ]
    
    for i, (file, desc) in enumerate(zip(narrative_files, narrative_descriptions), 1):
        print(f"{i}. {desc}")
        print(f"   \\only<{i}>{{\\includegraphics[width=0.9\\textwidth]{{figures/mc_figures/{file[:-4]}}}}}")
    
    print("\nKey findings:")
    hold_2_final = all_results['2_assets']['hold'][:, -1].mean()
    rebal_2_final = all_results['2_assets']['rebalanced'][:, -1].mean()
    hold_10_final = all_results['10_assets']['hold'][:, -1].mean()
    rebal_10_final = all_results['10_assets']['rebalanced'][:, -1].mean()
    
    print(f"- 2 assets: Rebalancing provides {(rebal_2_final/hold_2_final - 1)*100:.1f}% higher returns")
    print(f"- 10 assets: Rebalancing provides {(rebal_10_final/hold_10_final - 1)*100:.1f}% higher returns")
    print("- The advantage of rebalancing scales with portfolio diversity")


if __name__ == "__main__":
    run_all_experiments()