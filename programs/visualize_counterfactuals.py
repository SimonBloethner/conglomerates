#!/usr/bin/env python3
"""
Visualize counterfactual analysis results (recreates plots from original counterfactuals.py)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from scipy import signal
import pickle
import argparse
from pathlib import Path
import os
from collections import defaultdict

# Set up matplotlib for publication-quality plots (without LaTeX dependency)
plt.rcParams.update({
    "text.usetex": False,  # Disable LaTeX to avoid dependency issues
    "font.family": "serif",
    "font.size": 20
})

class CounterfactualVisualizer:
    def __init__(self, results_file='counterfactual_results/counterfactual_results_final.pkl'):
        self.results_file = Path(results_file)
        self.results = None
        self.shares = None
        self.hyperparameters = None
        
        # Load results first to get hyperparameters
        self.load_results()
        
        # Set up output directory based on hyperparameters
        self.setup_output_paths()
        
        # Color mapping
        self.normalize = None
        self.colormap = cm.viridis
        
    def setup_output_paths(self):
        """Set up output paths based on environment and hyperparameters"""
        is_mac = os.getcwd().find('Simon') > 0
        is_windows = os.getcwd().find('nomis') > 0
        
        # Base paths
        if is_mac:
            base_path = '/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE/latex/figures'
        elif is_windows:
            base_path = 'C:\\Users\\nomis\\PycharmProjects\\conglomerates\\conglomerates\\figures'
        else:
            base_path = 'counterfactual_figures'
        
        # Create hyperparameter-based subdirectory
        if self.hyperparameters:
            hp = self.hyperparameters
            folder_name = f"m{hp.get('markets', 100)}_f{hp.get('firms_per_market', 100)}_s{hp.get('steps', 10000)}_t{hp.get('merge_thresh', 0.05):.3f}_b0_{hp.get('b0', 1e-5):.1e}_b1_{hp.get('b1', 1.2):.1f}"
            self.path_figures = Path(base_path) / folder_name
        else:
            self.path_figures = Path(base_path) / 'default'
        
        # Create output directory
        Path(self.path_figures).mkdir(parents=True, exist_ok=True)
    
    def load_results(self):
        """Load merged counterfactual results"""
        print(f"Loading results from {self.results_file}")
        
        if not self.results_file.exists():
            print(f"Error: {self.results_file} not found")
            print("Run merge_counterfactual_results.py first")
            return False
        
        with open(self.results_file, 'rb') as f:
            self.results = pickle.load(f)
        
        # Extract share values and sort
        self.shares = np.array(sorted(self.results.keys()))
        
        # Extract hyperparameters from first share (should be same for all)
        first_share = self.shares[0]
        if 'hyperparameters' in self.results[first_share]:
            self.hyperparameters = self.results[first_share]['hyperparameters']
            print(f"Extracted hyperparameters: {self.hyperparameters}")
        else:
            print("No hyperparameters found in results, using defaults")
            self.hyperparameters = {}
        
        # Set up color normalization
        self.normalize = mcolors.Normalize(vmin=self.shares.min(), vmax=self.shares.max())
        
        print(f"Loaded results for {len(self.shares)} share values")
        return True
    
    def plot_market_share_quantiles(self):
        """Plot market share quantiles over time (recreates shares.pdf)"""
        print("Creating market share quantiles plot...")
        
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 12))
        
        quantile_labels = ['50%', '90%', '99%', 'Maximum']
        
        for plot_idx in range(4):
            for share_idx, share_val in enumerate(self.shares):
                if share_val == 0.0:
                    linestyle = '--'
                else:
                    linestyle = '-'
                
                data = self.results[share_val]['market_share_quantiles_avg'][plot_idx, :]
                axes[plot_idx].plot(data, 
                                  color=self.colormap(self.normalize(share_val)), 
                                  linestyle=linestyle)
            
            axes[plot_idx].set_title(quantile_labels[plot_idx])
        
        # Add colorbar
        scalarmappable = cm.ScalarMappable(norm=self.normalize, cmap=self.colormap)
        scalarmappable.set_array(self.shares)
        fig.tight_layout()
        colorbar = plt.colorbar(scalarmappable, ax=axes, orientation='horizontal', 
                              fraction=0.05, pad=0.1)
        colorbar.set_label('α', labelpad=1)
        
        plt.savefig(f'{self.path_figures}/shares.pdf', format='pdf', dpi=300)
        plt.show()
    
    def plot_gini_quantiles(self):
        """Plot Gini coefficient quantiles over time (recreates ginis.pdf)"""
        print("Creating Gini coefficient quantiles plot...")
        
        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(16, 11))
        
        quantile_labels = ['10%', '25%', '50%', '75%', '90%']
        
        for plot_idx in range(5):
            for share_idx, share_val in enumerate(self.shares):
                if share_val == 0.0:
                    linestyle = '--'
                else:
                    linestyle = '-'
                
                data = self.results[share_val]['gini_quantiles_avg'][:, plot_idx]
                axes[plot_idx].plot(data, 
                                  color=self.colormap(self.normalize(share_val)), 
                                  linestyle=linestyle)
            
            axes[plot_idx].set_title(quantile_labels[plot_idx])
        
        # Add colorbar
        scalarmappable = cm.ScalarMappable(norm=self.normalize, cmap=self.colormap)
        scalarmappable.set_array(self.shares)
        fig.tight_layout()
        colorbar = plt.colorbar(scalarmappable, ax=axes, orientation='horizontal', 
                              fraction=0.05, pad=0.1)
        colorbar.set_label('α', labelpad=1)
        
        plt.savefig(f'{self.path_figures}/ginis.pdf', format='pdf', dpi=300)
        plt.show()
    
    def plot_conglomerate_dynamics(self):
        """Plot conglomerate size and count over time (recreates sizes.pdf)"""
        print("Creating conglomerate dynamics plot...")
        
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 11))
        
        for share_idx, share_val in enumerate(self.shares):
            if share_val == 0.0:
                continue  # Skip baseline as in original
            
            # Mean conglomerate size
            axes[0].plot(self.results[share_val]['mean_members_avg'], 
                        color=self.colormap(self.normalize(share_val)))
            
            # Mean conglomerate count
            axes[1].plot(self.results[share_val]['num_cong_avg'], 
                        color=self.colormap(self.normalize(share_val)))
        
        axes[0].set_title('Mean Conglomerate Size')
        axes[1].set_title('Mean Conglomerate Count')
        
        # Add colorbar
        scalarmappable = cm.ScalarMappable(norm=self.normalize, cmap=self.colormap)
        scalarmappable.set_array(self.shares)
        fig.tight_layout()
        colorbar = plt.colorbar(scalarmappable, ax=axes, orientation='horizontal', 
                              fraction=0.05, pad=0.1)
        colorbar.set_label('α', labelpad=1)
        
        plt.savefig(f'{self.path_figures}/sizes.pdf', format='pdf', dpi=300)
        plt.show()
    
    def plot_polynomial_estimates(self):
        """Plot polynomial coefficient estimates over time (recreates ests.pdf)"""
        print("Creating polynomial estimates plot...")
        
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))
        
        for share_idx, share_val in enumerate(self.shares):
            if share_val == 0.0:
                continue  # Skip baseline as in original
            
            est_data = self.results[share_val]['poly_estimates_avg']
            
            # Plot beta_1 and beta_2 (indices 1 and 0 as in original)
            axes[0].plot(est_data[:, 1], color=self.colormap(self.normalize(share_val)))
            axes[1].plot(est_data[:, 0], color=self.colormap(self.normalize(share_val)))
        
        axes[0].set_title('β₁')
        axes[1].set_title('β₂')
        
        # Add colorbar
        scalarmappable = cm.ScalarMappable(norm=self.normalize, cmap=self.colormap)
        scalarmappable.set_array(self.shares)
        fig.tight_layout()
        colorbar = plt.colorbar(scalarmappable, ax=axes, orientation='horizontal', 
                              fraction=0.05, pad=0.1)
        colorbar.set_label('α', labelpad=1)
        
        plt.savefig(f'{self.path_figures}/ests.pdf', format='pdf', dpi=300)
        plt.show()
    
    def plot_mobility_distributions(self):
        """Plot rank mobility distributions (recreates mobility.pdf)"""
        print("Creating mobility distributions plot...")
        
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 11))
        
        # Parameters for ridge plot
        overlap = 0.8
        offset = 0.8
        
        for i, share_val in enumerate(self.shares):
            color = self.colormap(self.normalize(share_val))
            
            # Plot range distributions
            if 'rank_ranges' in self.results[share_val]:
                ranges_data = self.results[share_val]['rank_ranges'].reshape(-1)
                density = gaussian_kde(ranges_data)
                x = np.linspace(ranges_data.min(), ranges_data.max(), 1000)
                y = density(x)
                
                # Scale and offset
                y = y / y.max() * overlap
                y = y + i * offset
                
                axes[0].fill_between(x, y, i * offset, alpha=0.8, color=color)
            
            # Plot standard deviation distributions
            if 'rank_std' in self.results[share_val]:
                std_data = self.results[share_val]['rank_std'].reshape(-1)
                density = gaussian_kde(std_data)
                x = np.linspace(std_data.min(), std_data.max(), 1000)
                y = density(x)
                
                # Scale and offset
                y = y / y.max() * overlap
                y = y + i * offset
                
                axes[1].fill_between(x, y, i * offset, alpha=0.8, color=color)
        
        # Remove y-ticks and set titles
        axes[0].set_yticks([])
        axes[1].set_yticks([])
        axes[0].set_title('Range')
        axes[1].set_title('Standard Deviation')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self.colormap, norm=self.normalize)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.15)
        cbar.set_label('α')
        
        plt.savefig(f'{self.path_figures}/mobility.pdf', format='pdf', dpi=300)
        plt.show()
    
    def detect_merger_waves(self, merger_time_series, threshold_multiplier=1.0, exclude_early=50):
        """Detect merger wave events in a single experiment"""
        if np.sum(merger_time_series) == 0:  # No mergers
            return []
        
        # Exclude early periods (lookback period) to avoid initialization noise
        if len(merger_time_series) <= exclude_early:
            return []
        
        # Work with post-initialization data only
        clean_series = merger_time_series[exclude_early:]
        
        if np.sum(clean_series) == 0:
            return []
            
        # Use moving average to smooth noise
        window_length = min(5, len(clean_series))
        if window_length % 2 == 0:
            window_length += 1  # Make odd for savgol_filter
        
        smoothed = signal.savgol_filter(clean_series, window_length=window_length, polyorder=1)
        
        # More sensitive threshold for wave detection
        baseline = np.median(smoothed)  # Use median instead of mean (more robust)
        mad = np.median(np.abs(smoothed - baseline))  # Median absolute deviation
        threshold = baseline + threshold_multiplier * mad
        
        # Minimum threshold to avoid noise
        min_threshold = np.percentile(smoothed, 75)  # 75th percentile
        threshold = max(threshold, min_threshold)
        
        if threshold <= baseline:  # Very low activity
            threshold = np.max(smoothed) * 0.6  # Use 60% of peak as threshold
            
        above_threshold = smoothed > threshold
        
        # Find wave events (consecutive periods above threshold)
        waves = []
        in_wave = False
        wave_start = None
        
        for t, is_elevated in enumerate(above_threshold):
            if is_elevated and not in_wave:
                wave_start = t
                in_wave = True
            elif not is_elevated and in_wave:
                wave_end = t - 1
                if wave_end > wave_start:  # Valid wave
                    # Convert back to original time series indices
                    orig_start = wave_start + exclude_early
                    orig_end = wave_end + exclude_early
                    wave_segment = merger_time_series[orig_start:orig_end+1]
                    peak_idx = np.argmax(wave_segment) + orig_start
                    waves.append({
                        'start': orig_start,
                        'end': orig_end,
                        'peak_time': peak_idx,
                        'peak_intensity': merger_time_series[peak_idx],
                        'duration': orig_end - orig_start + 1,
                        'total_mergers': np.sum(merger_time_series[orig_start:orig_end+1]),
                        'threshold_used': threshold
                    })
                in_wave = False
                
        # Handle case where wave extends to end
        if in_wave and wave_start is not None:
            wave_end = len(clean_series) - 1
            orig_start = wave_start + exclude_early
            orig_end = wave_end + exclude_early
            wave_segment = merger_time_series[orig_start:orig_end+1]
            peak_idx = np.argmax(wave_segment) + orig_start
            waves.append({
                'start': orig_start,
                'end': orig_end,
                'peak_time': peak_idx,
                'peak_intensity': merger_time_series[peak_idx],
                'duration': orig_end - orig_start + 1,
                'total_mergers': np.sum(merger_time_series[orig_start:orig_end+1]),
                'threshold_used': threshold
            })
        
        return waves
    
    def extract_wave_shapes(self, merger_time_series, wave_window=10):
        """Extract standardized wave shapes around peaks"""
        waves = self.detect_merger_waves(merger_time_series)
        wave_shapes = []
        
        for wave in waves:
            peak_time = wave['peak_time']
            # Extract window around peak
            start_idx = max(0, peak_time - wave_window//2)
            end_idx = min(len(merger_time_series), peak_time + wave_window//2 + 1)
            
            wave_shape = merger_time_series[start_idx:end_idx]
            
            # Normalize by peak value to compare shapes
            if wave['peak_intensity'] > 0:
                normalized_shape = wave_shape / wave['peak_intensity']
                wave_shapes.append({
                    'shape': normalized_shape,
                    'original_peak': wave['peak_intensity'],
                    'timing': peak_time,
                    'duration': wave['duration']
                })
        
        return wave_shapes
    
    def aggregate_wave_shapes(self, all_wave_shapes):
        """Aggregate wave shapes across all experiments"""
        if not all_wave_shapes:
            return None, None
            
        # Find common length (median length to avoid outliers)
        lengths = [len(shape['shape']) for shape in all_wave_shapes]
        target_length = int(np.median(lengths))
        
        normalized_shapes = []
        
        for shape_data in all_wave_shapes:
            shape = shape_data['shape']
            
            # Resize to target length
            if len(shape) > target_length:
                # Downsample
                indices = np.linspace(0, len(shape)-1, target_length).astype(int)
                resized = shape[indices]
            elif len(shape) < target_length:
                # Interpolate to upsample
                x_old = np.linspace(0, 1, len(shape))
                x_new = np.linspace(0, 1, target_length)
                resized = np.interp(x_new, x_old, shape)
            else:
                resized = shape
                
            normalized_shapes.append(resized)
        
        # Calculate mean and std
        mean_shape = np.mean(normalized_shapes, axis=0)
        std_shape = np.std(normalized_shapes, axis=0)
        
        return mean_shape, std_shape
    
    def plot_merger_wave_analysis(self):
        """Create comprehensive merger wave analysis visualization"""
        print("Creating merger wave analysis...")
        
        # Check if merger data is available
        sample_share = list(self.results.keys())[0]
        if 'mergers_per_period_all' not in self.results[sample_share]:
            print("❌ Merger data not found in results!")
            print("Available keys:", list(self.results[sample_share].keys()))
            print("You need to re-run counterfactual analysis with updated code.")
            return
        
        print("✅ Merger data found!")
        
        # Analyze merger data for all share values
        analysis_results = {}
        
        for share in self.shares:
            if 'mergers_per_period_all' not in self.results[share]:
                continue
                
            merger_data = self.results[share]['mergers_per_period_all']
            n_experiments, n_periods = merger_data.shape
            
            # Baseline aggregation
            merger_mean = np.mean(merger_data, axis=0)
            merger_std = np.std(merger_data, axis=0)
            merger_quantiles = np.percentile(merger_data, [10, 25, 50, 75, 90], axis=0)
            
            # Event-based analysis
            all_waves = []
            wave_timings = []
            wave_intensities = []
            wave_durations = []
            experiments_with_waves = 0
            
            # Wave shape analysis
            all_wave_shapes = []
            
            for exp_idx in range(n_experiments):
                experiment_mergers = merger_data[exp_idx, :]
                waves = self.detect_merger_waves(experiment_mergers)
                
                if waves:
                    experiments_with_waves += 1
                    all_waves.extend(waves)
                    
                    for wave in waves:
                        wave_timings.append(wave['peak_time'])
                        wave_intensities.append(wave['peak_intensity'])
                        wave_durations.append(wave['duration'])
                    
                    # Extract wave shapes
                    shapes = self.extract_wave_shapes(experiment_mergers)
                    all_wave_shapes.extend(shapes)
            
            # Aggregate wave shapes
            mean_wave_shape, std_wave_shape = self.aggregate_wave_shapes(all_wave_shapes)
            
            # Store results
            analysis_results[share] = {
                'baseline': {
                    'mean': merger_mean,
                    'std': merger_std,
                    'quantiles': merger_quantiles,
                    'total_mergers_per_exp': np.sum(merger_data, axis=1)
                },
                'waves': {
                    'wave_events': all_waves,
                    'experiments_with_waves': experiments_with_waves,
                    'total_experiments': n_experiments,
                    'wave_frequency': len(all_waves) / n_experiments,
                    'wave_timings': wave_timings,
                    'wave_intensities': wave_intensities,
                    'wave_durations': wave_durations
                },
                'wave_shapes': {
                    'mean_shape': mean_wave_shape,
                    'std_shape': std_wave_shape,
                    'n_shapes': len(all_wave_shapes)
                }
            }
            
            print(f"α={share:.2f}: {experiments_with_waves}/{n_experiments} experiments had waves, "
                  f"{len(all_waves)} total waves, avg {len(all_waves)/n_experiments:.1f} waves/exp")
        
        # Create visualization with better spacing
        fig = plt.figure(figsize=(20, 16))
        
        # Panel 1: Traditional time series with quantile bands
        ax1 = plt.subplot(3, 2, 1)
        
        # Create better color mapping for continuous appearance
        norm = plt.Normalize(vmin=self.shares.min(), vmax=self.shares.max())
        colors = [plt.cm.viridis(norm(share)) for share in self.shares]
        
        # Show only selected α values for clarity
        selected_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Representative subset
        
        for share in selected_alphas:
            if share not in analysis_results:
                continue
            i = list(self.shares).index(share)  # Get index for color
            data = analysis_results[share]['baseline']
            q10, q25, q50, q75, q90 = data['quantiles']
            
            x = np.arange(len(q50))
            # Only show median line with subtle confidence band
            ax1.fill_between(x, q25, q75, alpha=0.15, color=colors[i])
            ax1.plot(x, q50, color=colors[i], linewidth=2.5, 
                    label=f'α={share:.1f}', alpha=0.8)
        
        ax1.set_title('Merger Frequency Over Time\n(Selected α Values)', fontsize=14)
        ax1.set_xlabel('Period', fontsize=12)
        ax1.set_ylabel('Mergers per Period', fontsize=12)
        ax1.legend(fontsize=10, ncol=2)  # Add legend for selected values
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Wave timing heatmap
        ax2 = plt.subplot(3, 2, 2)
        timing_data = []
        
        for share in self.shares:
            if share not in analysis_results:
                continue
            timings = analysis_results[share]['waves']['wave_timings']
            timing_data.extend(timings)
        
        if timing_data:
            max_period = max(timing_data)
            # Better binning: 50-period bins (roughly every 50 steps) for clarity
            n_bins = max(10, (max_period - 50) // 50)  # Exclude first 50 periods, then 50-period bins
            bin_edges = np.linspace(50, max_period, n_bins + 1)
            
            hist_data = []
            
            for share in self.shares:
                if share not in analysis_results:
                    hist_data.append(np.zeros(n_bins))
                    continue
                timings = analysis_results[share]['waves']['wave_timings']
                if timings:
                    # Filter out early periods and bin
                    filtered_timings = [t for t in timings if t >= 50]
                    if filtered_timings:
                        hist, _ = np.histogram(filtered_timings, bins=bin_edges)
                    else:
                        hist = np.zeros(n_bins)
                else:
                    hist = np.zeros(n_bins)
                hist_data.append(hist)
            
            if hist_data and any(len(h) > 0 for h in hist_data):
                # Ensure all arrays have same length
                max_len = max(len(h) for h in hist_data if len(h) > 0)
                padded_hist = []
                for h in hist_data:
                    if len(h) == 0:
                        padded_hist.append(np.zeros(max_len))
                    elif len(h) < max_len:
                        padded_hist.append(np.pad(h, (0, max_len - len(h))))
                    else:
                        padded_hist.append(h[:max_len])
                
                im = ax2.imshow(padded_hist, aspect='auto', cmap='YlOrRd', interpolation='nearest')
                ax2.set_title('Wave Timing Distribution\n(Excluding First 50 Periods)', fontsize=14)
                ax2.set_xlabel('Period Bins (50-period intervals)', fontsize=12)
                ax2.set_ylabel('Share Value (α)', fontsize=12)
                ax2.set_yticks(range(len(self.shares)))
                ax2.set_yticklabels([f'{s:.2f}' for s in self.shares], fontsize=10)
                
                # Better x-axis labels
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                n_ticks = min(5, len(bin_centers))
                tick_indices = np.linspace(0, len(bin_centers)-1, n_ticks, dtype=int)
                ax2.set_xticks(tick_indices)
                ax2.set_xticklabels([f'{int(bin_centers[i])}' for i in tick_indices], fontsize=10)
                
                plt.colorbar(im, ax=ax2, label='Wave Count')
        
        # Panel 3: Average wave shape
        ax3 = plt.subplot(3, 2, 3)
        
        for i, share in enumerate(self.shares):
            if share not in analysis_results:
                continue
            wave_shape_data = analysis_results[share]['wave_shapes']
            if wave_shape_data['mean_shape'] is not None:
                mean_shape = wave_shape_data['mean_shape']
                std_shape = wave_shape_data['std_shape']
                x_shape = np.arange(len(mean_shape)) - len(mean_shape)//2
                
                ax3.fill_between(x_shape, mean_shape - std_shape, mean_shape + std_shape, 
                               alpha=0.3, color=colors[i])
                ax3.plot(x_shape, mean_shape, color=colors[i], label=f'α={share:.2f}', linewidth=2)
        
        ax3.set_title('Canonical Wave Shape\n(Normalized, Peak-Centered)', fontsize=14)
        ax3.set_xlabel('Periods Relative to Peak', fontsize=12)
        ax3.set_ylabel('Normalized Merger Intensity', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # Panel 4: Wave characteristics by α
        ax4 = plt.subplot(3, 2, 4)
        
        wave_frequencies = []
        avg_intensities = []
        avg_durations = []
        
        for share in self.shares:
            if share not in analysis_results:
                wave_frequencies.append(0)
                avg_intensities.append(0)
                avg_durations.append(0)
                continue
                
            wave_data = analysis_results[share]['waves']
            wave_frequencies.append(wave_data['wave_frequency'])
            
            if wave_data['wave_intensities']:
                avg_intensities.append(np.mean(wave_data['wave_intensities']))
                avg_durations.append(np.mean(wave_data['wave_durations']))
            else:
                avg_intensities.append(0)
                avg_durations.append(0)
        
        # Create line plot with dual y-axes for better readability
        ax4_twin = ax4.twinx()
        
        # Plot wave frequency on primary axis
        line1 = ax4.plot(self.shares, wave_frequencies, 'o-', color='blue', linewidth=2, 
                        markersize=6, label='Wave Frequency')
        
        # Plot intensity on secondary axis  
        line2 = ax4_twin.plot(self.shares, avg_intensities, 's-', color='red', linewidth=2, 
                             markersize=6, label='Avg Intensity')
        
        # Plot duration on secondary axis
        line3 = ax4_twin.plot(self.shares, avg_durations, '^-', color='green', linewidth=2, 
                             markersize=6, label='Avg Duration')
        
        ax4.set_title('Wave Characteristics by α', fontsize=14)
        ax4.set_xlabel('Share Value (α)', fontsize=12)
        ax4.set_ylabel('Wave Frequency', fontsize=12, color='blue')
        ax4_twin.set_ylabel('Intensity / Duration', fontsize=12, color='red')
        
        # Color the y-axis labels to match the data
        ax4.tick_params(axis='y', labelcolor='blue')
        ax4_twin.tick_params(axis='y', labelcolor='red')
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left', fontsize=10)
        
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Total merger distribution (mean with error bands)
        ax5 = plt.subplot(3, 2, 5)
        
        means = []
        stds = []
        valid_shares = []
        
        for share in self.shares:
            if share not in analysis_results:
                continue
            totals = analysis_results[share]['baseline']['total_mergers_per_exp']
            means.append(np.mean(totals))
            stds.append(np.std(totals))
            valid_shares.append(share)
        
        if means:
            means = np.array(means)
            stds = np.array(stds)
            valid_shares = np.array(valid_shares)
            
            # Plot mean line
            ax5.plot(valid_shares, means, 'b-', linewidth=2, label='Mean')
            
            # Add error bands (mean ± std)
            ax5.fill_between(valid_shares, means - stds, means + stds, 
                           alpha=0.3, color='blue', label='±1 std')
        
        ax5.set_title('Total Mergers per Experiment\n(Mean ± Std)', fontsize=14)
        ax5.set_xlabel('Share Value (α)', fontsize=12)
        ax5.set_ylabel('Total Mergers', fontsize=12)
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=10)
        
        # Panel 6: Wave intensity vs timing scatter
        ax6 = plt.subplot(3, 2, 6)
        
        for i, share in enumerate(self.shares):
            if share not in analysis_results:
                continue
            wave_data = analysis_results[share]['waves']
            if wave_data['wave_timings'] and wave_data['wave_intensities']:
                ax6.scatter(wave_data['wave_timings'], wave_data['wave_intensities'], 
                          color=colors[i], alpha=0.6, s=30)
        
        ax6.set_title('Wave Intensity vs Timing', fontsize=14)
        ax6.set_xlabel('Wave Peak Time', fontsize=12)
        ax6.set_ylabel('Wave Peak Intensity', fontsize=12)
        ax6.grid(True, alpha=0.3)
        
        # Add shared colorbar for α values
        plt.subplots_adjust(left=0.08, bottom=0.15, right=0.95, top=0.95, wspace=0.3, hspace=0.4)
        
        # Create colorbar
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])  # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Share Value (α)', fontsize=14, labelpad=10)
        cbar.ax.tick_params(labelsize=12)
        
        plt.savefig(f'{self.path_figures}/merger_waves.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_all_plots(self):
        """Create all visualization plots"""
        if not self.load_results():
            return False
        
        print("Creating all counterfactual visualizations...")
        
        self.plot_market_share_quantiles()
        self.plot_gini_quantiles()
        self.plot_conglomerate_dynamics()
        self.plot_polynomial_estimates()
        self.plot_mobility_distributions()
        self.plot_merger_wave_analysis()
        
        print(f"All plots saved to {self.path_figures}/")
        return True

def main():
    parser = argparse.ArgumentParser(description='Visualize counterfactual results')
    parser.add_argument('--results', type=str, 
                       default='counterfactual_results/counterfactual_results_final.pkl',
                       help='Path to merged results file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for figures (default: auto-detect)')
    
    args = parser.parse_args()
    
    visualizer = CounterfactualVisualizer(args.results)
    
    if args.output_dir:
        visualizer.path_figures = args.output_dir
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    visualizer.create_all_plots()

if __name__ == "__main__":
    main()