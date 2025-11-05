#!/usr/bin/env python3
"""
Comprehensive merger wave analysis and visualization
Combines baseline aggregation, event-based analysis, and wave shape aggregation
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy import signal
from collections import defaultdict
import seaborn as sns

# Set plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif", 
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12
})

class MergerWaveAnalyzer:
    def __init__(self, threshold_multiplier=1.5, wave_window=10):
        self.threshold_multiplier = threshold_multiplier
        self.wave_window = wave_window
        
    def detect_merger_waves(self, merger_time_series):
        """Detect merger wave events in a single experiment"""
        if np.sum(merger_time_series) == 0:  # No mergers
            return []
            
        # Use moving average to smooth noise
        smoothed = signal.savgol_filter(merger_time_series, window_length=min(5, len(merger_time_series)), polyorder=1)
        
        # Dynamic threshold based on experiment's own statistics
        threshold = np.mean(smoothed) + self.threshold_multiplier * np.std(smoothed)
        
        if threshold <= 0:  # Very low merger activity
            threshold = np.max(smoothed) * 0.7  # Use 70% of peak as threshold
            
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
                    wave_segment = merger_time_series[wave_start:wave_end+1]
                    peak_idx = np.argmax(wave_segment) + wave_start
                    waves.append({
                        'start': wave_start,
                        'end': wave_end,
                        'peak_time': peak_idx,
                        'peak_intensity': merger_time_series[peak_idx],
                        'duration': wave_end - wave_start + 1,
                        'total_mergers': np.sum(merger_time_series[wave_start:wave_end+1]),
                        'threshold_used': threshold
                    })
                in_wave = False
                
        # Handle case where wave extends to end
        if in_wave and wave_start is not None:
            wave_end = len(merger_time_series) - 1
            wave_segment = merger_time_series[wave_start:wave_end+1]
            peak_idx = np.argmax(wave_segment) + wave_start
            waves.append({
                'start': wave_start,
                'end': wave_end,
                'peak_time': peak_idx,
                'peak_intensity': merger_time_series[peak_idx],
                'duration': wave_end - wave_start + 1,
                'total_mergers': np.sum(merger_time_series[wave_start:wave_end+1]),
                'threshold_used': threshold
            })
        
        return waves
    
    def extract_wave_shapes(self, merger_time_series):
        """Extract standardized wave shapes around peaks"""
        waves = self.detect_merger_waves(merger_time_series)
        wave_shapes = []
        
        for wave in waves:
            peak_time = wave['peak_time']
            # Extract window around peak
            start_idx = max(0, peak_time - self.wave_window//2)
            end_idx = min(len(merger_time_series), peak_time + self.wave_window//2 + 1)
            
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
    
    def analyze_merger_data(self, all_results):
        """Comprehensive analysis of merger data"""
        shares = sorted(all_results.keys())
        analysis_results = {}
        
        for share in shares:
            if 'mergers_per_period_all' not in all_results[share]:
                print(f"Warning: No merger data for share {share}")
                continue
                
            merger_data = all_results[share]['mergers_per_period_all']
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
        
        return analysis_results
    
    def create_visualizations(self, analysis_results):
        """Create comprehensive visualization panels"""
        shares = sorted(analysis_results.keys())
        
        fig = plt.figure(figsize=(16, 12))
        
        # Panel 1: Traditional time series with quantile bands
        ax1 = plt.subplot(2, 3, 1)
        colors = plt.cm.viridis(np.linspace(0, 1, len(shares)))
        
        for i, share in enumerate(shares):
            data = analysis_results[share]['baseline']
            mean_line = data['mean']
            q10, q25, q50, q75, q90 = data['quantiles']
            
            x = np.arange(len(mean_line))
            ax1.fill_between(x, q10, q90, alpha=0.2, color=colors[i])
            ax1.fill_between(x, q25, q75, alpha=0.3, color=colors[i])
            ax1.plot(x, q50, color=colors[i], label=f'α={share:.2f}', linewidth=2)
        
        ax1.set_title('Merger Frequency Over Time\\n(Median + Quantile Bands)')
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Mergers per Period')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Wave timing heatmap
        ax2 = plt.subplot(2, 3, 2)
        timing_data = []
        timing_labels = []
        
        for share in shares:
            timings = analysis_results[share]['waves']['wave_timings']
            if timings:
                timing_data.extend(timings)
                timing_labels.extend([f'{share:.2f}'] * len(timings))
        
        if timing_data:
            # Create histogram data for heatmap
            max_period = max(timing_data)
            hist_data = []
            
            for share in shares:
                timings = analysis_results[share]['waves']['wave_timings']
                hist, _ = np.histogram(timings, bins=max_period//5, range=(0, max_period))
                hist_data.append(hist)
            
            if hist_data:
                im = ax2.imshow(hist_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
                ax2.set_title('Wave Timing Distribution')
                ax2.set_xlabel('Period Bins')
                ax2.set_ylabel('Share Value (α)')
                ax2.set_yticks(range(len(shares)))
                ax2.set_yticklabels([f'{s:.2f}' for s in shares])
                plt.colorbar(im, ax=ax2, label='Wave Count')
        
        # Panel 3: Average wave shape
        ax3 = plt.subplot(2, 3, 3)
        
        for i, share in enumerate(shares):
            wave_shape_data = analysis_results[share]['wave_shapes']
            if wave_shape_data['mean_shape'] is not None:
                mean_shape = wave_shape_data['mean_shape']
                std_shape = wave_shape_data['std_shape']
                x_shape = np.arange(len(mean_shape)) - len(mean_shape)//2
                
                ax3.fill_between(x_shape, mean_shape - std_shape, mean_shape + std_shape, 
                               alpha=0.3, color=colors[i])
                ax3.plot(x_shape, mean_shape, color=colors[i], label=f'α={share:.2f}', linewidth=2)
        
        ax3.set_title('Canonical Wave Shape\\n(Normalized, Peak-Centered)')
        ax3.set_xlabel('Periods Relative to Peak')
        ax3.set_ylabel('Normalized Merger Intensity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # Panel 4: Wave characteristics by α
        ax4 = plt.subplot(2, 3, 4)
        
        wave_frequencies = []
        avg_intensities = []
        avg_durations = []
        
        for share in shares:
            wave_data = analysis_results[share]['waves']
            wave_frequencies.append(wave_data['wave_frequency'])
            
            if wave_data['wave_intensities']:
                avg_intensities.append(np.mean(wave_data['wave_intensities']))
                avg_durations.append(np.mean(wave_data['wave_durations']))
            else:
                avg_intensities.append(0)
                avg_durations.append(0)
        
        x_pos = np.arange(len(shares))
        width = 0.25
        
        ax4.bar(x_pos - width, wave_frequencies, width, label='Wave Frequency', alpha=0.8)
        
        # Scale intensity and duration for visibility
        max_freq = max(wave_frequencies) if wave_frequencies else 1
        scaled_intensities = [i/max(avg_intensities)*max_freq if max(avg_intensities) > 0 else 0 for i in avg_intensities]
        scaled_durations = [d/max(avg_durations)*max_freq if max(avg_durations) > 0 else 0 for d in avg_durations]
        
        ax4.bar(x_pos, scaled_intensities, width, label='Avg Intensity (scaled)', alpha=0.8)
        ax4.bar(x_pos + width, scaled_durations, width, label='Avg Duration (scaled)', alpha=0.8)
        
        ax4.set_title('Wave Characteristics by α')
        ax4.set_xlabel('Share Value (α)')
        ax4.set_ylabel('Scaled Values')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'{s:.2f}' for s in shares])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Total merger distribution
        ax5 = plt.subplot(2, 3, 5)
        
        total_merger_data = []
        for share in shares:
            totals = analysis_results[share]['baseline']['total_mergers_per_exp']
            total_merger_data.append(totals)
        
        bp = ax5.boxplot(total_merger_data, labels=[f'{s:.2f}' for s in shares])
        ax5.set_title('Total Mergers per Experiment\\n(Distribution)')
        ax5.set_xlabel('Share Value (α)')
        ax5.set_ylabel('Total Mergers')
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Wave intensity vs timing scatter
        ax6 = plt.subplot(2, 3, 6)
        
        for i, share in enumerate(shares):
            wave_data = analysis_results[share]['waves']
            if wave_data['wave_timings'] and wave_data['wave_intensities']:
                ax6.scatter(wave_data['wave_timings'], wave_data['wave_intensities'], 
                          color=colors[i], alpha=0.6, label=f'α={share:.2f}', s=30)
        
        ax6.set_title('Wave Intensity vs Timing')
        ax6.set_xlabel('Wave Peak Time')
        ax6.set_ylabel('Wave Peak Intensity')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

def load_counterfactual_results():
    """Load merged counterfactual results"""
    possible_files = [
        'counterfactual_results/all_counterfactuals_merged.pkl',
        'all_counterfactuals_merged.pkl',
        'counterfactual_results.pkl'
    ]
    
    for file_path in possible_files:
        if Path(file_path).exists():
            print(f"Loading results from: {file_path}")
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    
    print("No counterfactual results found. Available files:")
    for file in Path('.').glob('**/*.pkl'):
        print(f"  {file}")
    return None

def main():
    """Main analysis function"""
    print("=== MERGER WAVE ANALYSIS ===")
    
    # Load results
    all_results = load_counterfactual_results()
    if all_results is None:
        return
    
    print(f"Loaded results for {len(all_results)} share values")
    
    # Check if merger data is available
    sample_share = list(all_results.keys())[0]
    if 'mergers_per_period_all' not in all_results[sample_share]:
        print("❌ Merger data not found in results!")
        print("Available keys:", list(all_results[sample_share].keys()))
        print("You need to re-run counterfactual analysis with updated code.")
        return
    
    print("✅ Merger data found!")
    
    # Initialize analyzer
    analyzer = MergerWaveAnalyzer(threshold_multiplier=1.5, wave_window=10)
    
    # Perform analysis
    print("Analyzing merger patterns...")
    analysis_results = analyzer.analyze_merger_data(all_results)
    
    # Create visualizations
    print("Creating visualizations...")
    fig = analyzer.create_visualizations(analysis_results)
    
    # Save results
    fig.savefig('merger_wave_analysis.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Saved analysis to: merger_wave_analysis.pdf")
    
    # Save analysis data
    with open('merger_wave_analysis_data.pkl', 'wb') as f:
        pickle.dump(analysis_results, f)
    print("Saved analysis data to: merger_wave_analysis_data.pkl")

if __name__ == "__main__":
    main()