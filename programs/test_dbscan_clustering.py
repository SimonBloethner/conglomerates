#!/usr/bin/env python3
"""
Test script for DBSCAN merger clustering
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def create_synthetic_merger_data():
    """Create synthetic merger time series with known wave patterns"""
    np.random.seed(42)
    
    # 1000 time periods
    time_series = np.zeros(1000)
    
    # Add some background noise (low-level random mergers)
    background = np.random.poisson(0.1, 1000)  # Low rate background
    time_series += background
    
    # Add merger waves at specific periods
    # Wave 1: periods 200-220 (early wave)
    time_series[200:221] += np.random.poisson(3, 21)  # Higher intensity
    
    # Wave 2: periods 450-470 (mid wave)  
    time_series[450:471] += np.random.poisson(2, 21)  # Medium intensity
    
    # Wave 3: periods 750-760 (late wave, shorter)
    time_series[750:761] += np.random.poisson(4, 11)  # High intensity, short
    
    # Add some isolated high-activity periods (should be noise)
    time_series[350] += 5
    time_series[600] += 3
    
    return time_series

def detect_merger_clusters_dbscan(merger_time_series, exclude_early=50, eps=10, min_samples=3):
    """DBSCAN clustering for merger waves (same as in parallel script)"""
    if np.sum(merger_time_series) == 0:
        return []
    
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
            scaled_time = (t + exclude_early) / len(merger_time_series)
            scaled_intensity = activity / np.max(clean_series)
            active_periods.append([scaled_time, scaled_intensity])
            period_indices.append(t + exclude_early)
    
    if len(active_periods) < min_samples:
        return []
    
    # Apply DBSCAN clustering
    active_periods = np.array(active_periods)
    
    # Scale eps appropriately for normalized features
    normalized_eps = eps / len(merger_time_series)
    
    dbscan = DBSCAN(eps=normalized_eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(active_periods)
    
    # Convert clusters back to merger wave format
    clusters = []
    
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:  # Noise points
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

def visualize_clusters(merger_time_series, clusters):
    """Visualize the detected clusters"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Full time series with detected clusters
    time_periods = np.arange(len(merger_time_series))
    ax1.plot(time_periods, merger_time_series, 'b-', alpha=0.7, linewidth=1, label='Merger Activity')
    
    # Color code the clusters
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for i, cluster in enumerate(clusters):
        color = colors[i % len(colors)]
        periods = cluster['periods']
        intensities = [merger_time_series[p] for p in periods]
        
        ax1.scatter(periods, intensities, color=color, s=30, alpha=0.8, 
                   label=f"Cluster {cluster['cluster_id']} (Peak: {cluster['peak_period']})")
        
        # Highlight cluster span
        ax1.axvspan(cluster['start_period'], cluster['end_period'], 
                   alpha=0.2, color=color)
    
    ax1.set_title('DBSCAN Merger Wave Detection')
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Merger Activity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cluster characteristics
    if clusters:
        cluster_ids = [c['cluster_id'] for c in clusters]
        peak_times = [c['peak_period'] for c in clusters]
        peak_intensities = [c['peak_intensity'] for c in clusters]
        durations = [c['duration'] for c in clusters]
        total_mergers = [c['total_mergers'] for c in clusters]
        
        ax2.scatter(peak_times, peak_intensities, s=[d*5 for d in durations], 
                   c=total_mergers, cmap='viridis', alpha=0.7)
        ax2.set_title('Cluster Characteristics (Size = Duration, Color = Total Mergers)')
        ax2.set_xlabel('Peak Time')
        ax2.set_ylabel('Peak Intensity')
        
        # Add colorbar
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Total Mergers in Cluster')
        
        # Annotate clusters
        for i, cluster in enumerate(clusters):
            ax2.annotate(f"C{cluster['cluster_id']}", 
                        (cluster['peak_period'], cluster['peak_intensity']),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('dbscan_clustering_test.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== Testing DBSCAN Merger Clustering ===")
    
    # Create synthetic data
    merger_data = create_synthetic_merger_data()
    
    print(f"Created synthetic time series with {len(merger_data)} periods")
    print(f"Total mergers: {np.sum(merger_data)}")
    print(f"Max activity in single period: {np.max(merger_data)}")
    
    # Test clustering with different parameters
    test_params = [
        {'eps': 5, 'min_samples': 3},
        {'eps': 10, 'min_samples': 3},
        {'eps': 15, 'min_samples': 5}
    ]
    
    for params in test_params:
        print(f"\n--- Testing with eps={params['eps']}, min_samples={params['min_samples']} ---")
        
        clusters = detect_merger_clusters_dbscan(merger_data, **params)
        
        print(f"Detected {len(clusters)} clusters:")
        for cluster in clusters:
            print(f"  Cluster {cluster['cluster_id']}: "
                  f"periods {cluster['start_period']}-{cluster['end_period']} "
                  f"(duration: {cluster['duration']}, "
                  f"peak: {cluster['peak_intensity']} at {cluster['peak_period']}, "
                  f"total: {cluster['total_mergers']})")
    
    # Visualize best result
    best_clusters = detect_merger_clusters_dbscan(merger_data, eps=10, min_samples=3)
    visualize_clusters(merger_data, best_clusters)
    
    print(f"\nVisualization saved as 'dbscan_clustering_test.png'")

if __name__ == "__main__":
    main()