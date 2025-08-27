"""
Script to regenerate plots from saved simulation data with customizable font sizes
This script loads the data saved by counterfactuals.py and recreates all the plots
with different font settings without re-running the simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
import os

# ============================================================================
# FONT CONFIGURATION - CUSTOMIZE THESE VALUES
# ============================================================================

# Option 1: For presentations (larger fonts)
FONT_SIZE_PRESENTATION = {
    "font.size": 28,           # Base font size
    "axes.titlesize": 32,      # Title font size (subplot titles)
    "axes.labelsize": 30,      # Axis label font size
    "xtick.labelsize": 28,     # X-tick labels
    "ytick.labelsize": 28,     # Y-tick labels
    "legend.fontsize": 30,     # Legend font size
    "figure.titlesize": 36     # Figure title size
}

# Option 2: For papers (moderate fonts)
FONT_SIZE_PAPER = {
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "figure.titlesize": 20
}

# Option 3: Original settings
FONT_SIZE_ORIGINAL = {
    "font.size": 20
}

# Choose which font configuration to use
FONT_CONFIG = FONT_SIZE_PRESENTATION  # Change this line to switch configurations

# ============================================================================

# Set up matplotlib with LaTeX and chosen font sizes
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif", 
    "font.serif": ["Computer Modern Roman"],
    **FONT_CONFIG
})

# Path configuration
is_mac = os.getcwd().find('Simon') > 0
is_windows = os.getcwd().find('nomis') > 0

if is_mac:
    path_figures = '/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE/latex/figures'
    path_outdata = '/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE/outdata'
elif is_windows:
    path_figures = 'C:\\Users\\nomis\\PycharmProjects\\conglomerates\\conglomerates\\figures'
    path_outdata = 'C:\\Users\\nomis\\PycharmProjects\\conglomerates\\conglomerates\\outdata'
else:
    path_figures = 'conglomerate/figures'
    path_outdata = 'conglomerate/outdata'

print("Loading saved simulation data...")

# Load all saved data
try:
    mean_gini = np.load(f'{path_outdata}/mean_gini.npy')
    mean_quantiles = np.load(f'{path_outdata}/mean_quantiles.npy')
    mean_members_ = np.load(f'{path_outdata}/mean_members_.npy')
    mean_conglomerates = np.load(f'{path_outdata}/mean_conglomerates.npy')
    
    # Load the .npz files
    mean_persistence_data = np.load(f'{path_outdata}/mean_persistence.npz')
    mean_persistence = [mean_persistence_data[f'arr_{i}'] for i in range(len(mean_persistence_data.files))]
    
    mean_persistence_quantiles_data = np.load(f'{path_outdata}/mean_persistence_quantiles.npz')
    mean_persistence_quantiles = [mean_persistence_quantiles_data[f'arr_{i}'] for i in range(len(mean_persistence_quantiles_data.files))]
    
    # mean_ests is saved as .npz format but with incorrect unpacking
    # The original saves with *mean_ests which splits the first dimension
    mean_ests_data = np.load(f'{path_outdata}/mean_ests.npz')
    # Reconstruct the original 3D array from the split arrays
    mean_ests_list = [mean_ests_data[f'arr_{i}'] for i in range(len(mean_ests_data.files))]
    mean_ests = np.array(mean_ests_list)
    
    ranges_data = np.load(f'{path_outdata}/ranges.npz')
    ranges = [ranges_data[f'arr_{i}'] for i in range(len(ranges_data.files))]
    
    sds_data = np.load(f'{path_outdata}/sds.npz')
    sds = [sds_data[f'arr_{i}'] for i in range(len(sds_data.files))]
    
    print("✓ All data loaded successfully!")
    
except FileNotFoundError as e:
    print(f"Error: Could not find data file: {e}")
    print("Make sure you have run counterfactuals.py first to generate the data files.")
    exit(1)

# Recreate the original parameters used in the simulation
shares = np.arange(0, 0.52, 0.02)
steps = 1000
ramp = 10

# Set up colormap
normalize = mcolors.Normalize(vmin=shares.min(), vmax=shares.max())
colormap = cm.viridis

print("Regenerating plots with new font sizes...")

# ============================================================================
# PLOT 1: Market Share Quantiles (shares.pdf)
# ============================================================================
print("Creating shares.pdf...")
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 12))

for plot_ in range(4):
    for row, share_ in enumerate(shares):
        axes[plot_].plot(mean_quantiles[row, plot_, :], 
                        color=colormap(normalize(share_)), 
                        linestyle='-' if row != 0 else '--')

axes[0].set_title('50\\%')
axes[1].set_title('90\\%')  
axes[2].set_title('99\\%')
axes[3].set_title('Maximum')

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(shares)
fig.tight_layout()
colorbar = plt.colorbar(scalarmappaple, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
colorbar.set_label(r'$\alpha$', labelpad=1, fontsize=FONT_CONFIG.get("axes.labelsize", 30))
colorbar.ax.tick_params(labelsize=FONT_CONFIG.get("xtick.labelsize", 28))
plt.savefig(f'{path_figures}/shares.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 2: Gini Coefficients (ginis.pdf) - Simplified for presentation
# ============================================================================
print("Creating ginis.pdf...")
# Use only 10%, 50%, 90% percentiles (indices 0, 2, 4) for cleaner presentation
selected_percentiles = [0, 2, 4]  # 10%, 50%, 90%
percentile_labels = ['10\\%', '50\\%', '90\\%']

fig, axes = plt.subplots(nrows=len(selected_percentiles), ncols=1, figsize=(16, 9))

for i, plot_ in enumerate(selected_percentiles):
    for row, share_ in enumerate(shares):
        axes[i].plot(mean_gini[row, plot_, :], 
                    color=colormap(normalize(share_)), 
                    linestyle='-' if row != 0 else '--')
    axes[i].set_title(percentile_labels[i])

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(shares)
fig.tight_layout()
colorbar = plt.colorbar(scalarmappaple, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
colorbar.set_label(r'$\alpha$', labelpad=1, fontsize=FONT_CONFIG.get("axes.labelsize", 30))
colorbar.ax.tick_params(labelsize=FONT_CONFIG.get("xtick.labelsize", 28))
plt.savefig(f'{path_figures}/ginis.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 3: Conglomerate Sizes (sizes.pdf)
# ============================================================================
print("Creating sizes.pdf...")
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 11))

for row, share_ in enumerate(shares):
    if row == 0:
        continue
    axes[0].plot(mean_members_[row, :], color=colormap(normalize(share_)))
    axes[1].plot(mean_conglomerates[row, :], color=colormap(normalize(share_)))

axes[0].set_title('Mean Conglomerate Size')
axes[1].set_title('Mean Conglomerate Count')

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(shares)
fig.tight_layout()
colorbar = plt.colorbar(scalarmappaple, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
colorbar.set_label(r'$\alpha$', labelpad=1, fontsize=FONT_CONFIG.get("axes.labelsize", 30))
colorbar.ax.tick_params(labelsize=FONT_CONFIG.get("xtick.labelsize", 28))
plt.savefig(f'{path_figures}/sizes.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 4: Persistence Quantiles (percentile_quantiles.pdf)
# ============================================================================
print("Creating percentile_quantiles.pdf...")
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 11))

for plot_ in range(3):
    axes[plot_].fill_between(x=shares, 
                           y1=mean_persistence_quantiles[plot_][:, 0], 
                           y2=mean_persistence_quantiles[plot_][:, -1], 
                           color="tab:blue", alpha=0.3)
    axes[plot_].fill_between(shares, 
                           y1=mean_persistence_quantiles[plot_][:, 1], 
                           y2=mean_persistence_quantiles[plot_][:, -2], 
                           color="tab:blue", alpha=0.3)
    axes[plot_].fill_between(shares, 
                           y1=mean_persistence_quantiles[plot_][:, 2], 
                           y2=mean_persistence_quantiles[plot_][:, -3], 
                           color="tab:blue", alpha=0.3)
    axes[plot_].plot(shares, mean_persistence_quantiles[plot_][:, 3], color="tab:blue")

axes[0].set_title('10\\%')
axes[1].set_title('90\\%')
axes[2].set_title('Maximum')

fig.tight_layout()
fig.text(0.5, 0.04, r'$\alpha$', ha='center', va='center')
plt.savefig(f'{path_figures}/percentile_quantiles.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 5: Estimates (ests.pdf)
# ============================================================================
print("Creating ests.pdf...")
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))

for row, share_ in enumerate(shares):
    if row == 0:
        continue
    est_level = mean_ests[row]
    axes[0].plot(est_level[:steps - ramp, 1], color=colormap(normalize(share_)))
    axes[1].plot(est_level[:steps - ramp:, 0], color=colormap(normalize(share_)))

axes[0].set_title(r'$\beta_1$')
axes[1].set_title(r'$\beta_2$')

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(shares)
fig.tight_layout()
colorbar = plt.colorbar(scalarmappaple, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
colorbar.set_label(r'$\alpha$', labelpad=1, fontsize=FONT_CONFIG.get("axes.labelsize", 30))
colorbar.ax.tick_params(labelsize=FONT_CONFIG.get("xtick.labelsize", 28))
plt.savefig(f'{path_figures}/ests.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 6: Mobility Ridge Plot (mobility.pdf)
# ============================================================================
print("Creating mobility.pdf...")
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 11))

n_distributions = len(ranges)

# Parameters for the ridge plot
overlap = 0.8  # Adjust this to change the overlap between distributions
offset = 0.8  # Vertical spacing between distributions

max_density = 0
for i, share in enumerate(shares):
    color = colormap(normalize(share))

    # For the first axis (ranges)
    row = ranges[i].reshape(-1)
    density = gaussian_kde(row)
    x = np.linspace(min(row), max(row), 1000)
    y = density(x)

    max_density = max(max_density, max(y))

    # Scale the density
    y = y / y.max() * overlap

    # Add the offset
    y = y + i * offset

    axes[0].fill_between(x, y, i * offset, alpha=0.8, color=color)

    # For the second axis (sds)
    row = sds[i].reshape(-1)
    density = gaussian_kde(row)
    x = np.linspace(min(row), max(row), 1000)
    y = density(x)

    # Scale the density
    y = y / y.max() * overlap

    # Add the offset
    y = y + i * offset

    axes[1].fill_between(x, y, i * offset, alpha=0.8, color=color)

# Remove y-ticks
axes[0].set_yticks([])
axes[1].set_yticks([])

# Set labels and title
axes[0].set_title('Range')
axes[1].set_title('Standard Deviation')

# Add a colorbar centered beneath both subplots
sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.15)
cbar.set_label(r'$\alpha$', fontsize=FONT_CONFIG.get("axes.labelsize", 30))
# Make sure colorbar tick labels are sized correctly
cbar.ax.tick_params(labelsize=FONT_CONFIG.get("xtick.labelsize", 28))
plt.savefig(f'{path_figures}/mobility.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

print("✓ All plots regenerated successfully!")
print(f"✓ Plots saved to: {path_figures}")
print(f"✓ Font configuration used: {list(FONT_CONFIG.keys())}")