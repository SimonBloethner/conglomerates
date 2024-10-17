import programs.collaborative_growth as collaborative_growth
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from tqdm import tqdm
import os
import warnings

import matplotlib as mpl

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})


is_mac = os.getcwd().find('Simon') > 0
is_windows = os.getcwd().find('nomis') > 0

if is_mac:
    path_figures = '/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE/latex/figures'
    path_outdata = '/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE/outdata'
elif is_windows:
    path_figures = 'C:\\Users\\nomis\\PycharmProjects\\conglomerates\\conglomerates\\figures'
else:
    path_figures = 'conglomerate/figures'

np.random.seed(7)
markets = 100
firms_per_market = 100
total_firms = markets * firms_per_market
steps = 1000
share = 0.9999
merge_thresh = 0.05
comparison = 4
break_thresh = 0.85
lookback = 50
ramp = 10
proportional = False
shares = np.arange(0, 0.52, 0.02)

plt.rcParams.update({'font.size': 12})

normalize = mcolors.Normalize(vmin=shares.min(), vmax=shares.max())
colormap = cm.viridis

counterfactuals = 10
persistence_quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

ranks_ = np.zeros((steps, markets, firms_per_market, counterfactuals, shares.shape[0]))
mean_quantiles = np.empty(shape=(shares.shape[0], 4, steps + 1))
mean_members_ = np.empty(shape=(shares.shape[0], steps))
mean_conglomerates = np.empty(shape=(shares.shape[0], steps))
mean_gini = np.empty(shape=(shares.shape[0], 5, steps + 1))
mean_ests = np.full([shares.shape[0], steps, 3], np.nan)
mean_persistence = [np.zeros((shares.shape[0], steps + 1)) for _ in range(3)]
mean_persistence_quantiles = [np.zeros((shares.shape[0], len(persistence_quantiles))) for _ in range(3)]

for trial, share in tqdm(enumerate(shares)):
    shares_quantiles = np.empty(shape=(counterfactuals, 4, steps + 1))
    gini_quantiles = np.empty(shape=(steps + 1, 5, counterfactuals))
    mean_members__ = np.empty(shape=(steps, counterfactuals))
    mean_conglomerates_ = np.empty(shape=(steps, counterfactuals))
    mean_ests_ = np.empty(shape=(steps, 3, counterfactuals))
    mean_persistence_ = [np.zeros(steps + 1) for _ in range(3)]
    mean_persistence_quantiles_ = [np.zeros((counterfactuals, len(persistence_quantiles))) for _ in range(3)]
    for experiment in range(counterfactuals):
        params_ = [markets, firms_per_market, steps, share, total_firms, merge_thresh, comparison, break_thresh,
                   proportional, lookback]
        res = collaborative_growth.model(params=params_)
        mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, max_shares, market_share, hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks = res
        ranks_[:, :, :, experiment, trial] = ranks
        temp = np.quantile(market_share, q=[0.5, 0.9, 0.99, 1], axis=2)
        shares_quantiles[experiment, :, :] = temp.mean(axis=1)
        gini_quantiles[:, :, experiment] = np.quantile(gini_coefficient, q=[0.1, 0.25, 0.5, 0.75, 0.9], axis=0).T
        mean_members__[:, experiment] = mean_members
        mean_conglomerates_[:, experiment] = num_cong
        for _ in range(3):
            mean_persistence_quantiles_[_][experiment, :] = np.quantile(percentile_ranks[_], q=persistence_quantiles)
            count = np.bincount(percentile_ranks[_].reshape(-1))
            temp = np.zeros(steps + 1)
            temp[:percentile_ranks[_].reshape(-1).max() + 1] = temp[:percentile_ranks[_].reshape(-1).max() + 1] + count
            mean_persistence_[_] = mean_persistence_[_] + temp

        for step in range(ramp, steps):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.RankWarning)
                coeffs = np.polyfit(avg_shares[step][:, 0], avg_shares[step][:, 1], 2)
            mean_ests_[step - ramp, :, experiment] = coeffs

    mean_quantiles[trial, :, :] = shares_quantiles.mean(0)
    mean_gini[trial, :, :] = gini_quantiles.mean(axis=2).T
    mean_members_[trial, :] = mean_members__.mean(axis=1)
    mean_conglomerates[trial, :] = mean_conglomerates_.mean(axis=1)
    mean_ests[trial, :] = mean_ests_.mean(axis=2)
    for _ in range(3):
        mean_persistence_quantiles[_][trial, :] = mean_persistence_quantiles_[_].mean(axis=0)
        mean_persistence[_][trial, :] = mean_persistence_[_] / counterfactuals

ranges = []
sds = []
for share in tqdm(range(shares.shape[0])):
    min_rank = np.min(ranks_[:, :, :, :, share], axis=0)
    max_rank = np.max(ranks_[:, :, :, :, share], axis=0)

    ranges_ = max_rank - min_rank
    ranges.append(ranges_)
    sds.append(np.std(ranks_[:, :, :, :, share], axis=0))


fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 8))

for plot_ in range(4):
    for row, share_ in enumerate(shares):
        axes[plot_].plot(mean_quantiles[row, plot_, :], color=colormap(normalize(share_)), linestyle='-' if row != 0 else '--')

axes[0].set_title('50%')
axes[1].set_title('90%')
axes[2].set_title('99%')
axes[3].set_title('Maximum')
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(shares)
fig.tight_layout()
colorbar = plt.colorbar(scalarmappaple, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
colorbar.set_label(r'$\alpha$', labelpad=1)
plt.savefig('{}/{}'.format(path_figures, 'shares.pdf'), format='pdf', dpi=300)

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 8))
for plot_ in range(5):
    for row, share_ in enumerate(shares):
        axes[plot_].plot(mean_gini[row, plot_, :], color=colormap(normalize(share_)), linestyle='-' if row != 0 else '--')

axes[0].set_title('10%')
axes[1].set_title('25%')
axes[2].set_title('50%')
axes[3].set_title('75%')
axes[4].set_title('90%')
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(shares)
fig.tight_layout()
colorbar = plt.colorbar(scalarmappaple, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
colorbar.set_label(r'$\alpha$', labelpad=1)
plt.savefig('{}/{}'.format(path_figures, 'ginis.pdf'), format='pdf', dpi=300)


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
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
colorbar.set_label(r'$\alpha$', labelpad=1)
plt.savefig('{}/{}'.format(path_figures, 'sizes.pdf'), format='pdf', dpi=300)


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
for plot_ in range(3):
    axes[plot_].fill_between(x=shares, y1=mean_persistence_quantiles[plot_][:, 0], y2=mean_persistence_quantiles[plot_][:, -1], color="tab:blue", alpha=0.3)
    axes[plot_].fill_between(shares, y1=mean_persistence_quantiles[plot_][:, 1], y2=mean_persistence_quantiles[plot_][:, -2], color="tab:blue", alpha=0.3)
    axes[plot_].fill_between(shares, y1=mean_persistence_quantiles[plot_][:, 2], y2=mean_persistence_quantiles[plot_][:, -3], color="tab:blue", alpha=0.3)
    axes[plot_].plot(shares, mean_persistence_quantiles[plot_][:, 3], color="tab:blue")
axes[0].set_title('10 %')
axes[1].set_title('90 %')
axes[2].set_title('Maximum')
fig.tight_layout()
fig.text(0.5, 0.04, r'$\alpha$', ha='center', va='center', fontsize=14)
plt.savefig('{}/{}'.format(path_figures, 'percentile_quantiles.pdf'), format='pdf', dpi=300)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

for row, share_ in enumerate(shares):
    if row == 0:
        continue

    axes[0].plot(mean_ests[row, :steps - ramp, 1], color=colormap(normalize(share_)))
    axes[1].plot(mean_ests[row, :steps - ramp:, 0], color=colormap(normalize(share_)))

axes[0].set_title(r'$\beta_1$')
axes[1].set_title(r'$\beta_2$')
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(shares)
fig.tight_layout()
colorbar = plt.colorbar(scalarmappaple, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
colorbar.set_label(r'$\alpha$', labelpad=1)
plt.savefig('{}/{}'.format(path_figures, 'ests.pdf'), format='pdf', dpi=300)


fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12, 8))

# Assuming 'ranges' is a list of your data ranges
# and 'shares' is a list of corresponding share values
n_distributions = len(ranges)

# Parameters for the ridge plot
overlap = 0.8  # Adjust this to change the overlap between distributions
offset = 0.8  # Vertical spacing between distributions

max_density = 0
for i, share in enumerate(shares):
    max_density = max(max_density, max(y))
    color = colormap(normalize(share))

    row = ranges[i].reshape(-1)
    density = gaussian_kde(row)
    x = np.linspace(min(row), max(row), 1000)
    y = density(x)

    # Scale the density
    y = y / y.max() * overlap

    # Add the offset
    y = y + i * offset

    axes[0].fill_between(x, y, i * offset, alpha=0.8, color=color)

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
cbar.set_label(r'$\alpha$')
plt.show()
plt.savefig('{}/{}'.format(path_figures, 'mobility.pdf'), format='pdf', dpi=300)

np.savez('{}/{}'.format(path_outdata, 'mean_persistence.npz'), *mean_persistence)
np.savez('{}/{}'.format(path_outdata, 'mean_persistence_quantiles.npz'), *mean_persistence_quantiles)
np.savez('{}/{}'.format(path_outdata, 'mean_gini.npz'), *mean_gini)
np.savez('{}/{}'.format(path_outdata, 'mean_quantiles.npz'), *mean_quantiles)
np.save('{}/{}'.format(path_outdata, 'mean_members_.npz'), mean_members_)
np.savez('{}/{}'.format(path_outdata, 'mean_conglomerates.npz'), *mean_conglomerates)
np.savez('{}/{}'.format(path_outdata, 'mean_ests.npz'), *mean_ests)
np.save('{}/{}'.format(path_outdata, 'ranks.npy'), ranks_)
