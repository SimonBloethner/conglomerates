import collaborative_growth
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from tqdm import tqdm
import os

is_local = os.getcwd().find('Simon') > 0

path_figures = '/Users/Simon/PycharmProjects/Collusion/figures' if is_local else 'conglomerate/figures'

np.random.seed(7)
markets = 100
firms_per_market = 100
total_firms = markets * firms_per_market
steps = 10
share = 0.9999
merge_thresh = 0.05
comparison = 4
break_thresh = 0.85
lookback = 50
proportional = False
shares = np.arange(0, 0.2, 0.01)

counterfactuals = 4

mean_quantiles = np.empty(shape=(shares.shape[0], 4, steps + 1))
mean_members_ = np.empty(shape=(shares.shape[0], steps))
mean_conglomerates = np.empty(shape=(shares.shape[0], steps))
mean_gini = np.empty(shape=(shares.shape[0], 5, steps + 1))

for trial, share in tqdm(enumerate(shares)):
    shares_quantiles = np.empty(shape=(counterfactuals, 4, steps + 1))
    gini_quantiles = np.empty(shape=(steps + 1, 5, counterfactuals))
    mean_members__ = np.empty(shape=(steps, counterfactuals))
    mean_conglomerates_ = np.empty(shape=(steps, counterfactuals))
    for experiment in range(counterfactuals):
        params_ = [markets, firms_per_market, steps, share, total_firms, merge_thresh, comparison, break_thresh,
                   proportional, lookback]
        res = collaborative_growth.model(params=params_)
        mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, max_shares, market_share, hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks = res
        shares_quantiles[experiment, :, :] = np.quantile(market_share, q=[0.5, 0.9, 0.99, 1], axis=(0, 2))
        gini_quantiles[:, :, experiment] = np.quantile(gini_coefficient, q=[0.1, 0.25, 0.5, 0.75, 0.9], axis=0).T
        mean_members__[:, experiment] = mean_members
        mean_conglomerates_[:, experiment] = num_cong

    mean_quantiles[trial, :, :] = shares_quantiles.mean(0)
    mean_gini[trial, :, :] = gini_quantiles.mean(axis=2).T
    mean_members_[trial, :] = mean_members__.mean(axis=1)
    mean_conglomerates[trial, :] = mean_conglomerates_.mean(axis=1)

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 8))

normalize = mcolors.Normalize(vmin=shares.min(), vmax=shares.max())
colormap = cm.viridis
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
colorbar.set_label('Pooling Rate', labelpad=1)
# plt.savefig('{}/{}'.format(path_figures, 'shares.png'))


fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 8))
for plot_ in range(5):
    for row, share_ in enumerate(shares):
        axes[plot_].plot(mean_gini[row, plot_, :], color=colormap(normalize(share_)), linestyle='-' if row != 0 else '--')

axes[0].set_title('50%')
axes[1].set_title('90%')
axes[2].set_title('99%')
axes[3].set_title('Maximum')
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(shares)
fig.tight_layout()
colorbar = plt.colorbar(scalarmappaple, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
colorbar.set_label('Pooling Rate', labelpad=1)
# plt.savefig('{}/{}'.format(path_figures, 'ginis.png'))


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
for row, share_ in enumerate(shares):
    if row == 0:
        continue
    axes[0].plot(mean_members_[row, :], color=colormap(normalize(share_)), linestyle='-' if row != 0 else '--')
    axes[1].plot(mean_conglomerates[row, :], color=colormap(normalize(share_)), linestyle='-' if row != 0 else '--')

axes[0].set_title('Mean Members')
axes[1].set_title('Mean Conglomerate Size')
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(shares)
fig.tight_layout()
colorbar = plt.colorbar(scalarmappaple, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
colorbar.set_label('Pooling Rate', labelpad=1)
# plt.savefig('{}/{}'.format(path_figures, 'sizes.png'))
