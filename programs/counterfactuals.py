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
steps = 1000
share = 0.9999
merge_thresh = 0.05
comparison = 4
break_thresh = 0.85
lookback = 50
proportional = False
shares = np.arange(0, 0.2, 0.01)

counterfactuals = 5

mean_quantiles = np.empty(shape=(shares.shape[0], 4, steps + 1))

for trial, share in tqdm(enumerate(shares)):
    quantiles = np.empty(shape=(counterfactuals, 4, steps + 1))
    for experiment in range(counterfactuals):
        params_ = [markets, firms_per_market, steps, share, total_firms, merge_thresh, comparison, break_thresh,
                   proportional, lookback]
        res = collaborative_growth.model(params=params_)
        mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, max_shares, market_share, hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks = res
        quantiles[experiment, :, :] = np.quantile(market_share, q=[0.5, 0.9, 0.99, 1], axis=(0, 2))

    mean_quantiles[trial, :, :] = quantiles.mean(0)


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

plt.savefig('{}/{}'.format(path_figures, 'shares.png'))
