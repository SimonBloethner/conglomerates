import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import collaborative_growth
import os

is_local = os.getcwd().find('Simon') > 0

path_figures = '/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE/latex/figures' if is_local else 'conglomerate/figures'

np.random.seed(7)
markets = 100
firms_per_market = 100
total_firms = markets * firms_per_market
steps = 1000
share = 0.9999
merge_thresh = 0.05
comparison = 2
break_thresh = 0.85
lookback = 50
proportional = False
shares = np.arange(0, 1.1, 0.1)
shares = np.array([0.2])


quantiles = np.empty(shape=(shares.shape[0], 4, steps + 1))

for trial, share in enumerate(shares):
    params_ = [markets, firms_per_market, steps, share, total_firms, merge_thresh, comparison, break_thresh, proportional, lookback]
    res = collaborative_growth.model(params=params_)

    mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, max_shares, market_share, hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks = res
    quantiles[trial, :, :] = np.quantile(market_share, q=[0.5, 0.9, 0.99, 1], axis=(0, 2))

percentiles = ['10', '90', 'Maximum']

ramp = 10
coeffs = np.zeros((steps - ramp, 3))
for step in range(ramp, steps):
    coeffs[step - ramp, :] = np.polyfit(avg_shares[step][:, 0], avg_shares[step][:, 1], 2)


fn = np.poly1d(coeffs[-1])
predictions = fn(avg_shares[-1][:, 0])

sign = np.sign(coeffs)
log_coeffs = np.log2(np.abs(coeffs))
log_coeffs = log_coeffs * sign

sorted_index = np.argsort(avg_shares[-1][:, 0])

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7, 7))

axes[0][0].plot(mean_members)
axes[0][0].set_title('(a) Mean # of members')
axes[0][1].plot(quantiles_members)
axes[0][1].set_title('(b) Quantiles of members')
axes[0][2].plot(num_cong)
axes[0][2].set_title('(c) # of conglomerates')
axes[1][0].scatter(x=avg_shares[-1][:, 0], y=avg_shares[-1][:, 1], label='Mean market share')
axes[1][0].plot(avg_shares[-1][:, 0][sorted_index], predictions[sorted_index], color='red')
axes[1][0].set_title('(d) Size vs. Mean share')
axes[1][0].set_yscale('log', base=2)
axes[1][1].scatter(x=avg_ranks[-1][:, 0], y=avg_ranks[-1][:, 1])
axes[1][1].set_title('(e) Size vs. Mean Rank')

ax1 = axes[1][2]
ax2 = ax1.twinx()

# Plot data on the first y-axis
color1 = 'tab:blue'
ax1.plot(coeffs[:, 0], color=color1, label=r'$\beta_2$')
ax1.tick_params(axis='y', labelcolor=color1)

# Plot data on the second y-axis
color2 = 'tab:orange'
ax2.plot(coeffs[:, 1], color=color2, label=r'$\beta_1$')
ax2.tick_params(axis='y', labelcolor=color2)

# Set title and combine legends
ax1.set_title('(f) Estimates')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', prop={'size': 8})

color_palette = cm.tab10(range(len(percentile_ranks)))
for perc in range(len(percentile_ranks)):
    axes[2][0].hist(percentile_ranks[perc].flat, bins=firms_per_market, color=color_palette[perc], alpha=0.7, label='{} {}'.format(percentiles[perc], '%' if perc != 2 else ''))
axes[2][0].legend(loc='upper center')
axes[2][0].set_title('(g) Rank Persistence')
axes[2][1].plot(max_shares)
axes[2][1].set_title('(h) Max market share')
colors = sns.color_palette("husl", n_colors=markets)
for market in range(markets):
    axes[2][2].plot(gini_coefficient[market, :], color=colors[market])
axes[2][2].set_title('(i) Gini coefficient')
fig.tight_layout()

plt.savefig('{}/{}'.format(path_figures, 'overview.eps'), format='eps')

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 8))

normalize = mcolors.Normalize(vmin=0, vmax=1)
colormap = cm.viridis
for plot_ in range(4):
    for row, share_ in enumerate(shares):
        axes[plot_].plot(quantiles[row, plot_, :], color=colormap(normalize(share_)), linestyle='-' if row != 0 else '--')

axes[0].set_title('50%')
axes[1].set_title('90%')
axes[2].set_title('99%')
axes[3].set_title('Maximum')
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(shares)
fig.tight_layout()
colorbar = plt.colorbar(scalarmappaple, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
colorbar.set_label('Pooling Rate', labelpad=1)

plt.show()
