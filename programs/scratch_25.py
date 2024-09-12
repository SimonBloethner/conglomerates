import numpy as np
from PricingModels import select_action_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm


steps = 10000
prices = np.zeros((steps, 100))
profits = np.zeros((steps, 100))
sizes = np.ones((steps, 100))
market_share = np.tile(1/100, (steps, 100))
market_growth = np.ones(steps)

def demand(p, share):
    n_firms = p.shape[0]
    minus_i = (np.ones((n_firms, n_firms)) - np.eye(n_firms)).astype(bool)
    price = np.tile(p, (n_firms, 1))
    minus_i = price[minus_i].reshape(n_firms, n_firms - 1)
    minus_i = np.max(minus_i, axis=1)

    d = 1 - p * (1 - share) + 0.5 * minus_i
    return d


def profit(p, share):
    pi = np.array(p) * demand(p, share)
    return pi


for step in tqdm(range(steps)):

    for agent in range(100):
        mask = np.ones(100, dtype=bool)
        mask[agent] = False
        p_ = np.max(prices[step - 1, mask])
        p__ = np.max(prices[step - 2, mask])
        state = [prices[step - 1, agent], p_, p__, market_share[step - 1, agent]]
        prices[step, agent] = select_action_matrix(state)

    profits[step, :] = profit(prices[step, :], share=market_share[step - 1, :]) * np.max(np.array([np.zeros(100), 1 - np.abs(np.random.normal(0, 0.01, 100))]), axis=0) * market_growth.prod() * market_share[step - 1, :]

    sizes[step, :] = sizes[step - 1, :] + sizes[step - 1, :] * np.log(1 + profits[step, :] / sizes[step - 1, :])
    market_share[step, :] = sizes[step, :] / np.sum(sizes[step, :], axis=0)
    market_growth[step] = sizes[step, :].sum() / sizes[step - 1, :].sum()

plt.plot(sizes)
plt.figure()
plt.plot(np.log(1 + profits[1:, :] / sizes[:-1, :]))
plt.figure()
plt.plot(market_share)
