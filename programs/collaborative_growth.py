import numpy as np
from scipy.stats import random_correlation


class Firm:
    def __init__(self, market, number, steps):
        self.id = number
        self.home_market = market
        self.states = np.ones(steps + 1)
        self.markets = [market]
        self.conglomerate = [self.id]
        self.rank = np.ones(steps + 1)


def logistic_cost(x, k, x_0):
    y = 1 / (1 + np.exp(-k * (x - x_0)))
    return y


def cost(size, progression, degree=2):
    if progression == 'linear':
        costs = size
    elif progression == 'polynomial':
        costs = size ** degree
    elif progression == 'logistic':
        costs = logistic_cost(size)
    else:
        costs = np.exp(size)

    return costs


def format_func(value, tick_number):
    return f'{value:.2f}'


def exit_(firms, firm):
    partners = firm.conglomerate.copy()
    partners.remove(firm.id)
    for partner in partners:
        firms[partner].markets.remove(firm.home_market)
        firms[partner].conglomerate.remove(firm.id)

    firm.conglomerate = [firm.id]
    firm.markets = [firm.home_market]


def model(params):
    min_mu = 0.01
    max_mu = 0.1
    min_sig = 0.01
    max_sig = 0.05

    min_bound = np.array([min_mu, min_sig])
    max_bound = np.array([max_mu, max_sig])

    mu_sig_corr = 0.7
    means = [0.1, 0.05]

    markets, firms_per_market, steps, share, total_firms, merge_thresh, comparison, break_thresh, proportional = params
    cov_mat = np.ones((2, 2)) * mu_sig_corr + np.diag(np.tile(1 - mu_sig_corr, 2))
    growth_vars = np.random.multivariate_normal(means, cov_mat, markets)
    growth_vars = growth_vars + np.abs(np.minimum(growth_vars.min(axis=0), 0))
    growth_vars = min_bound + (growth_vars / growth_vars.max(axis=0)) * (max_bound - min_bound)

    eigen_vals = np.random.uniform(0.1, 3, markets)
    eigen_vals = eigen_vals * markets / eigen_vals.sum()

    market_corr = random_correlation.rvs(tuple(eigen_vals), random_state=np.random.default_rng())

    # market_corr = np.diag(np.ones(markets))
    # market_corr[np.triu_indices(markets, k=1)] = np.random.uniform(0.1, 0.8, int(markets * (markets - 1) / 2))
    # market_corr = market_corr + market_corr.T - np.diag(np.ones(markets))
    market_cov = np.outer(growth_vars[:, 1], growth_vars[:, 1]) * market_corr

    realizations = np.random.multivariate_normal(growth_vars[:, 0], market_cov, size=(steps, firms_per_market)) + 1
    markets_structure = np.array([[x, y] for x in range(markets) for y in range(firms_per_market)])

    ids = np.arange(markets * firms_per_market).reshape(markets, firms_per_market)

    firms = [Firm(market=market, number=ids[market, firm], steps=steps) for market in range(markets) for firm in
             range(firms_per_market)]
    conglomerates = []
    for step in range(steps):
        draws = np.where(np.random.uniform(0, 1, total_firms) < merge_thresh)[0]

        if draws.shape[0] > 0:
            for firm in draws:
                active_markets = firms[firm].markets
                if len(active_markets) == markets:
                    continue

                target_markets = markets_structure[np.logical_not(np.in1d(markets_structure[:, 0], active_markets)), :]
                target = target_markets[np.random.choice(np.arange(target_markets.shape[0]), 1)]
                target = np.where((markets_structure == target).all(axis=1))[0][0]

                if not np.isin(firms[target].markets, firms[firm].markets).any():
                    joint_markets = firms[target].markets + firms[firm].markets
                    joint_markets.sort()

                    conglomerate = firms[target].conglomerate + firms[firm].conglomerate
                    conglomerate.sort()

                    for partner in conglomerate:
                        firms[partner].markets = joint_markets.copy()
                        firms[partner].conglomerate = conglomerate.copy()

        period_conglomerates = list(
            set(tuple(lst) for lst in [firm.conglomerate for firm in firms if len(firm.conglomerate) > 1]))
        conglomerates.append(period_conglomerates)

        solo = [firm.id for firm in firms if len(firm.conglomerate) == 1]

        for conglomerate in period_conglomerates:
            indices = markets_structure[conglomerate, :]
            returns = realizations[step, indices[:, 1], indices[:, 0]]
            states = np.array([firms[firm].states[step] for firm in conglomerate])
            gains = states * returns - states
            if proportional:
                pool = (gains * share).sum() * (1 - logistic_cost(len(conglomerate), k=1, x_0=markets))
            else:
                pool = (gains * share).sum() - states.sum() * logistic_cost(len(conglomerate), k=1,
                                                                            x_0=markets * 0.4)  # len(conglomerate) / markets

            returns = (1 - share) * gains + pool / len(
                conglomerate) if share > 0 else gains  # Equal distribution of pool contents.

            for partner, firm in enumerate(conglomerate):
                firms[firm].states[step + 1] = firms[firm].states[step] + returns[partner]
                if firms[firm].states[step + 1] < 0:
                    firms[firm].states[step + 1] = 1
                    exit_(firms, firms[firm])

        for firm in solo:
            index = markets_structure[firm, :]
            firms[firm].states[step + 1] = firms[firm].states[step] * realizations[step, index[1], index[0]]

        results = np.array([firm.states[step] for firm in firms]).reshape(firms_per_market, markets)
        ranks = np.argsort(results, axis=0)

        for firm in firms:
            firm.rank[step] = ranks[tuple(markets_structure[firm.id])]

        if step > comparison:
            for firm in firms:
                if len(firm.conglomerate) > 1:
                    # if firm.states[step] / firm.states[step - comparison] < break_thresh:
                    if firm.rank[step] - firm.rank[step - comparison] < 3:
                        exit_(firms, firm)

    results = np.array([firm.states for firm in firms]).T

    num_cong = []
    members = []
    for conglomerate in conglomerates:
        num_cong.append(len(conglomerate))
        members.append([len(members) for members in conglomerate])

    mean_members = np.array([np.mean(member) for member in members])

    quantiles_members = np.array([np.quantile(member, q=[0.1, 0.25, 0.5, 0.75, 0.9]) for member in members])

    market_share = np.array([results[:,
                             np.arange(market * firms_per_market, (market + 1) * firms_per_market)] / results[:,
                                                                                                      np.arange(
                                                                                                          market * firms_per_market,
                                                                                                          (
                                                                                                                      market + 1) * firms_per_market)].sum(
        axis=1).reshape(steps + 1, 1) for market in range(markets)])
    # shape = markets x steps + 1 x firms_per_market
    mean_share = market_share.mean(axis=2)
    quantiles_shares = np.quantile(market_share, q=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99], axis=2)
    max_shares = np.max(market_share, axis=2)

    squared_market_share = market_share ** 2
    hhi = np.sum(squared_market_share, axis=2)

    sorted_market_share = np.sort(market_share, axis=2)
    cum_market_share = np.cumsum(sorted_market_share, axis=2)
    sums = np.sum(sorted_market_share, axis=2)
    sums = sums[..., np.newaxis]
    Lorenz_curve = cum_market_share / sums
    area_under_curve = np.trapz(y=Lorenz_curve, axis=2, dx=1 / firms_per_market)
    gini_coefficient = 1 - 2 * area_under_curve

    ranks = np.array([np.argsort(market_share[:, step_, :], axis=1) for step_ in range(steps)])

    percentile_thresh = firms_per_market * np.array([0.1, 0.9, ranks.max() / firms_per_market])

    percentile_ranks = [(ranks >= thresh).sum(axis=0) for thresh in percentile_thresh]

    avg_shares = []
    avg_ranks = []
    for period in range(steps):
        avg_share = []
        avg_rank = []
        for conglomerate in conglomerates[period]:
            to_append = [len(conglomerate), market_share[:, period, :][np.where(np.isin(ids, conglomerate))].mean()]
            avg_share.append(to_append)
            to_append = [len(conglomerate), ranks[period, :, :][np.where(np.isin(ids, conglomerate))].mean()]
            avg_rank.append(to_append)

        avg_shares.append(np.array(avg_share))
        avg_ranks.append(np.array(avg_rank))

    model_results = [mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, max_shares, market_share,
                     hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks]

    return model_results
