import numpy as np
from tqdm import tqdm
import os

np.random.seed(1)

path_ = os.getcwd()
local = path_.find('Simon') > 0
if local:
    path_ = '/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE'
else:
    path_ = 'conglomerate'


class Firm:
    def __init__(self, market, number, market_id, steps, lookback):
        self.id = number
        self.market_id = market_id
        self.home_market = market
        self.states = np.ones(steps + 1)
        self.markets = [market]
        self.conglomerate = [self.id]
        self.outside_profits = np.ones(lookback)
        self.entered = None
        self.conglomerate_id = None


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
    firm.entered = None
    firm.conglomerate_id = None


def model(params):
    markets, firms_per_market, steps, share, total_firms, merge_thresh, comparison, break_thresh, proportional, lookback = params
    prices = np.zeros((steps, firms_per_market, markets))
    increments = np.zeros((steps, firms_per_market, markets))
    profits = np.ones((steps, firms_per_market, markets))
    sizes = np.ones((steps, firms_per_market, markets))
    market_share = np.tile(1 / firms_per_market, (steps, firms_per_market, markets))
    lower_bound = 0.9
    upper_bound = 1.1
    costs = np.random.lognormal(0, 1, size=(steps, firms_per_market, markets)) * 0

    markets_structure = np.array([[x, y] for x in range(markets) for y in range(firms_per_market)])

    market_growth = np.ones((steps, markets))

    # merge_thresh = merge_thresh if share > 0 else 0     # Avoids going into the merging process if there is no pooling.

    ids = np.arange(markets * firms_per_market).reshape(markets, firms_per_market)

    firms = [Firm(market=market, number=ids[market, firm], market_id=firm, steps=steps, lookback=lookback) for market in range(markets) for firm in range(firms_per_market)]
    all_time_conglomerates = []
    conglomerates = {}
    for step in tqdm(range(steps)):
    # for step in range(steps):
        to_del = []
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
                    home_id = firms[firm].conglomerate_id
                    cong_id = firms[target].conglomerate_id

                    joint_markets = firms[target].markets + firms[firm].markets
                    joint_markets.sort()

                    conglomerate = firms[target].conglomerate + firms[firm].conglomerate
                    conglomerate.sort()

                    if len(conglomerate) > 2:
                        if home_id is None or cong_id is None:
                            conglomerate_id = home_id if cong_id is None else cong_id
                            true_pool = conglomerates[conglomerate_id]['pool']
                            synth_pool = np.zeros(lookback).astype(float)
                            indices = markets_structure[conglomerate][:, [1, 0]]
                            for enumer, synth in enumerate(range(max(step - lookback, 0), step)):
                                returns = increments[synth, indices[:, 1], indices[:, 0]]
                                states = np.array([firms[firm].states[synth] for firm in conglomerate])
                                gains = states * (1 + returns) - states
                                if proportional:
                                    pool = (gains * share).sum() * (1 - logistic_cost(len(conglomerate), k=1, x_0=markets))
                                else:
                                    pool = (gains * share).sum() - states.sum() * logistic_cost(len(conglomerate), k=1, x_0=markets * 0.4)  # len(conglomerate) / markets

                                synth_pool[enumer] = pool

                            if synth_pool.sum() < true_pool.sum():
                                continue
                        else:
                            true_pool0 = conglomerates[home_id]['pool']
                            true_pool1 = conglomerates[cong_id]['pool']
                            synth_pool = np.zeros(lookback).astype(float)
                            indices = markets_structure[conglomerate][:, [1, 0]]
                            for enumer, synth in enumerate(range(max(step - lookback, 0), step)):
                                returns = increments[synth, indices[:, 1], indices[:, 0]]
                                states = np.array([firms[firm].states[synth] for firm in conglomerate])
                                gains = states * (1 + returns) - states
                                if proportional:
                                    pool = (gains * share).sum() * (1 - logistic_cost(len(conglomerate), k=1, x_0=markets))
                                else:
                                    pool = (gains * share).sum() - states.sum() * logistic_cost(len(conglomerate), k=1, x_0=markets * 0.4)  # len(conglomerate) / markets

                                synth_pool[enumer] = pool

                            if synth_pool.sum() < true_pool0.sum() or synth_pool.sum() < true_pool1.sum():
                                continue

                    if cong_id is None and home_id is None:
                        existing = np.array(list(conglomerates.keys()))
                        if len(existing) == 0:
                            new_id = 0
                        else:
                            all_ = np.arange(len(conglomerates.keys()))
                            lowest_new = all_[np.logical_not(np.isin(all_, existing))]
                            if lowest_new.shape[0] == 0:
                                new_id = np.max(all_) + 1
                            else:
                                new_id = np.min(lowest_new)

                    for partner in conglomerate:
                        firms[partner].markets = joint_markets.copy()
                        firms[partner].conglomerate = conglomerate.copy()

                    if home_id is not None and cong_id is not None:
                        conglomerates[cong_id] = {'firms': conglomerate.copy(), 'pool': synth_pool}

                        for member in conglomerate:
                            firms[member].conglomerate_id = cong_id
                            firms[firm].entered = step

                        del conglomerates[home_id]

                    elif home_id is None and cong_id is not None:
                        conglomerates[cong_id]['firms'] = conglomerate.copy()
                        firms[firm].conglomerate_id = cong_id
                        firms[firm].entered = step

                    elif home_id is not None and cong_id is None:
                        conglomerates[home_id]['firms'] = conglomerate.copy()
                        firms[target].conglomerate_id = home_id
                        firms[target].entered = step

                    elif home_id is None and cong_id is None:
                        firms[firm].conglomerate_id = new_id
                        firms[target].conglomerate_id = new_id
                        firms[firm].entered = step
                        firms[target].entered = step
                        conglomerates[new_id] = {'firms': conglomerate.copy(), 'pool': np.zeros(lookback).astype(float)}

        period_conglomerates = list(set(tuple(lst) for lst in [firm.conglomerate for firm in firms if len(firm.conglomerate) > 1]))
        all_time_conglomerates.append(period_conglomerates)

        solo = [firm.id for firm in firms if len(firm.conglomerate) == 1]

        prices[step, :, :] = price_opt(market_share[step - 1, :, :], cost=costs[step, :, :])
        profits[step, :, :] = profit(prices[step, :, :], market_share[step - 1, :, :], cost=costs[step, :, :]) # * market_growth.prod(axis=0) * market_share[step - 1, :, :]

        # reinvestment = (profits[step, :, :] / sizes[step - 1, :, :]) + 1

        # normalized_reinvestment = (reinvestment - np.min(reinvestment, axis=0)) / (np.max(reinvestment, axis=0) - np.min(reinvestment, axis=0))  # Assumption: axis=0 means that there are normalizations for each market, not across all firms!
        # if np.isnan(normalized_reinvestment).any():
        #     print('!')
        #
        # means = lower_bound + normalized_reinvestment * (upper_bound - lower_bound)

        reinvestment = (profits[step, :, :] / sizes[step - 1, :, :])
        sds = 0.05 * reinvestment
        means = share + 1
        if (sds < 0).any():
            print('!')
        increments[step, :, :] = np.random.normal(means, sds)
        increments[increments < -1] = -0.99

        for conglomerate_ in conglomerates.keys():
            conglomerate = conglomerates[conglomerate_]['firms']
            indices = markets_structure[conglomerate, :]
            inc = increments[step, indices[:, 1], indices[:, 0]]
            states = sizes[step - 1, indices[:, 1], indices[:, 0]]
            gains = states * inc - states
            if proportional:
                pool = (gains * share).sum() * (1 - logistic_cost(len(conglomerate), k=1, x_0=markets))
            else:
                pool = (gains * share).sum() - states.sum() * logistic_cost(len(conglomerate), k=1, x_0=markets * 0.4)  # len(conglomerate) / markets

            returns = ((1 - share) * gains + pool / len(conglomerate)) if share > 0 else gains  # Equal distribution of pool contents.

            del_firms = []
            for partner, firm in enumerate(conglomerate):
                tup = tuple([markets_structure[firm][1], markets_structure[firm][0]])
                firms[firm].outside_profits[step % lookback] = inc[partner]
                firms[firm].states[step + 1] = firms[firm].states[step] + returns[partner]
                sizes[(step, ) + tup] = sizes[(step - 1, ) + tup] + returns[partner]
                if sizes[(step, ) + tup] < 0:
                    firms[firm].states[step + 1] = 1
                    sizes[(step,) + tup] = 1
                    del_firms.append(firm)
                    exit_(firms, firms[firm])

            conglomerates[conglomerate_]['pool'][step % lookback] = pool
            for firm in del_firms:
                conglomerates[conglomerate_]['firms'].remove(firm)

            if len(conglomerates[conglomerate_]['firms']) < 2:
                to_del.append(conglomerate_)

        for firm in solo:
            firms[firm].outside_profits[step % lookback] = increments[step, firms[firm].market_id, firms[firm].home_market]
            firms[firm].states[step + 1] = firms[firm].states[step] * increments[step, firms[firm].market_id, firms[firm].home_market]
            sizes[step, firms[firm].market_id, firms[firm].home_market] = sizes[step - 1, firms[firm].market_id, firms[firm].home_market] * increments[step, firms[firm].market_id, firms[firm].home_market]
            if sizes[step, firms[firm].market_id, firms[firm].home_market] < 0:
                firms[firm].states[step + 1] = 1
                sizes[step, firms[firm].market_id, firms[firm].home_market] = 1
            if sizes[step, firms[firm].market_id, firms[firm].home_market] < 0:
                print('!')

        market_share[step, :, :] = sizes[step, :, :] / np.sum(sizes[step, :, :], axis=0)
        if (market_share[step, :, :] < 0).any():
            print('!')
        market_growth[step, :] = sizes[step, :, :].sum(axis=0) / sizes[step - 1, :, :].sum(axis=0)

        if step > lookback:
            for firm in firms:
                if len(firm.conglomerate) > 1:
                    if firm.entered < step - lookback:
                        outside_profit = np.prod(firm.outside_profits) ** (1 / lookback)
                        if np.isnan(outside_profit):
                            print('!')
                        inside_profits = firm.states[step - lookback:step]
                        inside_profits = np.prod(inside_profits[1:] / inside_profits[:-1]) ** (1 / lookback)
                        if outside_profit > inside_profits:
                            conglomerates[firm.conglomerate_id]['firms'].remove(firm.id)
                            if len(conglomerates[firm.conglomerate_id]['firms']) < 2:
                                to_del.append(firm.conglomerate_id)
                            exit_(firms, firm)

        for deletion in to_del:
            for firm in conglomerates[deletion]['firms']:
                exit_(firms, firms[firm])
            del conglomerates[deletion]

    num_cong = []
    members = []
    for conglomerate in all_time_conglomerates:
        num_cong.append(len(conglomerate))
        members.append([len(members) for members in conglomerate])

    mean_members = np.array([np.mean(member) for member in members])

    quantiles_members = np.array([np.quantile(member, q=[0.1, 0.25, 0.5, 0.75, 0.9]) for member in members])

    # shape = markets x steps + 1 x firms_per_market
    quantiles_shares = np.quantile(market_share, q=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99], axis=1)
    max_shares = np.max(market_share, axis=1)

    squared_market_share = market_share ** 2
    hhi = np.sum(squared_market_share, axis=1)

    sorted_market_share = np.sort(market_share, axis=1)
    cum_market_share = np.cumsum(sorted_market_share, axis=1)
    sums = np.sum(sorted_market_share, axis=1)
    sums = sums[..., np.newaxis]
    lorenz_curve = cum_market_share / sums
    area_under_curve = np.trapz(y=lorenz_curve, axis=1, dx=1 / firms_per_market)
    gini_coefficient = 1 - 2 * area_under_curve

    ranks = np.array([np.argsort(np.argsort(market_share[step_, :, :], axis=0)) for step_ in range(steps)])

    percentile_thresh = firms_per_market * np.array([0.1, 0.9, ranks.max() / firms_per_market])

    percentile_ranks = [(ranks >= thresh).sum(axis=0) for thresh in percentile_thresh]

    avg_shares = []
    avg_ranks = []
    for period in range(steps):
        avg_share = []
        avg_rank = []
        for conglomerate in all_time_conglomerates[period]:
            coords = markets_structure[list(conglomerate)]
            to_append = [len(conglomerate), market_share[period, :, :][(coords[:, 0], coords[:, 1])].mean()]
            avg_share.append(to_append)
            to_append = [len(conglomerate), ranks[period, :, :][(coords[:, 0], coords[:, 1])].mean()]
            avg_rank.append(to_append)

        avg_shares.append(np.array(avg_share))
        avg_ranks.append(np.array(avg_rank))

    model_results = [mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, max_shares, market_share,
                     hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks, sizes, profits, prices]

    return model_results


def price_opt(share, cost, scale=1):
    p_opt = (1 + (1 - share * scale) * cost) / (1 - share * scale + 1e-6)
    return p_opt


def profit(p, share, cost, scale=1):
    pi = (p - cost) * np.exp(-1 * (1 - share * scale + 1e-6) * p)
    return pi
