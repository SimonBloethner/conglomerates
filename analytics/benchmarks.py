#!/usr/bin/env python3
"""
Analytical benchmarks for pooling gain calculations.

Pure NumPy/SciPy implementation - no model imports.
"""
import numpy as np
from scipy import stats
import csv
from pathlib import Path


# Default Monte Carlo parameters
MC_DRAWS = 2_000_000
MC_SEED = 42


def management_cost_function(size, cost_type, c0=None, c1=None, c2=None):
    """
    Management cost function (reimplemented locally from collaborative_growth.py).

    Parameters:
    -----------
    size : int or array
        Conglomerate size (number of firms)
    cost_type : str
        One of 'linear', 'quadratic', 'exponential', 'power_law'
    c0, c1, c2 : float, optional
        Cost parameters. If None, uses calibrated defaults.

    Returns:
    --------
    float or array : Management cost
    """
    # Cost-function-specific default parameters (calibrated for convergence at size=40)
    defaults = {
        'power_law': {'c0': 0.00002032, 'c1': 1.2, 'c2': 0.001},
        'linear': {'c0': 0.00001, 'c1': 0.00004225, 'c2': 0.001},
        'quadratic': {'c0': 0.0001, 'c1': 0.000001, 'c2': 0.00000097},
        'exponential': {'c0': 0.001, 'c1': 0.01326571, 'c2': 0.001},
    }

    if cost_type not in defaults:
        raise ValueError(
            f"Unknown cost_type: {cost_type}. "
            f"Supported: 'linear', 'quadratic', 'exponential', 'power_law'"
        )

    # Use provided parameters or defaults
    func_defaults = defaults[cost_type]
    c0 = c0 if c0 is not None else func_defaults['c0']
    c1 = c1 if c1 is not None else func_defaults['c1']
    c2 = c2 if c2 is not None else func_defaults['c2']

    if cost_type == 'linear':
        return c0 + c1 * size
    elif cost_type == 'quadratic':
        return c0 + c1 * size + c2 * size ** 2
    elif cost_type == 'exponential':
        return c0 * np.exp(c1 * size)
    elif cost_type == 'power_law':
        return c0 * size ** c1


def iqr_to_scale(family, iqr):
    """
    Convert IQR to distribution scale parameter.

    Parameters:
    -----------
    family : str
        Distribution family: 'normal', 'laplace', 't3'
    iqr : float
        Inter-quartile range

    Returns:
    --------
    float : Scale parameter (sigma for normal, b for Laplace, s for t)
    """
    if family == 'normal':
        # IQR = 2 * z_0.75 * sigma = 2 * 0.6745 * sigma ≈ 1.349 * sigma
        return iqr / 1.3489795003921634  # 2 * norm.ppf(0.75)
    elif family == 'laplace':
        # IQR = 2 * b * ln(2)
        return iqr / (2 * np.log(2))
    elif family == 't3':
        # IQR = 2 * s * t_inv(0.75, df=3)
        t_inv_075 = stats.t.ppf(0.75, df=3)
        return iqr / (2 * t_inv_075)
    else:
        raise ValueError(f"Unknown family: {family}")


def draw_log_shocks(n_draws, family, mu, iqr, rng):
    """
    Draw log-shocks X from the specified family.

    Parameters:
    -----------
    n_draws : int
        Number of draws
    family : str
        Distribution family
    mu : float
        Location parameter
    iqr : float
        Inter-quartile range
    rng : np.random.Generator
        Random number generator

    Returns:
    --------
    array : Log-shocks X such that e^X is the growth factor
    """
    scale = iqr_to_scale(family, iqr)

    if family == 'normal':
        return rng.normal(mu, scale, n_draws)
    elif family == 'laplace':
        return rng.laplace(mu, scale, n_draws)
    elif family == 't3':
        # Student-t with df=3, scaled and shifted
        return mu + scale * rng.standard_t(df=3, size=n_draws)
    else:
        raise ValueError(f"Unknown family: {family}")


def pooling_gain(K, alpha, family, mu, iqr, cost, n_draws=MC_DRAWS, seed=MC_SEED):
    """
    Compute the planner's per-member time-average growth gain from proportional pooling.

    The gain is:
        E[log((1-α)·e^X₁ + α·mean_j e^Xⱼ)] - E[log e^X₁] - cost

    Parameters:
    -----------
    K : int
        Number of members in the pool
    alpha : float
        Pooling fraction (0 to 1)
    family : str
        Distribution family: 'normal', 'laplace', 't3'
    mu : float
        Location parameter of log-shocks
    iqr : float
        Inter-quartile range of log-shocks
    cost : float
        Per-member management cost
    n_draws : int
        Number of Monte Carlo draws
    seed : int
        Random seed

    Returns:
    --------
    tuple : (gain, standard_error)
    """
    rng = np.random.default_rng(seed)

    # Draw log-shocks for K members
    # Shape: (n_draws, K)
    X = np.column_stack([
        draw_log_shocks(n_draws, family, mu, iqr, rng)
        for _ in range(K)
    ])

    # Compute growth factors e^X
    eX = np.exp(X)

    # Member 1's standalone log-growth: log(e^X₁) = X₁
    standalone = X[:, 0]

    # Pooled outcome for member 1:
    # (1-α)·e^X₁ + α·mean_j(e^Xⱼ)
    pool_mean = np.mean(eX, axis=1)
    pooled_factor = (1 - alpha) * eX[:, 0] + alpha * pool_mean
    pooled = np.log(pooled_factor)

    # Per-draw gain
    gain_per_draw = pooled - standalone - cost

    # Mean and standard error
    mean_gain = np.mean(gain_per_draw)
    se = np.std(gain_per_draw, ddof=1) / np.sqrt(n_draws)

    return mean_gain, se


def k_star(alpha, family, mu, iqr, cost_type, c0=None, c1=None, c2=None, M=None):
    """
    Find the optimal pool size K* that maximizes pooling_gain - Φ(K).

    WARNING: K* is only economically interpretable when K* ≤ M (number of markets).
    A conglomerate cannot have more members than there are markets. For very low
    management costs or heavy-tailed distributions, the theoretical optimal K* may
    exceed M; in that case, the monopoly-reversion result applies (all firms merge
    into a single entity). When M is provided, K* is capped at M.

    Parameters:
    -----------
    alpha : float
        Pooling fraction
    family : str
        Distribution family
    mu : float
        Location parameter
    iqr : float
        Inter-quartile range
    cost_type : str
        Cost function type
    c0, c1, c2 : float, optional
        Cost function parameters
    M : int, optional
        Number of markets. If provided, K* is capped at M.

    Returns:
    --------
    tuple : (K_star, max_gain)
    """
    best_K = 1
    best_gain = float('-inf')

    # Search up to 60 or M if specified
    max_search = min(60, M) if M is not None else 60

    for K in range(1, max_search + 1):
        cost = management_cost_function(K, cost_type, c0, c1, c2)
        gain, _ = pooling_gain(K, alpha, family, mu, iqr, cost)

        if gain > best_gain:
            best_gain = gain
            best_K = K

    return best_K, best_gain


def barrier_exponent(c):
    """
    Compute barrier exponent from reflection coefficient.

    Parameters:
    -----------
    c : float
        Reflection coefficient (0 < c < 1)

    Returns:
    --------
    float : Barrier exponent 1/(1-c)
    """
    return 1.0 / (1.0 - c)


def c_for_exponent(target):
    """
    Compute reflection coefficient for a target barrier exponent.

    Parameters:
    -----------
    target : float
        Target barrier exponent (> 1)

    Returns:
    --------
    float : Reflection coefficient c = 1 - 1/target
    """
    return 1.0 - 1.0 / target


def body_tail_split(alpha, K, family, mu, iqr, q=0.01, n_draws=MC_DRAWS, seed=MC_SEED):
    """
    Compute the share of pooling gain from tail events.

    A tail event is when any member's X is below the q-quantile.

    Parameters:
    -----------
    alpha : float
        Pooling fraction
    K : int
        Number of members
    family : str
        Distribution family
    mu : float
        Location parameter
    iqr : float
        Inter-quartile range
    q : float
        Quantile threshold (default 0.01 = 1st percentile)
    n_draws : int
        Number of Monte Carlo draws
    seed : int
        Random seed

    Returns:
    --------
    float : Share of pooling gain from tail events (0 to 1)
    """
    rng = np.random.default_rng(seed)

    # Draw log-shocks for K members
    X = np.column_stack([
        draw_log_shocks(n_draws, family, mu, iqr, rng)
        for _ in range(K)
    ])

    # Compute the q-quantile threshold
    scale = iqr_to_scale(family, iqr)
    if family == 'normal':
        threshold = mu + scale * stats.norm.ppf(q)
    elif family == 'laplace':
        threshold = mu + scale * stats.laplace.ppf(q)
    elif family == 't3':
        threshold = mu + scale * stats.t.ppf(q, df=3)
    else:
        raise ValueError(f"Unknown family: {family}")

    # Identify tail events: any member's X below threshold
    is_tail = np.any(X < threshold, axis=1)

    # Compute growth factors
    eX = np.exp(X)

    # Standalone and pooled outcomes
    standalone = X[:, 0]
    pool_mean = np.mean(eX, axis=1)
    pooled_factor = (1 - alpha) * eX[:, 0] + alpha * pool_mean
    pooled = np.log(pooled_factor)

    # Per-draw gain (excluding cost for this calculation)
    gain_per_draw = pooled - standalone

    # Total gain
    total_gain = np.sum(gain_per_draw)

    # Tail contribution
    tail_gain = np.sum(gain_per_draw[is_tail])

    if total_gain <= 0:
        return 0.0

    return tail_gain / total_gain


def generate_benchmarks_csv(output_path='analytics/benchmarks.csv', M=50):
    """
    Generate benchmarks.csv with K* and gain for various parameter combinations.

    Parameters:
    -----------
    output_path : str
        Path to output CSV file
    M : int
        Number of markets (caps K* at this value)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    families = ['normal', 'laplace', 't3']
    cost_types = ['linear', 'quadratic', 'exponential', 'power_law']
    alphas = [0.1, 0.3, 0.5]
    mu = 0.05
    iqr = 0.2

    rows = []

    print(f"Generating benchmarks (K* capped at M={M})...")

    for family in families:
        for cost_type in cost_types:
            for alpha in alphas:
                print(f"  {family}, {cost_type}, alpha={alpha}...")

                K_opt, gain_opt = k_star(alpha, family, mu, iqr, cost_type, M=M)

                # Body/tail split at K=5
                tail_share = body_tail_split(alpha, 5, family, mu, iqr, q=0.01)

                rows.append({
                    'family': family,
                    'cost_type': cost_type,
                    'alpha': alpha,
                    'mu': mu,
                    'iqr': iqr,
                    'K_star': K_opt,
                    'gain_at_K_star': gain_opt,
                    'tail_share_K5': tail_share,
                })

    # Write CSV
    fieldnames = ['family', 'cost_type', 'alpha', 'mu', 'iqr',
                  'K_star', 'gain_at_K_star', 'tail_share_K5']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path}")
    return rows


if __name__ == '__main__':
    generate_benchmarks_csv()
