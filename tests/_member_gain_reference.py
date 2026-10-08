#!/usr/bin/env python3
"""
C19: Reference implementation of member_gain_loggain (pre-hoist).

This is the unmodified implementation vendored for bit-identical verification.
"""
import numpy as np
import numba as nb


@nb.njit(cache=True)
def member_gain_loggain_reference(i, members, alpha, window_returns, window_sizes, Phi_K, sharing_rule_code, g):
    """
    Compute demeaned log-growth gain for member i in a proposed conglomerate.

    The loggain rule removes first-order noise (α·(mean r̄ − mean r_i)) by demeaning
    each partner's window returns before replay. This makes the decision:
    - Independent of realized common growth (which cannot be controlled)
    - Focused on the pooling benefit from idiosyncratic variance reduction

    Parameters:
    -----------
    i : int
        Index of the firm within members array (0-indexed position in members)
    members : ndarray, shape (K,)
        Indices of firms in the proposed set (used to slice window_returns/sizes)
    alpha : float
        Pooling fraction (share)
    window_returns : ndarray, shape (h, total_firms) or (h, K)
        Log returns for h historical steps. If total_firms, indexed by members[j].
        If K, indexed directly by j.
    window_sizes : ndarray, shape (h, total_firms) or (h, K)
        Firm sizes (levels, not log) for h historical steps
    Phi_K : float
        Management cost for this conglomerate size
    sharing_rule_code : int
        0 = equal, 1 = proportional
    g : float
        Common growth rate (added back after demeaning)

    Returns:
    --------
    delta_hat : float
        Mean log gain from pooling: Δ̂_i = (1/h) Σ_τ [ log(1 + p_iτ) − log(1 + ε_iτ) ]
        Returns -inf if any 1 + p_iτ <= 0 (invalid)
    """
    h = window_returns.shape[0]
    K = len(members)

    # Determine if window_returns is (h, total_firms) or (h, K)
    # If window_returns.shape[1] > K, we need to index by members[j]
    # Otherwise, assume it's already sliced to K firms
    full_indexing = window_returns.shape[1] > K

    # Step 1: Extract returns for members and compute per-firm demeaned returns
    # ε_jτ = r̃_jτ − mean_τ(r̃_jτ) + g
    epsilon = np.empty((h, K), dtype=np.float64)

    for j in range(K):
        if full_indexing:
            firm_j = members[j]
        else:
            firm_j = j

        # Compute mean of this firm's returns over the window
        mean_r_j = 0.0
        for tau in range(h):
            mean_r_j += window_returns[tau, firm_j]
        mean_r_j /= h

        # Demean and add common growth
        for tau in range(h):
            epsilon[tau, j] = window_returns[tau, firm_j] - mean_r_j + g

    # Step 2: Extract sizes for members
    sizes = np.empty((h, K), dtype=np.float64)
    for tau in range(h):
        for j in range(K):
            if full_indexing:
                sizes[tau, j] = window_sizes[tau, members[j]]
            else:
                sizes[tau, j] = window_sizes[tau, j]

    # Step 3: Compute pooled net return for member i at each tau
    delta_hat = 0.0

    for tau in range(h):
        # Compute weights and pool contribution
        sum_s = 0.0
        sum_s_eps = 0.0
        for j in range(K):
            s_j = sizes[tau, j]
            e_j = np.expm1(epsilon[tau, j])  # Convert log return to linear return
            sum_s += s_j
            sum_s_eps += s_j * e_j

        s_i = sizes[tau, i]
        e_i = np.expm1(epsilon[tau, i])

        if sharing_rule_code == 1:
            # Proportional sharing: p_iτ = (1−α)·ε_iτ + α·Σ_j w_jτ ε_jτ − Φ(K̂)
            # where w_jτ = s_jτ / Σ s_kτ
            avg_weighted_gain = sum_s_eps / sum_s if sum_s > 0 else 0.0
            p_i = (1.0 - alpha) * e_i + alpha * avg_weighted_gain - Phi_K
        else:
            # Equal sharing: p_iτ = (1−α)·ε_iτ + [α·Σ_j s_jτ ε_jτ − Φ(K̂)·Σ_j s_jτ] / (K̂·s_iτ)
            # Pool in dollars: α·Σ_j Δ_jτ − Φ(K̂)·S_τ, distributed equally per firm
            pool_dollars = alpha * sum_s_eps - Phi_K * sum_s
            equal_share = pool_dollars / K
            s_i_safe = max(s_i, 1e-300)
            p_i = (1.0 - alpha) * e_i + equal_share / s_i_safe

        # Check for invalid (would cause exit)
        if p_i <= -1.0:
            return -np.inf

        # Accumulate: log(1 + p_i) - log(1 + ε_i)
        delta_hat += np.log1p(p_i) - np.log1p(e_i)

    # Mean over window
    delta_hat /= h

    return delta_hat
