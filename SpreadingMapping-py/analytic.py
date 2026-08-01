"""
Analytical results from the paper.

  Eq. 5   transmissibility
  Eq. 7   p_{n,k}
  Fig. 3  toy network, and its source-to-destination probability under each
          mapping — the two differ
"""

from __future__ import annotations

import numpy as np
from scipy.special import betaln, gammaln

__all__ = [
    "transmissibility",
    "p_nk",
    "toy_chain_graph",
    "toy_exact",
    "toy_meanfield",
]


def transmissibility(beta: float, gamma: float) -> float:
    """
    Eq. 5: probability of transmitting along a link before recovering.
    For the Poissonian process this is beta / (beta + gamma).
    """
    if beta < 0 or gamma < 0:
        raise ValueError(f"Rates must be non-negative (got beta={beta}, gamma={gamma}).")
    if beta == 0:
        return 0.0
    if gamma == 0:
        return 1.0                       # SI limit: every link eventually fires
    return beta / (beta + gamma)


def p_nk(n: int, k: int, beta: float, gamma: float) -> float:
    """
    Eq. 7: probability that exactly k of a node's n out-links are active.
    
    Not Binomial(n, p) — the n links share the node's single recovery time, and
    that correlation is what the exact mapping keeps. Computed in log space.
    """
    if not (0 <= k <= n):
        raise ValueError(f"Need 0 <= k <= n (got n={n}, k={k}).")
    if beta <= 0:
        raise ValueError(f"beta must be positive (got {beta}).")
    if gamma == 0:
        return 1.0 if k == n else 0.0    # SI limit: all mass on k = n

    g = gamma / beta
    log_p = (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
             + np.log(g) + betaln(g + n - k, k + 1))
    return float(np.exp(log_p))


def toy_chain_graph(n_chains: int, length: int) -> np.ndarray:
    """
    Fig. 3(a): n_chains parallel chains of `length` nodes from source (node 0) to
    destination (node 1). Directed, so every chain node has out-degree one — which
    is why the analytic result uses p_{1,1}. Each chain carries length + 1 links.
    """
    if n_chains < 1 or length < 1:
        raise ValueError("n_chains and length must both be at least 1.")

    n_nodes = 2 + n_chains * length
    A = np.zeros((n_nodes, n_nodes))
    for c in range(n_chains):
        nodes = list(range(2 + c * length, 2 + (c + 1) * length))
        A[0, nodes[0]] = 1.0
        for a, b in zip(nodes, nodes[1:]):
            A[a, b] = 1.0
        A[nodes[-1], 1] = 1.0
    return A


def toy_exact(n_chains: int, length: int, beta: float, gamma: float) -> float:
    """
    Fig. 3, exact mapping:  P = 1 - sum_j p_{n_c,j} (1 - p_{1,1}^l)^j
    
    The source's out-links are correlated through its recovery time, hence p_{n_c,j}
    rather than a binomial; the rest of each chain is independent, hence p_{1,1}^l.
    """
    chain_fails = 1.0 - p_nk(1, 1, beta, gamma) ** length
    total = sum(p_nk(n_chains, j, beta, gamma) * chain_fails**j
                for j in range(n_chains + 1))
    return 1.0 - total


def toy_meanfield(n_chains: int, length: int, beta: float, gamma: float) -> float:
    """
    The same under the mean-field mapping, every link independent with
    p = beta/(beta+gamma):   P = 1 - (1 - p^{l+1})^{n_c}
    """
    p = transmissibility(beta, gamma)
    return 1.0 - (1.0 - p ** (length + 1)) ** n_chains
