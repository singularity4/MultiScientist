"""
Rejection-free Gibbs sampler.

"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from SIR_mapping import DynamicSSSP

__all__ = ["mcmc_sir_exact"]


class _ExactChain:
    """State: one recovery time per node, one transmission time per arc."""

    def __init__(self, A, beta, gamma, rng):
        if beta <= 0:
            raise ValueError(f"beta must be positive (got {beta}).")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative (got {gamma}).")

        M = sp.csr_matrix(A)
        self.n_nodes = M.shape[0]
        self.indptr = M.indptr           # out-arcs of node i: indptr[i]:indptr[i+1]
        self.row = np.repeat(np.arange(self.n_nodes, dtype=np.intp), np.diff(M.indptr))
        self.n_arcs = M.indices.size
        self.beta, self.gamma, self.rng = beta, gamma, rng

        self.pending: set[int] = set()          # arcs changed since the last repair
        self.tau_node = self._recovery(self.n_nodes)
        self.rho = self._transmission(self.n_arcs)
        self.weights = np.where(
            self.rho <= self.tau_node[self.row], self.rho, np.inf
        )

    def _transmission(self, size):
        return self.rng.exponential(scale=1.0 / self.beta, size=size)

    def _recovery(self, size):
        if self.gamma == 0:
            return np.full(size, np.inf)
        return self.rng.exponential(scale=1.0 / self.gamma, size=size)

    def step(self):
        """Resample one node's recovery time and all of its out-arcs."""
        node = self.rng.integers(self.n_nodes)
        self.tau_node[node] = self._recovery(1)[0]
        lo, hi = self.indptr[node], self.indptr[node + 1]
        if lo == hi:
            return                                    # no outgoing arcs
        self.rho[lo:hi] = self._transmission(hi - lo)
        self.weights[lo:hi] = np.where(
            self.rho[lo:hi] <= self.tau_node[node], self.rho[lo:hi], np.inf
        )
        self.pending.update(range(int(lo), int(hi)))


def mcmc_sir_exact(A, beta, gamma, source, num_samples, burn_in=0, thin=1, rng=None,
                   incremental=True):
    """
    samples = mcmc_sir_exact(A, beta, gamma, source, num_samples, burn_in, thin)

    Returns an (num_samples, N) array of first infection times. Successive states
    differ in one node's neighbourhood, so samples are correlated; `thin` should
    be on the order of the node count.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1 (got {num_samples}).")
    if burn_in < 0:
        raise ValueError(f"burn_in must be non-negative (got {burn_in}).")
    if thin < 1:
        raise ValueError(f"thin must be at least 1 (got {thin}).")

    rng = np.random.default_rng() if rng is None else rng
    chain = _ExactChain(A, beta, gamma, rng)

    paths = DynamicSSSP(A, source)
    paths.full(chain.weights)
    chain.pending.clear()

    samples = np.empty((num_samples, chain.n_nodes))
    taken = 0
    for step in range(1, burn_in + num_samples * thin + 1):
        chain.step()
        if step > burn_in and (step - burn_in) % thin == 0:
            if incremental:
                samples[taken] = paths.update(
                    chain.weights, np.fromiter(chain.pending, dtype=np.intp))
            else:
                samples[taken] = paths.full(chain.weights)
            chain.pending.clear()
            taken += 1
    return samples
