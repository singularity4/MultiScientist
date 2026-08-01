"""
Rejection-free Gibbs sampler over the mean-field ensemble.

Transmission and recovery times are drawn per edge rather than per node, which discards the correlation
among a node's out-arcs; valid when beta >> gamma. On a symmetric topology the
two arcs of an edge share one weight, so the ensemble is undirected.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from SIR_mapping import DynamicSSSP

__all__ = ["mcmc_sir_meanfield"]


class _MeanFieldChain:
    """State: one weight per edge, from an independent (rho, tau) pair."""

    def __init__(self, A, beta, gamma, rng):
        if beta <= 0:
            raise ValueError(f"beta must be positive (got {beta}).")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative (got {gamma}).")

        M = sp.csr_matrix(A)
        self.n_nodes = M.shape[0]
        self.col = M.indices
        self.row = np.repeat(np.arange(self.n_nodes, dtype=np.intp), np.diff(M.indptr))
        self.n_arcs = self.col.size
        self.beta, self.gamma, self.rng = beta, gamma, rng
        self.weights = np.full(self.n_arcs, np.inf)
        self.pending: set[int] = set()          # arcs changed since the last repair

        if (M != M.T).nnz == 0:
            # Undirected: the two arcs of an edge share one weight.
            key = (np.minimum(self.row, self.col).astype(np.int64) * self.n_nodes
                   + np.maximum(self.row, self.col))
            _, edge_of_arc = np.unique(key, return_inverse=True)
            self.n_edges = int(edge_of_arc.max()) + 1 if self.n_arcs else 0
            self.arcs_of_edge = [
                np.flatnonzero(edge_of_arc == e) for e in range(self.n_edges)
            ]
        else:
            self.n_edges = self.n_arcs
            self.arcs_of_edge = [np.array([a]) for a in range(self.n_arcs)]

        for e in range(self.n_edges):
            self.weights[self.arcs_of_edge[e]] = self._draw_edge()

    def _draw_edge(self):
        rho = self.rng.exponential(scale=1.0 / self.beta)
        if self.gamma == 0:
            return rho
        tau = self.rng.exponential(scale=1.0 / self.gamma)
        return rho if rho <= tau else np.inf

    def step(self):
        """Assign a new weight to one randomly chosen edge."""
        e = self.rng.integers(self.n_edges)
        self.weights[self.arcs_of_edge[e]] = self._draw_edge()
        self.pending.update(int(a) for a in self.arcs_of_edge[e])


def mcmc_sir_meanfield(A, beta, gamma, source, num_samples, burn_in=0, thin=1, rng=None,
                       incremental=True):
    """
    samples = mcmc_sir_meanfield(A, beta, gamma, source, num_samples, burn_in, thin)

    Returns an (num_samples, N) array of first infection times. 
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1 (got {num_samples}).")
    if burn_in < 0:
        raise ValueError(f"burn_in must be non-negative (got {burn_in}).")
    if thin < 1:
        raise ValueError(f"thin must be at least 1 (got {thin}).")

    rng = np.random.default_rng() if rng is None else rng
    chain = _MeanFieldChain(A, beta, gamma, rng)

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
