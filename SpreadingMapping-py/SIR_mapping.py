"""
Mapping SIR dynamics to weighted shortest paths.

A realization of the process is a weighted network: edge weights are propagation
delays drawn by the inverse Smirnov transform (Eq. 1; Poissonian form Eq. 2),
and a node's first-infection time is the weighted shortest path from the source.

Expectations are averages over an ensemble of realizations (Eq. 3).

An arc whose transmitting node recovers before it transmits has weight infinity
and is stored by omission. Representing it as a zero weight would make it a free
instantaneous arc — the opposite — and every result would be wrong while still
looking plausible.
"""

from __future__ import annotations

from heapq import heappop, heappush

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

__all__ = ["SIR_mapping", "sample_realization", "propagation_times", "arc_list",
           "ensemble", "DynamicSSSP"]


def arc_list(A):
    """Arcs as index arrays: arc k runs from I[k] to J[k]."""
    M = sp.csr_matrix(A)
    if M.shape[0] != M.shape[1]:
        raise ValueError(f"Adjacency matrix must be square (got {M.shape}).")
    C = M.tocoo()
    return C.row.astype(np.intp), C.col.astype(np.intp), M.shape[0]


def _is_symmetric(A) -> bool:
    M = sp.csr_matrix(A)
    return (M != M.T).nnz == 0


def _draw_exact(I, J, n_nodes, beta, gamma, rng):
    """
    Eq. 2, exact mapping: one recovery time per node, one transmission time per
    arc. All arcs out of a node share its recovery deadline — the dynamical
    correlation the mean-field mapping discards.

    numpy's `exponential(scale=1/beta)` has mean 1/beta, matching `exprnd`.
    """
    rho = rng.exponential(scale=1.0 / beta, size=I.size)
    if gamma == 0:
        return rho                                   # SI limit: nothing recovers
    tau_node = rng.exponential(scale=1.0 / gamma, size=n_nodes)
    return np.where(rho <= tau_node[I], rho, np.inf)


def _draw_meanfield(I, J, n_nodes, beta, gamma, rng, symmetric):
    """
    Eq. 2, mean-field mapping: transmission and recovery drawn per edge,
    independently. Valid when beta >> gamma. The paper specifies an *undirected*
    ensemble, so on a symmetric topology both directions share one weight.
    """
    if not symmetric:
        rho = rng.exponential(scale=1.0 / beta, size=I.size)
        if gamma == 0:
            return rho
        tau = rng.exponential(scale=1.0 / gamma, size=I.size)
        return np.where(rho <= tau, rho, np.inf)

    upper = I < J
    n_edges = int(upper.sum())
    rho_edge = rng.exponential(scale=1.0 / beta, size=n_edges)
    if gamma == 0:
        weight_edge = rho_edge
    else:
        tau_edge = rng.exponential(scale=1.0 / gamma, size=n_edges)
        weight_edge = np.where(rho_edge <= tau_edge, rho_edge, np.inf)

    # Key each undirected edge by (min, max) endpoints so both arcs agree.
    key = np.minimum(I, J).astype(np.int64) * n_nodes + np.maximum(I, J)
    edge_key = key[upper]
    order = np.argsort(edge_key, kind="stable")
    return weight_edge[order][np.searchsorted(edge_key[order], key)]


def sample_realization(A, beta, gamma, rng=None, mapping="exact"):
    """
    One weighted network G_k from the ensemble (Eq. 1; Poissonian form Eq. 2).

    Returns a CSR matrix holding only the finite propagation delays. Inactive
    arcs are absent.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive (got {beta}).")
    if gamma < 0:
        raise ValueError(f"gamma must be non-negative (got {gamma}).")
    rng = np.random.default_rng() if rng is None else rng

    I, J, n_nodes = arc_list(A)
    if mapping == "exact":
        w = _draw_exact(I, J, n_nodes, beta, gamma, rng)
    elif mapping == "meanfield":
        w = _draw_meanfield(I, J, n_nodes, beta, gamma, rng, _is_symmetric(A))
    else:
        raise ValueError(f"mapping must be 'exact' or 'meanfield' (got {mapping!r}).")

    keep = np.isfinite(w)
    return sp.csr_matrix((w[keep], (I[keep], J[keep])), shape=(n_nodes, n_nodes))


def propagation_times(W, source):
    """First-infection times from `source`; inf where the epidemic never reaches."""
    return dijkstra(csgraph=W, directed=True, indices=source)


class DynamicSSSP:
    """
    Shortest paths from one source. Checked against full recomputation in tests.
    """

    def __init__(self, A, source: int):
        M = sp.csr_matrix(A)
        self.n = M.shape[0]
        self.source = int(source)
        self.out_indptr = M.indptr
        self.out_col = M.indices
        self.tail = np.repeat(np.arange(self.n, dtype=np.intp), np.diff(M.indptr))

        # Reverse index, so an invalidated node can be re-reached from outside.
        order = np.argsort(self.out_col, kind="stable")
        self.in_arc = order.astype(np.intp)
        counts = np.bincount(self.out_col, minlength=self.n)
        self.in_indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.intp)

        self.dist = np.full(self.n, np.inf)
        self.parent_arc = np.full(self.n, -1, dtype=np.intp)

    def full(self, w: np.ndarray) -> np.ndarray:
        """Recompute from scratch, rebuilding the shortest-path tree."""
        dist = np.full(self.n, np.inf)
        parent_arc = np.full(self.n, -1, dtype=np.intp)
        settled = np.zeros(self.n, dtype=bool)
        dist[self.source] = 0.0
        heap = [(0.0, self.source)]
        while heap:
            d, u = heappop(heap)
            if settled[u]:
                continue
            settled[u] = True
            for k in range(self.out_indptr[u], self.out_indptr[u + 1]):
                if not np.isfinite(w[k]):
                    continue
                v = self.out_col[k]
                nd = d + w[k]
                if nd < dist[v]:
                    dist[v] = nd
                    parent_arc[v] = k
                    heappush(heap, (nd, int(v)))
        self.dist, self.parent_arc = dist, parent_arc
        return dist

    def update(self, w: np.ndarray, changed: np.ndarray) -> np.ndarray:
        """Repair the distances after the arcs in `changed` took new weights."""
        dist, parent_arc = self.dist, self.parent_arc
        changed = np.asarray(changed, dtype=np.intp)
        if changed.size == 0:
            return dist

        # Children of each node in the shortest-path tree, for subtree walking.
        children = [[] for _ in range(self.n)]
        for v in range(self.n):
            k = parent_arc[v]
            if k >= 0:
                children[self.tail[k]].append(v)

        # Invalidate the subtree below every changed arc that carried a path.
        dirty = np.zeros(self.n, dtype=bool)
        stack = [self.out_col[k] for k in changed if parent_arc[self.out_col[k]] == k]
        while stack:
            v = stack.pop()
            if dirty[v]:
                continue
            dirty[v] = True
            dist[v] = np.inf
            parent_arc[v] = -1
            stack.extend(children[v])

        # Frontier: routes back into the affected region, plus changed arcs that
        # now offer a shortcut outside it.
        heap: list[tuple[float, int]] = []
        for v in np.flatnonzero(dirty):
            for j in range(self.in_indptr[v], self.in_indptr[v + 1]):
                k = self.in_arc[j]
                u = self.tail[k]
                if dirty[u] or not np.isfinite(w[k]) or not np.isfinite(dist[u]):
                    continue
                nd = dist[u] + w[k]
                if nd < dist[v]:
                    dist[v] = nd
                    parent_arc[v] = k
                    heappush(heap, (nd, int(v)))
        for k in changed:
            u, v = self.tail[k], self.out_col[k]
            if dirty[u] or not np.isfinite(w[k]) or not np.isfinite(dist[u]):
                continue
            nd = dist[u] + w[k]
            if nd < dist[v]:
                dist[v] = nd
                parent_arc[v] = k
                heappush(heap, (nd, int(v)))

        settled = np.zeros(self.n, dtype=bool)
        while heap:
            d, u = heappop(heap)
            if settled[u] or d > dist[u]:
                continue
            settled[u] = True
            for k in range(self.out_indptr[u], self.out_indptr[u + 1]):
                if not np.isfinite(w[k]):
                    continue
                v = self.out_col[k]
                nd = d + w[k]
                if nd < dist[v]:
                    dist[v] = nd
                    parent_arc[v] = k
                    heappush(heap, (nd, int(v)))

        self.dist, self.parent_arc = dist, parent_arc
        return dist


def SIR_mapping(A, N=None, E=None, beta=None, gamma=None, source=None,
                rng=None, mapping="exact"):
    """
    D = SIR_mapping(A, N, E, beta, gamma, source)
    
    INPUT   adjacency matrix A (unweighted), node count N, arc count E,
            SIR parameters beta, gamma, id of source node
    OUTPUT  vector of first infection times from source to other nodes
    
    N and E are checked against A and may be None. `source` is 0-indexed.
    """
    if beta is None or gamma is None or source is None:
        raise ValueError("beta, gamma and source are required.")

    I, J, n_nodes = arc_list(A)
    if N is not None and N != n_nodes:
        raise ValueError(f"N={N} does not match the {n_nodes} nodes in A.")
    if E is not None and E != I.size:
        raise ValueError(f"E={E} does not match the {I.size} arcs in A.")

    return propagation_times(sample_realization(A, beta, gamma, rng, mapping), source)


def ensemble(A, beta, gamma, source, n_samples, rng=None, mapping="exact"):
    """
    n_samples independent realizations, as an (n_samples, N) array of first
    infection times. Averaging over rows estimates <f(G)> by Eq. 3.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1 (got {n_samples}).")
    rng = np.random.default_rng() if rng is None else rng
    n_nodes = sp.csr_matrix(A).shape[0]
    out = np.empty((n_samples, n_nodes))
    for k in range(n_samples):
        out[k] = propagation_times(
            sample_realization(A, beta, gamma, rng, mapping), source
        )
    return out
