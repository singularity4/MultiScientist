"""
Time-critical immunization strategies (Fig. 7).

An outbreak from a known source is observed at t0. Budget of m immunizations is available but
doses take effect only at t0 + Delta_t, so a recipient node infected inside that window is
a wasted dose. Three strategies select the node targets: random, hubs (proportional to degree) 
and temporal (proportional to p_tilde, Eq. 10).

"""

from __future__ import annotations

from heapq import heappop, heappush

import numpy as np
import scipy.sparse as sp

from SIR_mapping import ensemble, propagation_times, sample_realization

__all__ = [
    "immunization_probabilities",
    "susceptible_candidates",
    "strategy_random",
    "strategy_hubs",
    "strategy_temporal",
    "infection_times_with_delay",
    "outbreak_size",
    "outbreak_size_with_delay",
    "outbreak_curve",
    "compare_strategies",
]


# --------------------------------------------------------------------------
# Eq. 10
# --------------------------------------------------------------------------

def immunization_probabilities(A, beta, gamma, source, t0, Delta_t, n_samples,
                      rng=None, mapping="exact"):
    """
    Eq. 10: for each node, the estimated probability it is *not* infected before
    `horizon` — Theta is one when the first-infection time reaches it. Nodes never
    reached count towards p_tilde, correctly: they are still susceptible.
    
    With horizon = t0 + Delta_t this is the paper's p_tilde. With Delta_t = 0 it
    gives susceptibility at t0, which defines the candidate set.
    """
    D = ensemble(A, beta, gamma, source, n_samples, rng=rng, mapping=mapping)
    return (D >= t0 + Delta_t).mean(axis=0)


def susceptible_candidates(p_susceptible_at_t0, source, threshold=0.5):
    """
    Candidate nodes a timing-aware strategy may immunize: susceptible at t0 in more than
    `threshold` of realizations, excluding the source.
    """
    candidates = np.flatnonzero(p_susceptible_at_t0 > threshold)
    return candidates[candidates != source]


# --------------------------------------------------------------------------
# The immunization strategies
# --------------------------------------------------------------------------

def _select(candidates, weights, m, rng, selection):
    """
    Choose m candidates, weighted. 'proportional' samples without replacement with
    probability proportional to the weights, as the paper describes; 'top' takes
    the m largest.
    """
    m = int(min(m, candidates.size))
    if m <= 0:
        return np.empty(0, dtype=np.intp)

    w = np.asarray(weights, dtype=float)
    if selection == "top":
        return candidates[np.argsort(-w, kind="stable")[:m]]
    if selection != "proportional":
        raise ValueError(f"selection must be 'proportional' or 'top' (got {selection!r}).")

    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        p = np.full(candidates.size, 1.0 / candidates.size)
    else:
        p = w / total
    return rng.choice(candidates, size=m, replace=False, p=p)


def strategy_random(A, beta, gamma, source, t0, m, n_samples,
                    rng=None, mapping="exact"):
    """
    immunized = strategy_random(A, beta, gamma, source, t0, m, n_samples)

    Baseline: m nodes drawn uniformly from the susceptible population at t0.
    
    """
    rng = np.random.default_rng() if rng is None else rng
    q = immunization_probabilities(A, beta, gamma, source, t0, 0.0, n_samples, rng, mapping)
    candidates = susceptible_candidates(q, source)
    return _select(candidates, np.ones(candidates.size), m, rng, "proportional")


def strategy_hubs(A, source, m, rng=None, selection="proportional"):
    """
    immunized = strategy_hubs(A, source, m)
    
    Immunize proportional to total degree. Reads the network topology only, which is why it 
    wastes doses on hubs the infection reaches first and
    performs worse than random in Fig. 7. Candidates are the whole network except
    the source.
    """
    rng = np.random.default_rng() if rng is None else rng
    M = sp.csr_matrix(A)
    degree = np.asarray(M.sum(axis=1)).ravel() + np.asarray(M.sum(axis=0)).ravel()
    nodes = np.arange(M.shape[0])
    candidates = nodes[nodes != source]
    return _select(candidates, degree[candidates], m, rng, selection)


def strategy_temporal(A, beta, gamma, source, t0, Delta_t, m, n_samples,
                      rng=None, mapping="exact", selection="proportional"):
    """
    immunized = strategy_temporal(A, beta, gamma, source, t0, Delta_t, m, n_samples)
    
    Immunize proportional to p_tilde (Eq. 10): reasons about temporal distances in
    the ensemble rather than degree, so fewest doses are wasted.
    """
    rng = np.random.default_rng() if rng is None else rng
    q = immunization_probabilities(A, beta, gamma, source, t0, 0.0, n_samples, rng, mapping)
    p_tilde = immunization_probabilities(A, beta, gamma, source, t0, Delta_t, n_samples,
                                rng, mapping)
    candidates = susceptible_candidates(q, source)
    return _select(candidates, p_tilde[candidates], m, rng, selection)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def infection_times_with_delay(W, source, immunized, effective_from):
    """
    First-infection times on one realization, with immunity effective only from
    `effective_from` (= t0 + Delta_t). An immunized node reached at or after that
    is protected and transmits nothing.
    
    One Dijkstra pass is exact: nodes settle in non-decreasing order, so a node's
    time is final when popped, and cutting its out-arcs only delays later nodes.
    The circularity resolves in a single sweep.
    """
    n_nodes = W.shape[0]
    indptr, indices, data = W.indptr, W.indices, W.data

    is_immunized = np.zeros(n_nodes, dtype=bool)
    is_immunized[np.asarray(immunized, dtype=np.intp)] = True

    infection_time = np.full(n_nodes, np.inf)
    best = np.full(n_nodes, np.inf)
    settled = np.zeros(n_nodes, dtype=bool)

    best[source] = 0.0
    heap = [(0.0, int(source))]
    while heap:
        d, u = heappop(heap)
        if settled[u]:
            continue
        settled[u] = True
        if is_immunized[u] and d >= effective_from:
            continue                                  # protected in time
        infection_time[u] = d
        for k in range(indptr[u], indptr[u + 1]):
            v = indices[k]
            if settled[v]:
                continue
            nd = d + data[k]
            if nd < best[v]:
                best[v] = nd
                heappush(heap, (nd, int(v)))
    return infection_time


def outbreak_size(A, beta, gamma, source, immunized, n_samples,
                  rng=None, mapping="exact"):
    """
    infected = outbreak_size(A, beta, gamma, source, immunized, n_samples)
    
    Expected final outbreak with a *perfect* immunization: immunized nodes are removed
    from the network. The Delta_t -> 0 limit of outbreak_size_with_delay. 
    """
    rng = np.random.default_rng() if rng is None else rng
    A_imm = sp.csr_matrix(A).tolil(copy=True)
    V = np.asarray(immunized, dtype=np.intp)
    A_imm[V, :] = 0
    A_imm[:, V] = 0
    A_imm = A_imm.tocsr()
    A_imm.eliminate_zeros()

    total = 0.0
    for _ in range(n_samples):
        D = propagation_times(sample_realization(A_imm, beta, gamma, rng, mapping), source)
        total += float(np.isfinite(D).sum())
    return total / n_samples


def outbreak_size_with_delay(A, beta, gamma, source, immunized, t0, Delta_t,
                             n_samples, rng=None, mapping="exact"):
    """
    Expected final outbreak size when doses can be wasted — the paper's scenario,
    and the scoring Fig. 7 requires.
    """
    rng = np.random.default_rng() if rng is None else rng
    total = 0.0
    for _ in range(n_samples):
        W = sample_realization(A, beta, gamma, rng, mapping)
        times = infection_times_with_delay(W, source, immunized, t0 + Delta_t)
        total += float(np.isfinite(times).sum())
    return total / n_samples


def outbreak_curve(A, beta, gamma, source, immunized, t0, Delta_t, times,
                   n_samples, rng=None, mapping="exact"):
    """
    Expected number infected up to each time in `times` — what Fig. 7(b, c) plots.
    More informative than the final size when the epidemic saturates. No strategy
    can separate before t0 + Delta_t; equal strategies there are a correctness check.
    """
    rng = np.random.default_rng() if rng is None else rng
    grid = np.asarray(times, dtype=float)
    if not np.all(np.isfinite(grid)):
        raise ValueError(
            "times must all be finite; a node never infected has time inf, so an "
            "infinite horizon would count it as infected. Use "
            "outbreak_size_with_delay for the final size."
        )
    totals = np.zeros(grid.size)
    for _ in range(n_samples):
        W = sample_realization(A, beta, gamma, rng, mapping)
        infection = infection_times_with_delay(W, source, immunized, t0 + Delta_t)
        totals += (infection[None, :] <= grid[:, None]).sum(axis=1)
    return totals / n_samples


# --------------------------------------------------------------------------
# Fig. 7
# --------------------------------------------------------------------------

def compare_strategies(A, beta, gamma, source, t0, Delta_t, m, n_samples,
                       rng=None, mapping="exact", selection="proportional",
                       perfect_immunity=False, verbose=True):
    """
    size_random, size_hubs, size_temporal = compare_strategies(
        A, beta, gamma, source, t0, Delta_t, m, n_samples)
    
    Reproduce Fig. 7. Selection and scoring use disjoint draws. Lower is better;
    expected ordering temporal < random < hubs. With perfect_immunity=True the ordering inverts.
    """
    rng = np.random.default_rng() if rng is None else rng

    if verbose:
        print("Selecting immunization targets...")
    q = immunization_probabilities(A, beta, gamma, source, t0, 0.0, n_samples, rng, mapping)
    p_tilde = immunization_probabilities(A, beta, gamma, source, t0, Delta_t, n_samples,
                                rng, mapping)
    candidates = susceptible_candidates(q, source)

    V_random = _select(candidates, np.ones(candidates.size), m, rng, "proportional")
    V_hubs = strategy_hubs(A, source, m, rng, selection)
    V_temporal = _select(candidates, p_tilde[candidates], m, rng, selection)

    if verbose:
        print("Scoring strategies...")
    if perfect_immunity:
        def score(V):
            return outbreak_size(A, beta, gamma, source, V, n_samples, rng, mapping)
    else:
        def score(V):
            return outbreak_size_with_delay(A, beta, gamma, source, V, t0, Delta_t,
                                            n_samples, rng, mapping)

    size_random = score(V_random)
    size_hubs = score(V_hubs)
    size_temporal = score(V_temporal)

    if verbose:
        print("\n--- Immunization strategy comparison (Fig. 7) ---")
        print(f"  Random   strategy: {size_random:.1f} nodes infected")
        print(f"  Hubs     strategy: {size_hubs:.1f} nodes infected")
        print(f"  Temporal strategy: {size_temporal:.1f} nodes infected")
        print("  (lower = better; temporal should win in the time-critical regime)")

    return size_random, size_hubs, size_temporal
