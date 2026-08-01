"""
Validation against the paper's analytical results (analytic.py).

Run: pytest test_spreading.py
"""

import numpy as np
import pytest
import scipy.sparse as sp

from SIR_mapping import ensemble, sample_realization
from analytic import p_nk, toy_chain_graph, toy_exact, toy_meanfield, transmissibility
from interventions import (
    infection_times_with_delay,
    outbreak_size,
    outbreak_size_with_delay,
    strategy_hubs,
    strategy_random,
    strategy_temporal,
)
from mcmc_sir_exact import mcmc_sir_exact
from mcmc_sir_meanfield import mcmc_sir_meanfield


# --------------------------------------------------------------------------
# Setup for the immunization tests
# --------------------------------------------------------------------------

BETA, GAMMA, SOURCE, M_DOSES, T0, DELTA_T = 0.5, 0.2, 1, 40, 1.0, 2.0


@pytest.fixture(scope="module")
def network():
    """
    The paper used the empirical Petster network. This is a Barabási-Albert
    synthetic network fallback, which shares the property that matters: heavy-tailed degrees,
    so there are hubs for the epidemic to reach early.
    """
    nx = pytest.importorskip("networkx")
    G = nx.barabasi_albert_graph(200, 2, seed=4)
    return nx.to_scipy_sparse_array(G, format="csr", dtype=float)


@pytest.fixture(scope="module")
def strategies(network):
    """One immunization set per strategy, chosen once and reused."""
    rng = np.random.default_rng(0)
    return {
        "temporal": strategy_temporal(network, BETA, GAMMA, SOURCE, T0, DELTA_T,
                                      M_DOSES, 600, rng),
        "random": strategy_random(network, BETA, GAMMA, SOURCE, T0, M_DOSES, 600, rng),
        "hubs": strategy_hubs(network, SOURCE, M_DOSES, rng),
    }


# --------------------------------------------------------------------------
# 1
# --------------------------------------------------------------------------

def test_eq7_is_self_consistent():
    """
    p_{n,k} must be a probability distribution, must collapse to Eq. 5 for a
    single link, and must *not* be binomial. If it were binomial the exact
    mapping would carry no dynamical correlations and test 2 would have nothing
    to detect.
    """
    for beta, gamma in [(1.0, 1.0), (0.5, 2.0), (3.0, 0.7)]:
        for n in (1, 5, 20):
            assert sum(p_nk(n, k, beta, gamma)
                       for k in range(n + 1)) == pytest.approx(1.0)
        assert p_nk(1, 1, beta, gamma) == pytest.approx(transmissibility(beta, gamma))

    # For beta = gamma the Gamma ratio collapses to 1/(n+1), independent of k.
    for k in range(21):
        assert p_nk(20, k, 1.0, 1.0) == pytest.approx(1.0 / 21)

    from math import comb
    n, p = 10, transmissibility(1.0, 1.0)
    assert not np.allclose(
        [p_nk(n, k, 1.0, 1.0) for k in range(n + 1)],
        [comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)],
        atol=1e-3,
    )


# --------------------------------------------------------------------------
# 2
# --------------------------------------------------------------------------

def test_sampler_matches_fig3_for_both_mappings():
    """
    The headline validation. On the toy network the source-to-destination
    probability is analytic and *differs between the two mappings*, so this
    fails if any part of Eq. 1/2, the arc bookkeeping or the shortest-path step
    is wrong — including a mean-field sampler masquerading as exact.
    """
    n_chains, length, beta, gamma = 20, 3, 1.0, 1.0
    A = toy_chain_graph(n_chains, length)
    target_exact = toy_exact(n_chains, length, beta, gamma)
    target_meanfield = toy_meanfield(n_chains, length, beta, gamma)
    assert target_meanfield - target_exact > 0.05, "targets must be distinguishable"

    n_samples, tolerance = 15_000, 0.02          # ~6 s.e.; targets 0.083 apart

    D = ensemble(A, beta, gamma, 0, n_samples,
                 rng=np.random.default_rng(2), mapping="exact")
    assert np.isfinite(D[:, 1]).mean() == pytest.approx(target_exact, abs=tolerance)

    D = ensemble(A, beta, gamma, 0, n_samples,
                 rng=np.random.default_rng(3), mapping="meanfield")
    assert np.isfinite(D[:, 1]).mean() == pytest.approx(target_meanfield, abs=tolerance)


# --------------------------------------------------------------------------
# 3
# --------------------------------------------------------------------------

def test_inactive_arcs_are_never_stored_as_zero():
    """
    An arc whose transmitting node recovers first has weight infinity (Eq. 1) and
    must be absent. Stored as a zero it would be a free instantaneous arc — the
    opposite — and everything downstream would be wrong while looking plausible.
    """
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    rng = np.random.default_rng(0)
    for _ in range(200):
        W = sample_realization(A, beta=0.05, gamma=20.0, rng=rng)  # nearly all inactive
        assert np.all(W.data > 0.0)


# --------------------------------------------------------------------------
# 4
# --------------------------------------------------------------------------

def test_both_mcmc_chains_match_analytic():
    """
    The chains must reach the same stationary distribution as direct sampling.
    A chain updating the wrong block of variables would land on the *other*
    mapping's target, so the two are also required to differ from each other.
    """
    n_chains, length, beta, gamma = 8, 2, 1.0, 1.0
    A = toy_chain_graph(n_chains, length)
    tolerance = 0.03                              # samples are correlated

    exact = np.isfinite(mcmc_sir_exact(
        A, beta, gamma, 0, 6000, 500, 20, np.random.default_rng(11))[:, 1]).mean()
    assert exact == pytest.approx(
        toy_exact(n_chains, length, beta, gamma), abs=tolerance)

    meanfield = np.isfinite(mcmc_sir_meanfield(
        A, beta, gamma, 0, 6000, 500, 30, np.random.default_rng(12))[:, 1]).mean()
    assert meanfield == pytest.approx(
        toy_meanfield(n_chains, length, beta, gamma), abs=tolerance)

    assert meanfield > exact

    # Repairing the paths incrementally after each batch of moves must give
    # exactly what recomputing them gives — not merely the same distribution.
    nx = pytest.importorskip("networkx")
    G = nx.barabasi_albert_graph(150, 2, seed=1)
    B = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    for chain in (mcmc_sir_exact, mcmc_sir_meanfield):
        repaired = chain(B, 0.5, 0.2, 1, 60, 20, 7,
                         np.random.default_rng(4), incremental=True)
        recomputed = chain(B, 0.5, 0.2, 1, 60, 20, 7,
                           np.random.default_rng(4), incremental=False)
        np.testing.assert_array_equal(repaired, recomputed)


# --------------------------------------------------------------------------
# 5
# --------------------------------------------------------------------------

def test_immunization_scoring_on_a_hand_checkable_graph():
    """
    Path 0 -> 1 -> 2 -> 3 with unit delays, so every infection time is known by
    hand. Covers the cases that matter: no immunization, a dose that lands, a
    dose that arrives too late, the inclusive boundary of Theta in Eq. 10, and
    the source, infected at t = 0 and beyond saving.
    """
    path = sp.csr_matrix(np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ]))
    unimmunized = [0.0, 1.0, 2.0, 3.0]

    np.testing.assert_allclose(
        infection_times_with_delay(path, 0, [], np.inf), unimmunized)

    # Node 1 is infected at t = 1; immunity effective from t = 0.5 protects it.
    np.testing.assert_allclose(
        infection_times_with_delay(path, 0, [1], 0.5), [0.0, np.inf, np.inf, np.inf])

    # Effective only from t = 2.0: the dose is wasted and the spread continues.
    np.testing.assert_allclose(
        infection_times_with_delay(path, 0, [1], 2.0), unimmunized)

    # Theta is one when the infection time equals the threshold.
    assert np.isinf(infection_times_with_delay(path, 0, [1], 1.0)[1])

    np.testing.assert_allclose(
        infection_times_with_delay(path, 0, [0], 0.5), unimmunized)


# --------------------------------------------------------------------------
# 6
# --------------------------------------------------------------------------

def test_outbreak_size_equals_zero_delay_scoring(network, strategies):
    """
    `outbreak_size` deletes immunized nodes outright, which is instant immunity.
    `outbreak_size_with_delay` at Delta_t = 0 reaches the same place by refusing
    every infection arriving at an immunized node. Different code paths, same
    process, so they must agree.
    """
    V = strategies["random"]
    removal = outbreak_size(
        network, BETA, GAMMA, SOURCE, V, 1500, np.random.default_rng(3))
    zero_delay = outbreak_size_with_delay(
        network, BETA, GAMMA, SOURCE, V, 0.0, 0.0, 1500, np.random.default_rng(4))
    assert removal == pytest.approx(zero_delay, rel=0.03)


# --------------------------------------------------------------------------
# 7
# --------------------------------------------------------------------------

def test_fig7_the_delay_inverts_which_strategy_wins(network, strategies):
    """
    The finding in three parts: hubs waste the most doses ("the hubs usually are
    infected earlier"); with every dose landing, hubs is far the best; with the
    delay modelled that advantage is gone. The inversion is the result.
    """
    D = ensemble(network, BETA, GAMMA, SOURCE, 400, rng=np.random.default_rng(6))
    landed = {name: float((D[:, np.asarray(V)] >= T0 + DELTA_T).sum(axis=1).mean())
              for name, V in strategies.items()}
    assert landed["temporal"] > landed["random"] > landed["hubs"]
    assert landed["temporal"] - landed["hubs"] > 3.0, "the waste gap should be wide"

    perfect = {name: outbreak_size(network, BETA, GAMMA, SOURCE, V, 250,
                                   np.random.default_rng(7))
               for name, V in strategies.items()}
    assert perfect["hubs"] < perfect["random"]
    assert perfect["hubs"] < perfect["temporal"]

    delayed = {name: outbreak_size_with_delay(network, BETA, GAMMA, SOURCE, V,
                                              T0, DELTA_T, 600,
                                              np.random.default_rng(8))
               for name, V in strategies.items()}
    assert delayed["hubs"] > min(delayed["temporal"], delayed["random"])

