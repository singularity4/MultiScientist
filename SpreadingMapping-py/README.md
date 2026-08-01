# SpreadingMapping

Simulating SIR processes on complex networks via weighted shortest paths (python implementation).

> Tolic, D., Kleineberg, K-K. & Antulov-Fantulin, N. (2018).
> *Simulating SIR processes on networks using weighted shortest paths.*
> **Scientific Reports 8**, 6562. https://doi.org/10.1038/s41598-018-24648-w

Requires `numpy` and `scipy`; `networkx` for the tests.

## Files

| | |
|---|---|
| `SIR_mapping.py` | the mapping (Eq. 1–2), shortest paths, direct sampling |
| `mcmc_sir_exact.py` | Gibbs sampler, exact ensemble |
| `mcmc_sir_meanfield.py` | Gibbs sampler, mean-field ensemble |
| `interventions.py` | time-critical immunization (Eq. 10, Fig. 7) |
| `analytic.py` | closed forms used as validation targets |
| `test_spreading.py` | validation suite |

## Usage

```python
import numpy as np
from SIR_mapping import SIR_mapping, ensemble
from interventions import compare_strategies

# One realization: first-infection times from the source, inf = never infected.
D = SIR_mapping(A, N, E, beta, gamma, source)

# An ensemble, for expectations (Eq. 3).
D = ensemble(A, beta, gamma, source, n_samples=10_000)

# Fig. 7: three immunization strategies, lower outbreak is better.
compare_strategies(A, beta, gamma, source, t0, Delta_t, m, n_samples)
```

Nodes are 0-indexed. Every function takes an optional
`rng=np.random.default_rng(seed)` for reproducibility. Pass `mapping="meanfield"`
for the simplified mapping, valid when β ≫ γ.

## Implementation notes

**Inactive arcs are absent, not zero.** An arc whose transmitting node recovers
before it transmits has weight ∞ (Eq. 1). 

**Exact vs mean-field.** The exact mapping draws one recovery time per *node*
and one transmission time per *arc*, so all arcs leaving a node share a
deadline — the dynamical correlation. The mean-field mapping draws both per
edge, discarding it, and yields an undirected ensemble.

**Doses can be wasted.** A node immunized at t₀ is protected only if its
infection time reaches t₀ + Δt; otherwise the dose is lost and the node
transmits normally. `outbreak_size_with_delay` models this;
`outbreak_size` is the instant-immunity limit.

**Strategies select proportionally**, to p̃ and to degree, as the paper
describes. `selection="top"` takes the m largest instead.

**The Markov chains repair the shortest paths** after each batch of moves rather
than recomputing them (`DynamicSSSP`), which is the chain's runtime advantage in
the paper's sampling section. Output is bit-identical to recomputing;
`incremental=False` recomputes. Measured speed-up against recomputation:

| nodes | thin = 1 | thin = 10 | thin = 50 |
|---|---|---|---|
| 200 | 7.6× | 2.0× | 0.8× |
| 500 | 11.5× | 3.1× | 1.1× |
| 1000 | 9.5× | 5.2× | 1.6× |

The gain falls as thinning rises, since more accumulated moves invalidate more
of the tree, and grows with network size. The advantage applies when
realizations from *all* sources are needed, as in source detection; with one
known source, direct sampling via `ensemble` remains competitive.

## Validation

`pytest test_spreading.py` — seven tests.

The toy network of Fig. 3 has an analytic source-to-destination probability that
*differs between the two mappings*, so it detects a sampler that has silently
lost its dynamical correlations. Pooled over 5 × 60,000 realizations:

| | simulated | analytic | |
|---|---|---|---|
| exact | 0.64150 | 0.642117 | −0.70 s.e. |
| mean-field | 0.72581 | 0.724941 | +1.07 s.e. |

Both Gibbs samplers reach the same targets, and the incremental path repair is
checked to reproduce full recomputation exactly.

For Fig. 7, on a Barabási–Albert network with β = 0.5, γ = 0.2, t₀ = 1, Δt = 2,
m = 0.2 N: doses still unspent when immunity takes effect are ≈ 15.0
(temporal), 13.0 (random), 9.9 (hubs), so hubs waste the most; with instant
immunity hubs is far the best, and with the delay modelled it is never best.

The port has also been run on the Petster network at the paper's Fig. 7
parameters (β = 0.03, γ = 0.01, T₀ = 3, T₀ + τ = 13, m = 0.2 N, source node 10),
reproducing the published results.

---

Reference implementation (MATLAB): https://github.com/singularity4/SpreadingMapping
