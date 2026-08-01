---
task: node immunization
question: reproduce Fig. 7 — outbreak size over time per immunization strategy
metric: expected infected nodes up to time t

scenario:
  beta: 0.03
  gamma: 0.01
  source: 10
  t0: 3.0
  delta_t: 10.0
  m: 0.2
  n_samples: 600
  mapping: exact
  selection: proportional
  t_max: 40.0
  t_step: 1.0

strategies: [random, hubs, temporal]
---

# Task — reproduce Fig. 7

## The question

Fig. 7 of the paper plots the total number of infected nodes up to time t, one
result per immunization strategy, with vertical lines at T_0 = 3 and
T_0 + tau = 13.

Evaluate the strategies at the scenario declared above and report the outbreak
size over time for each.

## The parameters

- **beta, gamma, source, t0** — properties of the outbreak
- **delta_t** — delay before the immunization takes effect
- **m** — doses, as a fraction of the network
- **n_samples** — ensemble size

To evaluate a different scenario, edit this file.

## The immunization strategies

- **random** — uniform over nodes still susceptible at t0
- **hubs** — by degree
- **temporal** — by the probability of still being uninfected at t0 + delta_t

A fourth result, **none**, is always computed: it immunizes nobody and is the
unmitigated baseline. It is not a candidate and cannot win.

## The network

Petster social network (2426 nodes).

network:
  kind: csv
  path: petster.csv
  
## Provenance

Fig. 7: beta = 0.03, gamma = 0.01, T_0 = 3, T_0 + tau = 13 so delta_t = 10,
m = 0.2 of the total number of nodes, source node 10. Panel (c) uses beta = 0.05
with the rest unchanged.

## Constraints

- Do not modify SpreadingMapping.
- Score all strategies in one call, under identical parameters.
- Evaluate the scenario once, at the declared sample count.
