# MultiScientist

MultiScientist is a multi-agent LLM team that simulates SIR spreading processes on complex networks and implements time-critical immunization strategies from:

> Tolic, D., Kleineberg, K.-K. & Antulov-Fantulin, N.
> Simulating SIR processes on networks using weighted shortest paths.
> *Scientific Reports* **8**, 6562 (2018).
> [doi:10.1038/s41598-018-24648-w](https://doi.org/10.1038/s41598-018-24648-w)

---

## Agent architecture

Three agents run one at a time: Analyst, Evaluator, and Observer. Agent coordination is orchestrated through the run directory, which is what makes each run resumable and auditable.

### Analyst agent

Reads `TASK.md` and the experiment log.

- **Audit:** Has the scenario already been evaluated?
- **Propose:** If not, write one proposal to `proposals/current.json`.
- **Document:** Once the evaluation result is in the log, interpret the final result for the best immunization strategy.

### Evaluation agent

Reads the proposal, calls run_scenario.py once for all strategies, writes the results and the seed to `logs/experiments.jsonl`, clears the proposal.

### Observer agent

Reports what was simulated, evaluated, and whether each agent stayed inside its role. Read-only. 

The observer agent reports the correctness check: before `t0 + Δt` no immunity has taken effect,
so all three network strategies must coincide.

### The loop

```
Analyst          audits, writes a proposal
    |
Evaluator        evaluates it, logs the evaluation results, clears the proposal
    |
Analyst          finds the log non-empty, interprets the results
    |
Observer         audits the run, reports the correctness check
    |
done             when the log is non-empty and the results are recorded
```

Each agent checks whether it has work this cycle and exits if not.

---

## The scenario parameters 

Epidemic parameters are stated in a task:

- `beta`, `gamma`, `source`, `t0` properties of the process/outbreak
- `delta_t` delay before the immunization takes effect
- `m` the dose budget, as a fraction of the network
- `n_samples` the ensemble size in Eq. 3 of the paper

All parameters are defined in `task-immunization/TASK.md`. Agents do not select, tune, or extend
parameters. To run a different scenario, edit the task file.

To reproduce the Fig. 7's values use: `beta = 0.03`, `gamma = 0.01`, `T_0 = 3`, `T_0 + tau = 13` so
`delta_t = 10`, `m = 0.2` of the network, source node 10, Petster network.

---

## Quickstart

Requires numpy, scipy and networkx.

```
python3 selftest.py          # verify the code
python3 launch.py my-run
cd ../my-run
# then read runbook.md and follow it
```

`selftest.py` checks the environment dependencies, engine, TASK.md, the named
network file and verifies that the MAS preserves what the simulation guarantees.

The simulation's own validation against the analytic results is in
`SpreadingMapping-py/test_spreading.py`.

`launch.py` builds a self-contained run directory: the system files, the SIR
simulation, the network data, the task, and one boot file per agent
`HEARTBEAT-<role>.md` with that role's instructions appended.

`runbook.md` is the orchestrator.

To evaluate a scenario by hand without running the loop:

```
python3 scorer/run_scenario.py --delta-t 10 --m 0.2 --n-samples 600
```

The network is read from `data/`, named by `network.path` in `TASK.md`.

Original Fig. 7 uses the [Petster social network](https://networkrepository.com/petster-hamster.php).

---

## The simulation

`SpreadingMapping-py/` is the original simulation algorithm implemented in python.
`scorer/run_scenario.py` calls its `strategy_random`, `strategy_hubs`,
`strategy_temporal` and `outbreak_curve`.

`compare_strategies` returns final outbreak sizes.

---

## Data

CSV with a `source,target` header, Network Repository `.edges`, or Matrix Market `.mtx`.

Node ids are relabelled to 0..n-1 in sorted order and the mapping is kept, so
`source: 10` in `TASK.md` refers to node 10 as numbered in the data file.

---

## Layout

```
selftest.py                verify the code
launch.py                  create a run directory
runbook.md                 orchestrator
LAUNCH.md                  settings cycle order, exit condition
system/helper.py           shared state
system/templates/          HEARTBEAT + the three role files
scorer/run_scenario.py     produces the Fig. 7 
task-immunization/TASK.md  the scenario
data/                      network datasets
SpreadingMapping-py/       the simulation method
```

---

## Extensions

Currently one scenario and three network immunization strategies. The agents do not search for new strategies.

Original Fig. 7 uses discrete-time SIR. This code draws exponential inter-event times (continuous time). 

Extensions: the concurrent configuration in which several analyst agents deliberate over the same task.
