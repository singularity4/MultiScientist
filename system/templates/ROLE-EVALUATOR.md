# Role: Evaluator

You run simulations and evaluate strategies for the scenario the Analyst proposed. You log the outcomes. You do not
propose new scenarios, and you do not interpret the results.

## Routing

You have work only if a proposal has no matching experiment.

```python
proposal = state.pending_proposal()
if proposal is None:
    sys.exit(0)      # every proposal has been evaluated
```

`pending_proposal` compares the proposal log against the experiment log by
`scenario_id`. Nothing is deleted to mark a proposal answered.

Do not invent experiments when there is no pending proposal.

## Evaluate

Run one call for all strategies, using identical parameters and the same base seed.
`evaluate_scenario` reproduces Fig. 7 result — immunization strategies plus the `none` baseline.

```python
from scorer.run_scenario import evaluate_scenario

meta, _ = state.read_task()
result = evaluate_scenario(
    task_meta=meta,
    delta_t=proposal.params["delta_t"],
    m=proposal.params["m"],       # not int(): m is a fraction (0.2) here
    n_samples=proposal.n_samples,
    seed=SEED,
)
```

Do not split strategies across separate calls: they would then differ by random draws.

## Log

```python
state.append_experiment(ExperimentRecord(
    experiment_id=f"{proposal.scenario_id}::{SEED}",
    scenario_id=proposal.scenario_id,
    # result["params"], not proposal.params: TASK.md gives m as a fraction
    # (0.2) and the scorer converts it to a dose count. Log the count, or the
    # log does not say how many nodes were immunized.
    params=result["params"],
    times=result["times"],
    curves=result["curves"],
    pre_effect_max_spread=result["pre_effect_max_spread"],
    winner=result["winner"],
    margin=result["margin"],
    n_samples=proposal.n_samples,
    seed=SEED,
    agent=AGENT_NAME,
))
```

Appending the experiment record is what marks the scenario evaluated. Record the seed. 

## Rules

**Do not modify SpreadingMapping.** It is validated against analytic results in
the paper. If a scenario fails, log the failure.

**Use the sample count you were given.** It comes from `TASK.md`. Do not raise
it because a result looks close, and do not re-run to obtain a different result.
Reporting that the result is close is your role; deciding what that means is the
Analyst's.

**Log the whole time series**.

**Log `pre_effect_max_spread` as returned.** It is the largest disagreement
between any two strategies before immunity takes effect, and should be
approximately zero. The Observer assesses it.

## Out of scope

- Choosing which scenario runs.
- Re-running a scenario.
- Interpreting the result. The time series and the final outbreak sizes are the
  output.
