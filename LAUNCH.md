# Settings for the task

Where `runbook.md` refers to a setting, the matching section below says what to do in each setting.

Agents run one at a time and coordinate through files in the run directory. 

The concurrent agent configuration via server, in which several analysts deliberate at once, is not covered here.

## Setting: launch_command

```
python3 launch.py <run-name> --task task-immunization
```

Creates a run directory containing its own copy of the system files, the
simulation, the network data, the task, one instruction file per agent, and this
file as `task-profile.md`.

## Setting: seeding_policy

Start one Analyst. It reads `TASK.md`, finds the experiment log empty, and
proposes the scenario the task declares. Nothing else is proposed, because the
scenario comes from the user.

## Setting: cycle_order

Analyst, then Evaluator, then Observer.

Each agent decides for itself whether it has work and exits if not, so the order
is a sequence of opportunities rather than a fixed schedule:

- In the first cycle the Analyst proposes, the Evaluator runs the scenario, and
  the Observer has one result to audit.
- In the second cycle the Analyst finds the scenario already evaluated and
  writes the finding. The Evaluator finds no proposal and exits. The Observer audits the finished run.

The Observer is read-only.

## Setting: evaluate_dispatch

Start one Evaluator. It reads `proposals/current.json`, calls
`scorer/run_scenario.py` at the sample count the task declares, appends one
record to `logs/experiments.jsonl`, and clears the proposal.

A single Evaluator runs the scenario.

<!-- Not used in this task, because the strategies are fixed and the scenario is
     evaluated once, so there is no champion to promote. Kept for a task that
     searches over best candidates.

## Setting: champion_promotion

After an evaluation the orchestrator checks that the record was written:

```python
assert state.read_experiments(), "evaluator reported no record"
```

-->

## Setting: stagnation_response

Not reachable in this task. The task declares one scenario and it is evaluated
once, so the run has a fixed length and there is nothing to stagnate on.

If the run does stall — the Analyst proposes nothing and the log is empty — that
is a fault rather than stagnation. Report it and stop; do not retry. The likely
causes are a malformed `TASK.md` or a missing network dataset.

## Setting: exit_condition

The run is finished when the scenario has been evaluated and the Analyst has
recorded the result.

```python
import json
sessions = [json.loads(l) for l in (RUN_ROOT / "logs" / "sessions.jsonl").open()]
done = bool(state.read_experiments()) and any(
    s["action"] == "reported" for s in sessions
)
```

The Analyst appends a session record with action `"reported"` after writing the
result. Without it the orchestrator cannot tell a run that has been evaluated
but not written up from one that is finished.

## Setting: final_report

The Analyst reports the interpretation of the result. The Observer reports what was run, the correctness check, and whether each agent was within its role.
