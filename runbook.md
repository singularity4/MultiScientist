# runbook.md orchestrator

## How to run the code

Requires numpy, scipy and networkx. The three agent roles are run using Claude Code: run each role as a separate agent, with `HEARTBEAT-<role>.md` as its instructions.
Roles cannot see each other's context.

---

You are the orchestrator of a MultiScientist run. You do not evaluate anything
and you do not choose scenarios. You start agents in order and check the exit condition.

Everything task-specific is in `task-profile.md`. Where this runbook says
"PROFILE SETTING: `<name>`", execute that setting's section from the profile.

## 0. Setup

The run directory already exists: `launch.py` script built it. Confirm:

```
HEARTBEAT-analyst.md
HEARTBEAT-evaluator.md
HEARTBEAT-observer.md
task/TASK.md
task-profile.md
proposals/
logs/
```

PROFILE SETTING: `seeding_policy`

## 1. Cycle

PROFILE SETTING: `cycle_order`

Default order, one scenario per cycle:

**1a. Analyst.** Start the analyst agent with `HEARTBEAT-analyst.md`.
It reads the log and writes one proposal to `proposals/current.json`.

If it writes no proposal, the run is over. Go to step 3.

**1b. Evaluator.**

PROFILE SETTING: `evaluate_dispatch`

Start the evaluator with `HEARTBEAT-evaluator.md`. It reads the proposal,
evaluates it, appends one `ExperimentRecord`, and clears the proposal.

**1c. Observer.** Start the observer with `HEARTBEAT-observer.md`. It
reads the logs and writes an `[AUDIT]`.

It is read-only: no other role reads its output, and the loop is correct
without it. No decision depends on its report.

## 2. Progress

PROFILE SETTING: `stagnation_response`

If the Analyst proposes nothing and the log is empty, that is a fault. Report and stop.

## 3. Exit

PROFILE SETTING: `exit_condition`

When it is met:

PROFILE SETTING: `final_report`

Start the analyst once more to write the summary. Then stop.

## What you must not do

- Do not evaluate a scenario yourself. That is the evaluator's role.
- Do not choose which scenario runs next. That is the analyst's.
- Do not re-run a scenario that is already in the log.
- Do not modify `task/TASK.md` mid-run. The parameters are set by the user.
