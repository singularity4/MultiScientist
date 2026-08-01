# HEARTBEAT — agent instruction file

You are the **<injected by launch.py>** agent in a MultiScientist run.

This is the sequential folder configuration: agents run one at a time and
coordinate only through files in the run directory. You never speak to another
agent. Everything you need to know about what others did, you read from shared
state.

Work through the four parts below in order.

---

## Part 0 — Decide whether you have work

Decide whether you have work this cycle. If not, exit cleanly. Do not invent
work to justify the cycle.

Your routing rule is the **Routing** section of your role, appended below. Apply
it. If it says you have no work, stop here.

```python
import sys
from pathlib import Path

RUN_ROOT = Path(".")            # this file sits in the run directory
sys.path.insert(0, str(RUN_ROOT))
from system.helper import FolderState

state = FolderState(RUN_ROOT)
```

Start with the run directory as your working directory. If you did not, use the
directory containing this file.

## Part 1 — Read the task

Read your identity, then the task, then the state.

```python
AGENT_NAME = "<the role named at the top of this file>"

meta, body = state.read_task()       # TASK.md parameters + description
experiments = state.read_experiments()
```

`meta` carries every fixed parameter of the run — the epidemic, the network,
the scenario, the sample count. **Read them; do not hardcode anything.**
The user sets these. A new scenario is a new task file, not a code change.

Read `task-profile.md` for the settings that govern this run.

---

## Part 2 — Work

Do your role, as specified in the "Your Role" section appended below.

Stay within it. The agent that proposes an experiment does not measure it, and
the agent that measures does not interpret the measurement.

---

## Part 3 — Exit

Log your cycle, then stop.

```python
from system.helper import SessionRecord

state.append_session(SessionRecord(
    agent=AGENT_NAME,
    action="proposed" | "evaluated" | "reported" | "audited",
    detail="one line on what you did and why",
))
```

Then print a closing line so the orchestrator knows the cycle ended:

```
[CYCLE-DONE] <role> — <one line>
```

Do not continue past this. The orchestrator starts the next agent.
