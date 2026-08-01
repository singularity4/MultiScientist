"""
Shared state for a run. Agents run one at a time and do not communicate directly; each reads what the
others did from the files below. Each run can therefore be audited a posteriori.

Files in a run directory:

    logs/proposals.jsonl       one line per scenario the Analyst proposed
    logs/experiments.jsonl     one line per scenario evaluated
    logs/sessions.jsonl        one line per agent cycle
    task/TASK.md               the task, copied in by launch.py
    task-profile.md            settings, copied from LAUNCH.md
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _now() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def scenario_id(params: dict[str, float]) -> str:
    """
    Stable identifier for a scenario.

    Built from the swept parameters only, sorted, so the same scenario always
    produces the same id regardless of dict ordering. Used as the key in the
    experiment log.

    Example:
        >>> scenario_id({"m": 40, "delta_t": 2.0})
        'delta_t=2.0|m=40'
    """
    return "|".join(f"{k}={v}" for k, v in sorted(params.items()))


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class ScenarioProposal:
    """
    A scenario the Analyst proposes for evaluation.

    Attributes:
        scenario_id: Stable id, from `scenario_id(params)`.
        params:      The scenario parameters, typically delta_t and m.
        n_samples:   Ensemble size passed to the scorer.
        reason:      One line on why this scenario was proposed.
        proposed_by: Agent name.
    """

    scenario_id: str
    params: dict[str, float]
    n_samples: int
    reason: str
    proposed_by: str
    proposed_at: str = field(default_factory=_now)


@dataclass
class ExperimentRecord:
    """
    The result of evaluating one scenario.

    All strategies are scored in one call under identical parameters and from
    the same base seed, so the results are comparable.

    Attributes:
        experiment_id: Unique id for this evaluation.
        scenario_id:   Which scenario.
        params:        The scenario parameters, repeated here so the log is
                       readable without joining against proposals.
        times:         The time grid the series are on.
        curves:        {strategy: expected infected up to each time}, including
                       the "none" baseline — the outbreak with no immunization.
                       This is Fig. 7 of the paper.
        pre_effect_max_spread:
                       Largest disagreement between any two strategies before
                       immunity takes effect. Approximately zero, since no
                       immunity has yet acted.
        winner:        Name of the lowest-scoring strategy. Lower is better.
        margin:        Difference between the best and second-best score.
                       Reported, not acted on.
        n_samples:     Ensemble size used.
        seed:          Seed, so the result is exactly reproducible.
        agent:         Which evaluator produced it.
        recorded_at:   Timestamp.

    The scorer is stochastic, and the margin between strategies is a small
    fraction of the outbreak size.
    """

    experiment_id: str
    scenario_id: str
    params: dict[str, float]
    times: list[float]
    curves: dict[str, list[float]]
    pre_effect_max_spread: float
    winner: str
    margin: float
    n_samples: int
    seed: int
    agent: str
    recorded_at: str = field(default_factory=_now)

    def scores(self) -> dict[str, float]:
        """
        Final outbreak size for every strategy, including the "none" baseline.
        """
        return {name: float(c[-1]) for name, c in self.curves.items()}


@dataclass
class SessionRecord:
    """One agent cycle, for the run's audit trail."""

    agent: str
    action: str
    detail: str
    at: str = field(default_factory=_now)


# ---------------------------------------------------------------------------
# The state
# ---------------------------------------------------------------------------


class FolderState:
    """
    Read and write a run directory.

    Logs are append-only. The proposal file is rewritten each cycle, since only
    one proposal is active at a time.

    Args:
        run_root: The run directory.

    Example:
        >>> state = FolderState(run_root)
        >>> meta, body = state.read_task()
        >>> proposal = state.pending_proposal()
    """

    def __init__(self, run_root: Path | str) -> None:
        self.root = Path(run_root)
        self.logs = self.root / "logs"

    # -- setup --------------------------------------------------------------

    def initialise(self) -> None:
        """Create the shared-state folders and empty logs. Idempotent."""
        self.logs.mkdir(parents=True, exist_ok=True)
        (self.logs / "proposals.jsonl").touch(exist_ok=True)
        (self.logs / "experiments.jsonl").touch(exist_ok=True)
        (self.logs / "sessions.jsonl").touch(exist_ok=True)

    # -- the task -----------------------------------------------------------

    def read_task(self) -> tuple[dict[str, Any], str]:
        """
        Read TASK.md.

        The frontmatter holds the parameters (beta, gamma, source, t0, network,
        delta_t, m, n_samples); the body describes the question for the agents
        to read. Changing the scenario means editing TASK.md, not the code.

        Returns:
            (metadata, body) — frontmatter as a dict, Markdown body as text.

        Raises:
            FileNotFoundError: if task/TASK.md is missing.
        """
        path = self.root / "task" / "TASK.md"
        if not path.exists():
            raise FileNotFoundError(
                f"No TASK.md at {path}. launch.py copies the chosen task into "
                "<run>/task/ — if it is missing, the run was not built by launch.py."
            )
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return {}, text
        _, front, body = text.split("---", 2)
        return _parse_frontmatter(front), body.strip()

    # -- proposals ----------------------------------------------------------

    def append_proposal(self, proposal: ScenarioProposal) -> None:
        """Append one proposal to the proposal log."""
        self._append(self.logs / "proposals.jsonl", proposal)

    def read_proposals(self) -> list[ScenarioProposal]:
        """Every proposal made so far, in order."""
        return [
            ScenarioProposal(**d) for d in self._read(self.logs / "proposals.jsonl")
        ]

    def pending_proposal(self) -> ScenarioProposal | None:
        """
        The oldest proposal with no matching experiment, or None if there is none.

        The Evaluator uses this to decide whether it has work. A proposal is
        answered when an experiment with the same `scenario_id` is in the
        experiment log, so nothing has to be deleted to mark it done.
        """
        evaluated = {r.scenario_id for r in self.read_experiments()}
        for proposal in self.read_proposals():
            if proposal.scenario_id not in evaluated:
                return proposal
        return None

    # -- experiments --------------------------------------------------------

    def append_experiment(self, record: ExperimentRecord) -> None:
        """Append one evaluated scenario to the canonical log."""
        self._append(self.logs / "experiments.jsonl", record)

    def read_experiments(self) -> list[ExperimentRecord]:
        """
        Every scenario evaluated so far.

        The Analyst reads this to check whether the task's scenario has already
        been run.
        """
        return [ExperimentRecord(**d) for d in self._read(self.logs / "experiments.jsonl")]

    def experiments_for(self, sid: str) -> list[ExperimentRecord]:
        """All evaluations of one scenario — more than one if it was resampled."""
        return [r for r in self.read_experiments() if r.scenario_id == sid]

    # -- sessions -----------------------------------------------------------

    def append_session(self, record: SessionRecord) -> None:
        """Log one agent cycle."""
        self._append(self.logs / "sessions.jsonl", record)

    # -- internals ----------------------------------------------------------

    def _append(self, path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(obj), separators=(",", ":")) + "\n")

    def _read(self, path: Path) -> Iterator[dict[str, Any]]:
        if not path.exists():
            return iter(())
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """
    Parse the YAML subset used in TASK.md.

    Handles scalars, inline lists, and one level of nesting. Implemented here
    rather than with PyYAML to avoid an additional dependency.
    """
    meta: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, meta)]

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        target = stack[-1][1]

        if not value:
            child: dict[str, Any] = {}
            target[key] = child
            stack.append((indent, child))
            continue

        target[key] = _coerce(value)
    return meta


def _coerce(value: str) -> Any:
    """Convert a YAML scalar or inline list to a Python value."""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_coerce(v.strip()) for v in inner.split(",")]
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("'\"")

