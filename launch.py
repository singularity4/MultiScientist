#!/usr/bin/env python3
"""
Creates a run directory for one task. A run directory is self-contained: it holds its own copy of the system files,
the SIR simulation, network data, the task, one instruction file per agent, and the shared-state folders the agents read and write. 
The template directory is not modified, so several runs can be started from it without interfering.

Usage:
    python3 launch.py                    # name from task + timestamp
    python3 launch.py my-run
    python3 launch.py my-run --task task-immunization
    python3 launch.py my-run --output-dir /tmp/runs

Each task directory must contain TASK.md, and a LAUNCH.md must exist either in
that directory or in one of its parents. LAUNCH.md holds the settings the
orchestrator reads.

Then:
    cd <run-dir>
    # open runbook.md in a Claude Code session
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


TEMPLATE_DIR = Path(__file__).resolve().parent

# The two agent roles and the role file each boots from.
#
# Each agent reads HEARTBEAT.md with its role file appended.
ROLE_FILES = {
    "analyst": "ROLE-ANALYST.md",
    "evaluator": "ROLE-EVALUATOR.md",
    "observer": "ROLE-OBSERVER.md",
}


def find_task_profile(task_dir: Path) -> Path:
    """
    Find the LAUNCH.md that governs a task.

    Searches the task directory, then its parents, so several tasks may share
    one LAUNCH.md while any of them can override it.

    Raises:
        FileNotFoundError: if no LAUNCH.md is found.
    """
    current = task_dir.resolve()
    template_root = TEMPLATE_DIR.resolve()
    while True:
        candidate = current / "LAUNCH.md"
        if candidate.exists():
            return candidate
        if current == template_root or template_root not in current.parents:
            raise FileNotFoundError(
                f"No LAUNCH.md found walking up from {task_dir} to {template_root}. "
                "Every task must ship a LAUNCH.md (there is no generic fallback)."
            )
        current = current.parent


def build_run_directory(run_root: Path, task_dir: Path, profile_path: Path) -> None:
    """
    Assemble a fresh run directory.

    Args:
        run_root:     Directory to create. Must not already exist.
        task_dir:     Task directory containing TASK.md.
        profile_path: The LAUNCH.md found by find_task_profile.
    """
    run_root.mkdir(parents=True, exist_ok=False)

    shutil.copytree(TEMPLATE_DIR / "system", run_root / "system")
    shutil.copytree(TEMPLATE_DIR / "scorer", run_root / "scorer")

    # The simulation is copied in too, so a run directory is
    # self-contained and TASK.md can use a path relative to the run root.
    simulation = TEMPLATE_DIR / "SpreadingMapping-py"
    if simulation.exists():
        shutil.copytree(simulation, run_root / "SpreadingMapping-py")

    # Copied to a fixed name, so agents need not know the task directory name.
    shutil.copytree(task_dir, run_root / "task")

    # Network files. The scorer resolves the path named in TASK.md against
    # <run>/data/.
    data_dir = TEMPLATE_DIR / "data"
    if data_dir.exists():
        shutil.copytree(data_dir, run_root / "data")

    shutil.copy(TEMPLATE_DIR / "runbook.md", run_root / "runbook.md")
    shutil.copy(profile_path, run_root / "task-profile.md")

    # One instruction file per agent at the top of the run directory: the
    # shared heartbeat with that role appended.
    heartbeat = (TEMPLATE_DIR / "system" / "templates" / "HEARTBEAT.md").read_text(
        encoding="utf-8"
    )
    for role, role_filename in ROLE_FILES.items():
        role_body = (
            TEMPLATE_DIR / "system" / "templates" / role_filename
        ).read_text(encoding="utf-8")

        booted = heartbeat.replace("<injected by launch.py>", role)
        booted += "\n\n---\n\n# Your Role\n\n" + role_body
        (run_root / f"HEARTBEAT-{role}.md").write_text(booted, encoding="utf-8")

    # Create the folders the agents read and write.
    sys.path.insert(0, str(run_root))
    from system.helper import FolderState

    FolderState(run_root).initialise()


def main() -> None:
    """Parse arguments, resolve the task and profile, build the run directory."""
    ap = argparse.ArgumentParser(
        description="Create a fresh MultiScientist run directory."
    )
    ap.add_argument("name", nargs="?", default=None, help="run name")
    ap.add_argument(
        "--task",
        default="task-immunization",
        help="task directory inside the template",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="where to create the run (default: template's parent)",
    )
    args = ap.parse_args()

    task_dir = TEMPLATE_DIR / args.task
    if not (task_dir / "TASK.md").exists():
        raise SystemExit(f"No TASK.md in {task_dir}")

    profile_path = find_task_profile(task_dir)

    name = args.name or (
        f"{args.task}-{datetime.now(timezone.utc).strftime('%m%d_%H%M')}"
    )
    out_dir = Path(args.output_dir) if args.output_dir else TEMPLATE_DIR.parent
    run_root = out_dir / name

    build_run_directory(run_root, task_dir, profile_path)

    print(f"Run created: {run_root}")
    print(f"  task    : {args.task}")
    print(f"  profile : {profile_path.relative_to(TEMPLATE_DIR)}")
    print(f"  agents  : {', '.join(ROLE_FILES)}")
    print(f"            HEARTBEAT-<role>.md, one per agent")
    print()
    print(f"  cd {run_root}")
    print("  # then open runbook.md in a Claude Code session")


if __name__ == "__main__":
    main()
