#!/usr/bin/env python3
"""
Tests the code before running a task.

Three tests confirm the environment: the simulation code imports, TASK.md parses,
and the network dataset is present. Five other tests that the agent loop passes
the task's parameters to the simulation unaltered.

The simulation has its own tests in SpreadingMapping-py/test_spreading.py, which
check the mapping against the analytic results from the paper.

    python3 selftest.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

failures: list[str] = []

# A small generated network, so the wrapper checks do not depend on the data
# file being present or on any property of the real one.
TEST_NETWORK = {"kind": "barabasi_albert", "n": 200, "m": 2, "seed": 4}


def check(name, fn):
    try:
        print(f"  ok    {name} — {fn()}")
    except Exception as exc:
        failures.append(name)
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        if "-v" in sys.argv:
            traceback.print_exc()


def task_meta():
    from system.helper import _parse_frontmatter
    text = (HERE / "task-immunization" / "TASK.md").read_text(encoding="utf-8")
    return _parse_frontmatter(text.split("---", 2)[1])


# --- environment ---


def simulation_imports():
    sys.path.insert(0, str(HERE / "SpreadingMapping-py"))
    import interventions as iv
    needed = ["strategy_random", "strategy_hubs", "strategy_temporal",
              "outbreak_curve", "immunization_probabilities", "_select"]
    missing = [n for n in needed if not hasattr(iv, n)]
    if missing:
        raise AttributeError(f"simulation is missing {missing}")
    return f"all {len(needed)} required functions present"


def task_parses():
    sc = task_meta()["scenario"]
    return (f"beta={sc['beta']} gamma={sc['gamma']} t0={sc['t0']} "
            f"delta_t={sc['delta_t']} m={sc['m']}")


def data_file_present():
    from scorer.run_scenario import _resolve_data_path
    name = task_meta().get("network", {}).get("path")
    if not name:
        return "task uses a generated network; no data file needed"
    p = _resolve_data_path(name)
    return f"{p.name} present, {sum(1 for _ in p.open()) - 1} edge rows"


# --- the loader ---


def loader_reads_csv():
    """
    Node ids in these files are integers with gaps, and the loader maps them to
    0..n-1. It also drops self-loops and repeated edges.
    """
    from scorer.run_scenario import read_edge_list
    tmp = HERE / "_selftest.csv"
    tmp.write_text("source,target\n99,98\n999,550\n999,42\n99,98\n99,99\n")
    try:
        A, index_of = read_edge_list(tmp)
        if A.shape[0] != 5:
            raise AssertionError(f"expected 5 nodes, got {A.shape[0]}")
        if A.nnz // 2 != 3:
            raise AssertionError(f"expected 3 edges, got {A.nnz // 2}")
        if A.diagonal().sum() != 0:
            raise AssertionError("self-loop not dropped")
        if index_of != {42: 0, 98: 1, 99: 2, 550: 3, 999: 4}:
            raise AssertionError(f"ids mapped wrongly: {index_of}")
        return "header skipped, duplicate and self-loop dropped, ids mapped"
    finally:
        tmp.unlink(missing_ok=True)


# --- the wrapper passes parameters through unaltered ---


def source_resolves_by_id():
    """
    TASK.md names the source by its id in the data file. Fig. 7 uses node 10.
    Passing 10 straight through as a row index would start the epidemic from a
    different node without raising anything, so this runs a scenario on a
    network where the two differ and checks the epidemic reached the right node.
    """
    from scorer.run_scenario import evaluate_scenario
    tmp = HERE / "_selftest_src.csv"
    # Node 10 is at index 1 and has one neighbour, 77. Node 5 is at index 0.
    tmp.write_text("source,target\n5,10\n10,77\n")
    try:
        meta = task_meta()
        meta["network"] = {"kind": "csv", "path": str(tmp)}
        meta["scenario"] = dict(meta["scenario"], source=10, t0=0.0, beta=5.0,
                                gamma=0.001)
        r = evaluate_scenario(meta, delta_t=0.5, m=1, n_samples=40, seed=0)
        # Starting at node 10 reaches all three; starting at index 10 would
        # raise, since the network has only three nodes.
        final = r["scores"]["none"]
        if final < 2.5:
            raise AssertionError(
                f"epidemic from node 10 reached {final:.1f} of 3 nodes; "
                "the source was probably not translated"
            )
        return f"scenario from node 10 reached {final:.1f} of 3 nodes"
    finally:
        tmp.unlink(missing_ok=True)


def doses_from_fraction():
    """A fraction below 1 is a share of the network; an integer is a count."""
    from scorer.run_scenario import evaluate_scenario
    meta = task_meta()
    meta["network"] = TEST_NETWORK
    r = evaluate_scenario(meta, delta_t=2.0, m=0.2, n_samples=15, seed=0)
    if r["params"]["m"] != 40:
        raise AssertionError(f"0.2 of 200 is 40 doses, got {r['params']['m']}")
    r = evaluate_scenario(meta, delta_t=2.0, m=25, n_samples=15, seed=0)
    if r["params"]["m"] != 25:
        raise AssertionError(f"m=25 should stay 25, got {r['params']['m']}")
    return "0.2 of 200 -> 40 doses; m=25 -> 25 doses"


def strategies_agree_before_effect():
    """
    No immunity has acted before t0 + delta_t, so every strategy must give the
    same outbreak size until then.
    """
    from scorer.run_scenario import evaluate_scenario
    meta = task_meta()
    meta["network"] = TEST_NETWORK
    r = evaluate_scenario(meta, delta_t=2.0, m=0.2, n_samples=20, seed=0)
    spread = r["pre_effect_max_spread"]
    if spread > 1e-9:
        raise AssertionError(f"strategies differ before immunity acts: {spread}")
    return f"max spread {spread:.1e} before t0 + delta_t"


def scorer_runs():
    """The whole scenario, at the parameters the task declares."""
    from scorer.run_scenario import evaluate_scenario
    meta = task_meta()
    sc = meta["scenario"]
    r = evaluate_scenario(meta, delta_t=sc["delta_t"], m=sc["m"],
                          n_samples=25, seed=0)
    s = r["scores"]
    return (f"random {s['random']:.1f}  hubs {s['hubs']:.1f}  "
            f"temporal {s['temporal']:.1f}  -> {r['winner']}")


print("MultiScientist selftest\n")
check("simulation imports", simulation_imports)
check("TASK.md parses", task_parses)
check("network file present", data_file_present)
check("loader reads csv", loader_reads_csv)
check("source resolves by id", source_resolves_by_id)
check("doses from fraction", doses_from_fraction)
check("strategies agree before effect", strategies_agree_before_effect)
check("scorer runs", scorer_runs)

print()
if failures:
    print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    sys.exit(1)
print("All checks passed. `python3 launch.py <name>` to create a run.")
