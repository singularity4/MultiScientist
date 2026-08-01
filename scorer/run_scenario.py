#!/usr/bin/env python3
"""
Evaluates one scenario by calling SpreadingMapping.

Builds the synthetic BA network, selects each strategy's immunization node set with the
simulation's `strategy_*` functions, and returns the outbreak size over time for
each, via `outbreak_curve`.

It does not use `compare_strategies`, which returns final outbreak sizes only. 

Fig. 7 of the paper plots the total number of infected nodes up to time t, one
result per strategy, with vertical lines at T_0 and T_0 + tau. This module
returns those results, via the simulation's `outbreak_curve`.

From an agent:

    from scorer.run_scenario import evaluate_scenario
    result = evaluate_scenario(task_meta=meta, delta_t=2.0, m=40,
                               n_samples=600, seed=7)

From the command line, to check a scenario by hand:

    python3 scorer/run_scenario.py --delta-t 2.0 --m 40 --n-samples 600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp


# --- network construction ---


def build_network(spec: dict[str, Any]) -> tuple[Any, dict[int, int]]:
    """
    Build the synthetic BA network.

    Args:
        spec: The `network` block from the frontmatter, e.g.
              {kind: barabasi_albert, n: 200, m: 2, seed: 4}, or
              {kind: csv, path: data/petster.csv} for an edge list.

    Returns:
        (A, index_of) — sparse adjacency matrix (CSR, float), and a mapping from
        the network's own node labels to matrix indices. 

    Raises:
        ValueError: on an unknown `kind`.
    """
    import networkx as nx

    kind = spec.get("kind", "barabasi_albert")

    if kind == "barabasi_albert":
        G = nx.barabasi_albert_graph(
            int(spec.get("n", 200)),
            int(spec.get("m", 2)),
            seed=int(spec.get("seed", 4)),
        )
    elif kind in ("csv", "edge_csv", "file", "edgelist", "konect"):
        path = spec.get("path")
        if not path:
            raise ValueError(f"network kind {kind!r} requires a 'path'")
        return read_edge_list(_resolve_data_path(path))
    else:
        raise ValueError(
            f"unknown network kind {kind!r}; expected one of "
            "barabasi_albert, csv, file, konect"
        )

    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    return A, {i: i for i in range(A.shape[0])}


# --- evaluation ---


def read_edge_list(path: str | Path):
    """
    Read an undirected edge list.

    Handles the formats from the public network repositories:

    - KONECT `out.*` files: tab-separated, with `%` comment lines at the top,
      and sometimes extra weight and timestamp columns after the two node ids.
    - Network Repository `.edges` files: space or comma separated, occasionally
      with a weight column.
    - Matrix Market `.mtx`: `%%` header, then a dimensions line, then entries.
    - Plain CSV with a `source,target` header row.

    Node ids are relabelled to 0..n-1 in sorted order of the original ids. This
    matters: TASK.md names a source node by its id in the file, and without a
    defined relabelling that id would resolve differently depending on file
    ordering. The mapping is returned so the caller can translate it.

    Args:
        path: Path to the edge-list file.

    Returns:
        (A, index_of) — sparse adjacency matrix (CSR, float), and a dict mapping
        the file's node ids to matrix row indices.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if no edges could be parsed. The first few unparsed lines
            are quoted in the message.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"edge list not found: {path}")

    edges: list[tuple[int, int]] = []
    skipped: list[str] = []
    mtx_dims_seen = False
    is_mtx = path.suffix.lower() == ".mtx"

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("%", "#")):
                continue
            # CSV header row.
            if line.replace(" ", "").lower().startswith("source,target"):
                continue
            # Matrix Market: the first non-comment line is "rows cols nnz".
            if is_mtx and not mtx_dims_seen:
                mtx_dims_seen = True
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                if len(skipped) < 5:
                    skipped.append(line)
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                if len(skipped) < 5:
                    skipped.append(line)
                continue
            if u != v:                      # drop self-loops
                edges.append((u, v))

    if not edges:
        detail = "; ".join(repr(x) for x in skipped) or "file was empty"
        raise ValueError(f"no edges parsed from {path}. First unparsed lines: {detail}")

    labels = sorted({n for e in edges for n in e})
    index_of = {label: i for i, label in enumerate(labels)}

    # Symmetrise and deduplicate in one step.
    pairs = set()
    for u, v in edges:
        a, b = index_of[u], index_of[v]
        pairs.add((a, b))
        pairs.add((b, a))

    rows = np.fromiter((q[0] for q in pairs), dtype=int, count=len(pairs))
    cols = np.fromiter((q[1] for q in pairs), dtype=int, count=len(pairs))
    A = sp.csr_matrix(
        (np.ones(len(pairs)), (rows, cols)), shape=(len(labels), len(labels))
    )
    return A, index_of


def evaluate_scenario(
    task_meta: dict[str, Any],
    delta_t: float,
    m: int,
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    """
    Reproduce Fig. 7 at one scenario: outbreak size over time for each strategy.

    All three strategies are scored on the same time grid from the same base
    seed, so the results are comparable. Evaluating them independently would
    introduce differences in the random draws.

    Args:
        task_meta:      Parsed TASK.md frontmatter. Supplies beta, gamma,
                        source, t0, the network, and the scorer settings.
        delta_t:        Delay before the immunization takes effect.
        m:              Dose budget. A float below 1 is a fraction of the network;
                        an integer is a count. The float 1.0 raises.

    The strategies evaluated are those named in TASK.md's `strategies` field,
    defaulting to all three.
        n_samples:      Ensemble size.
        seed:           Random seed.

    Returns:
        dict with keys:
            times      the time grid, including T_0 and T_0 + delta_t exactly
            curves     {strategy: [expected infected up to each t]}
            scores     {strategy: final outbreak size}
            winner     lowest final value (fewer infected is better), excluding
                       "none", which is the unmitigated baseline rather than a
                       strategy
            runner_up  second lowest
            margin     runner_up minus winner
            pre_effect_max_spread
                       largest disagreement between any two strategies at
                       t <= t0 + delta_t. Should be ~0: no immunity has acted
                       yet, so they must agree. The Observer checks it.
            params, n_samples, seed

    """
    sc = task_meta.get("scenario", {})
    iv = _import_interventions()

    A, index_of = build_network(task_meta.get("network", {}))
    rng = np.random.default_rng(seed)

    # TASK.md names the source by the network's own node id; translate it to a
    # matrix index.
    source_label = int(sc.get("source", 1))
    if source_label not in index_of:
        raise ValueError(
            f"source node {source_label} is not in the network "
            f"(it has {A.shape[0]} nodes)"
        )
    source = index_of[source_label]

    # The paper gives m as a fraction of the network (0.2). A float below 1 is
    # read as a fraction, anything else as a dose count. The type is checked
    # rather than the value, so `m: 1` in TASK.md is one dose while `m: 1.0`
    # would otherwise be indistinguishable from it.
    if isinstance(m, float) and m < 1.0:
        m_doses = int(round(m * A.shape[0]))
    elif isinstance(m, float) and m == 1.0:
        raise ValueError(
            "m=1.0 is ambiguous: a float below 1 means a fraction of the "
            "network, an integer means a dose count. Write `m: 1` for one dose "
            "or `m: 0.9999` for the whole network."
        )
    else:
        m_doses = int(m)

    beta = float(sc.get("beta", 0.5))
    gamma = float(sc.get("gamma", 0.2))
    t0 = float(sc.get("t0", 1.0))
    mapping = str(sc.get("mapping", "exact"))
    selection = str(sc.get("selection", "proportional"))

    # T_0 and T_0 + delta_t are included exactly: the figure marks them, and the
    # pre-effect check evaluates at the latter.
    t_max = float(sc.get("t_max", 40.0))
    t_step = float(sc.get("t_step", 1.0))
    times = np.unique(
        np.concatenate([np.arange(0.0, t_max + t_step, t_step), [t0, t0 + delta_t]])
    )

    # Keyword arguments: the three signatures differ in their optional
    # parameters (random takes `mapping`, hubs `selection`, temporal both).
    all_strategies = {
        "random": iv.strategy_random(
            A, beta, gamma, source, t0, m_doses, int(n_samples),
            rng=rng, mapping=mapping,
        ),
        "hubs": iv.strategy_hubs(
            A, source, m_doses, rng=rng, selection=selection,
        ),
        "temporal": iv.strategy_temporal(
            A, beta, gamma, source, t0, float(delta_t), m_doses, int(n_samples),
            rng=rng, mapping=mapping, selection=selection,
        ),
    }

    # TASK.md may name a subset of the strategies. "none" immunizes nobody and
    # is always computed: it is Fig. 7's unmitigated baseline, not a candidate.
  
    all_strategies["none"] = np.array([], dtype=int)
    wanted = task_meta.get("strategies") or list(all_strategies)
    if isinstance(wanted, str):
        wanted = [wanted]
    if "none" not in wanted:
        wanted = ["none", *wanted]
    unknown = [w for w in wanted if w not in all_strategies]
    if unknown:
        raise ValueError(
            f"TASK.md names unknown strategies {unknown}; "
            f"available: {sorted(all_strategies)}"
        )
    immunized = {k: all_strategies[k] for k in wanted}

    # One time series per strategy, all from the same base seed.
    curves = {
        name: iv.outbreak_curve(A, beta, gamma, source, v, t0, float(delta_t),
                                times, int(n_samples),
                                np.random.default_rng(seed), mapping).tolist()
        for name, v in immunized.items()
    }

    # Before immunity takes effect the strategies should agree.
    pre = times <= t0 + float(delta_t)
    names = list(curves)
    pre_spread = 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.asarray(curves[names[i]])[pre]
            b = np.asarray(curves[names[j]])[pre]
            pre_spread = max(pre_spread, float(np.abs(a - b).max()))

    scores = {name: float(c[-1]) for name, c in curves.items()}
    # "none" is the baseline, not a candidate: it cannot win.
    ranked = sorted(
        ((k, v) for k, v in scores.items() if k != "none"), key=lambda kv: kv[1]
    )

    return {
        "times": times.tolist(),
        "curves": curves,
        "pre_effect_max_spread": pre_spread,
        "scores": scores,
        "winner": ranked[0][0],
        "runner_up": ranked[1][0],
        "margin": ranked[1][1] - ranked[0][1],
        "params": {"delta_t": float(delta_t), "m": m_doses},
        "n_samples": int(n_samples),
        "seed": int(seed),
    }


def _resolve_data_path(path: str | Path) -> Path:
    """
    Locate a data file named in TASK.md.

    Tries the path as given, then relative to the run root, then in `data/`
    under the run root. This allows a task to name a data file without an
    absolute path.

    Args:
        path: The `path` value from the network block.

    Returns:
        The resolved path.

    Raises:
        FileNotFoundError: the message lists every location tried.
    """
    run_root = Path(__file__).resolve().parents[1]
    candidates = [Path(path), run_root / path, run_root / "data" / Path(path).name]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"network data file {path!r} not found. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _simulation_path() -> Path:
    """
    Locate the bundled SpreadingMapping directory.

    `launch.py` copies the simulation into every run, so it always sits beside this
    file's parent.
    """
    return Path(__file__).resolve().parents[1] / "SpreadingMapping-py"


def _import_interventions():
    """
    Import SpreadingMapping's interventions module.

    Imported here rather than at module level so the error message can name the
    location that was tried.
    """
    p = _simulation_path()
    if not (p / "interventions.py").exists():
        raise ImportError(
            f"SpreadingMapping's interventions.py not found at {p}. launch.py "
            "copies the simulation into each run; if it is missing, the run was not "
            "built by launch.py."
        )
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    import interventions  # type: ignore
    return interventions


# --- command line ---


def main() -> None:
    """Evaluate one scenario from the command line."""
    ap = argparse.ArgumentParser(description="Evaluate one immunization scenario.")
    ap.add_argument("--task", default="task/TASK.md", help="path to TASK.md")
    ap.add_argument("--delta-t", type=float, required=True)
    ap.add_argument("--m", type=float, required=True,
                    help="dose count, or a fraction of the network if below 1")
    ap.add_argument("--n-samples", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="raw output instead of a table")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    task_path = Path(args.task)
    text = task_path.read_text(encoding="utf-8")
    from system.helper import _parse_frontmatter

    meta = _parse_frontmatter(text.split("---", 2)[1]) if text.startswith("---") else {}

    result = evaluate_scenario(
        meta,
        delta_t=args.delta_t,
        m=args.m,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # Coarse grid, including the two times in Fig. 7.
    sc = meta.get("scenario", {})
    t0 = float(sc.get("t0", 1.0))
    t_eff = t0 + args.delta_t
    times = result["times"]
    curves = result["curves"]
    names = list(curves)

    print(f"m = {result['params']['m']} doses, n_samples = {result['n_samples']}, "
          f"seed = {result['seed']}")
    print()
    header = f"{'t':>6}" + "".join(f"{n:>10}" for n in names)
    print(header)
    print("-" * len(header))
    for i, t in enumerate(times):
        show = t in (t0, t_eff) or i == len(times) - 1 or t % 5 == 0
        if not show:
            continue
        row = f"{t:6.0f}" + "".join(f"{curves[n][i]:10.1f}" for n in names)
        if t == t0:
            row += "   <- t0, immunization decided"
        elif t == t_eff:
            row += "   <- immunity effective"
        print(row)

    print()
    print(f"ordering: " + " < ".join(
        f"{k} ({v:.1f})" for k, v in sorted(result["scores"].items(),
                                            key=lambda kv: kv[1])))
    spread = result["pre_effect_max_spread"]
    print(f"pre-effect max spread (t <= {t_eff:.0f}): {spread:.6f}"
          f"  {'OK' if spread < 1e-6 else 'CHECK THIS'}")


if __name__ == "__main__":
    main()
