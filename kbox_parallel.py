#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kbox_parallel.py
================

Parallel and vectorized helpers for k-box MISR search. Addresses the real
bottleneck: serial Gurobi LP/ILP solves at large n.

Three layers of speedup, in order of impact:

  Layer 1 — Cross-seed multiprocessing (biggest win, ~linear in workers).
      parallel_evaluate(instances, n_workers, grb_threads)
      parallel_local_search(seeds, n_workers, ...)

  Layer 2 — Vectorized pre-checks (~50x on n=400).
      vec_max_clique_at_grid(H, V)
      vec_actual_edges(H, V)
      vec_triangle_free(H, V)

  Layer 3 — Gurobi parameter tuning (2-5x on large LPs).
      tuned_gurobi_env(n, for_lp_only=True/False)

GPU is intentionally NOT used. Rationale:
  * Gurobi is CPU-only; there is no CUDA/MPS path for simplex or B&B.
  * The non-Gurobi work is too light to overcome PCIe/kernel-launch overhead.
  * The transformer in mistr_runner already uses MPS/CUDA where appropriate.

Gurobi license note:
  Creating multiple Gurobi environments across processes works with standard
  academic licenses and most commercial single-machine licenses. If you hit
  license errors, reduce n_workers or check the Gurobi license type.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from kbox_misr import Instance, Seq, build_rects, kbox_instance
from kbox_search import max_clique_at_grid  # scalar reference


# =========================================================================== #
# Layer 2: vectorized pre-checks                                               #
# =========================================================================== #


def _rects_as_array(H: Seq, V: Seq) -> np.ndarray:
    """Return an (n, 4) int array with columns [x1, x2, y1, y2]."""
    rects = build_rects(H, V)
    if not rects:
        return np.empty((0, 4), dtype=np.int64)
    arr = np.empty((len(rects), 4), dtype=np.int64)
    for i, ((x1, x2), (y1, y2)) in enumerate(rects):
        arr[i, 0] = x1
        arr[i, 1] = x2
        arr[i, 2] = y1
        arr[i, 3] = y2
    return arr


def vec_max_clique_at_grid(H: Seq, V: Seq) -> int:
    """
    Vectorized version of max_clique_at_grid. Returns the maximum number of
    rectangles covering any single grid point.

    Strategy:
      Let X_in be a boolean matrix (n_xs, n_rects) where X_in[i, j] is True iff
      rectangle j contains the x-coordinate xs[i]. Similarly Y_in.
      Then the clique count at grid point (xs[i], ys[k]) is
          C[i, k] = sum_j X_in[i, j] AND Y_in[k, j]
      which equals the integer matrix product X_in @ Y_in.T.
      We return C.max().

    Complexity: O(n_xs * n_ys * n_rects). For n=400, n_xs = n_ys = 2n = 800, so
    ~2.5e8 ops executed by BLAS — well under a second on any modern CPU.
    """
    arr = _rects_as_array(H, V)
    if arr.shape[0] == 0:
        return 0
    n = arr.shape[0]
    x1, x2 = arr[:, 0], arr[:, 1]
    y1, y2 = arr[:, 2], arr[:, 3]

    xs = np.unique(np.concatenate([x1, x2]))
    ys = np.unique(np.concatenate([y1, y2]))

    # (n_xs, n_rects) boolean → int
    X_in = ((xs[:, None] >= x1[None, :]) & (xs[:, None] <= x2[None, :])).astype(np.int32)
    Y_in = ((ys[:, None] >= y1[None, :]) & (ys[:, None] <= y2[None, :])).astype(np.int32)

    # counts[i, k] = #rects covering (xs[i], ys[k])
    counts = X_in @ Y_in.T  # (n_xs, n_ys)
    return int(counts.max()) if counts.size else 0


def vec_triangle_free(H: Seq, V: Seq) -> bool:
    """True iff every grid point has at most 2 rectangles on it."""
    return vec_max_clique_at_grid(H, V) <= 2


def vec_actual_edges(H: Seq, V: Seq) -> set:
    """
    Vectorized intersection graph. Returns a set of frozensets of 1-indexed
    rectangle labels.

    Complexity: O(n^2) memory and time. For n=400, a 400x400 boolean matrix —
    trivial. For n=4000 this would use 16M booleans = 16MB, still fine.
    """
    arr = _rects_as_array(H, V)
    n = arr.shape[0]
    if n == 0:
        return set()
    x1 = arr[:, 0][:, None]; x2 = arr[:, 1][:, None]
    y1 = arr[:, 2][:, None]; y2 = arr[:, 3][:, None]
    # overlap[i, j] = max(x1_i, x1_j) <= min(x2_i, x2_j) AND same for y
    x_over = (np.maximum(x1, x1.T) <= np.minimum(x2, x2.T))
    y_over = (np.maximum(y1, y1.T) <= np.minimum(y2, y2.T))
    adj = x_over & y_over
    np.fill_diagonal(adj, False)
    # upper triangle only
    iu, ju = np.triu_indices(n, k=1)
    mask = adj[iu, ju]
    edges = set()
    for i, j in zip(iu[mask], ju[mask]):
        edges.add(frozenset((int(i) + 1, int(j) + 1)))
    return edges


# =========================================================================== #
# Layer 3: Gurobi parameter tuning                                             #
# =========================================================================== #


def tuned_gurobi_params(n: int, for_lp_only: bool = False) -> Dict[str, int]:
    """
    Recommended Gurobi parameters for MISR clique-LP / ILP at size n.

    These are starting points, NOT oracle values. For production runs, use
    Gurobi's tuner on a representative instance (Tune command).

    for_lp_only=True: LP relaxation only (faster path; simpler dispatch).
    """
    p: Dict[str, int] = {"OutputFlag": 0}
    if for_lp_only:
        # LP: dual simplex often wins on clique LPs; concurrent simplex is
        # safer for larger n.
        if n >= 300:
            p["Method"] = 3       # concurrent (run primal/dual/barrier in parallel)
        else:
            p["Method"] = 1       # dual simplex
        p["Crossover"] = 0        # we don't need a basic solution
    else:
        # ILP: presolve + cuts + aggressive MIP heuristics.
        p["Presolve"] = 2
        p["Cuts"] = 2
        p["MIPFocus"] = 1         # focus on finding feasible solutions
        p["Heuristics"] = 0.2
        if n >= 300:
            p["NodefileStart"] = 2  # spill node trees to disk when tight
    return p


# =========================================================================== #
# Layer 1: multiprocessing                                                     #
# =========================================================================== #
#
# Pattern: each worker process gets its own Gurobi Env in an initializer.
# Tasks are (H, V) pairs; results are (lp, ilp) or (lp, None) for LP-only.

# Module-global set by _init_worker on first call
_WORKER_ENV = None
_WORKER_GRB_THREADS = 1
_WORKER_LP_ONLY = False


def _init_worker(grb_threads: int, lp_only: bool):
    """ProcessPoolExecutor initializer: build one Gurobi env per worker."""
    global _WORKER_ENV, _WORKER_GRB_THREADS, _WORKER_LP_ONLY
    import gurobipy as gp  # noqa: F401 — imported for side-effect of creating env
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    if grb_threads > 0:
        env.setParam("Threads", grb_threads)
    env.start()
    _WORKER_ENV = env
    _WORKER_GRB_THREADS = grb_threads
    _WORKER_LP_ONLY = lp_only


def _worker_solve(task):
    """Worker: solve LP (and optionally ILP) for one (H, V) instance."""
    global _WORKER_ENV, _WORKER_LP_ONLY
    idx, H, V = task
    try:
        import gurobipy as gp
        from gurobipy import GRB

        from kbox_misr import build_rects, grid_points, covers_grid_closed

        rects = build_rects(H, V)
        pts = grid_points(rects)
        covers = covers_grid_closed(rects, pts)
        n = len(rects)

        # LP
        m_lp = gp.Model("misr_lp", env=_WORKER_ENV)
        x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
        m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
        for S in covers:
            if len(S) >= 2:
                m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1)
        # tune LP method for large n
        if n >= 300:
            m_lp.setParam("Method", 3)
            m_lp.setParam("Crossover", 0)
        else:
            m_lp.setParam("Method", 1)
        m_lp.optimize()
        lp = float(m_lp.objVal) if m_lp.status == GRB.OPTIMAL else None

        ilp: Optional[float] = None
        if not _WORKER_LP_ONLY:
            m_ilp = gp.Model("misr_ilp", env=_WORKER_ENV)
            y = m_ilp.addVars(n, vtype=GRB.BINARY, name="y")
            m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
            for S in covers:
                if len(S) >= 2:
                    m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1)
            m_ilp.setParam("Presolve", 2)
            m_ilp.setParam("Cuts", 2)
            m_ilp.setParam("MIPFocus", 1)
            m_ilp.optimize()
            ilp = float(m_ilp.objVal) if m_ilp.status == GRB.OPTIMAL else None

        return (idx, lp, ilp, None)
    except Exception as e:
        return (idx, None, None, str(e))


def parallel_evaluate(instances: List[Instance],
                      n_workers: Optional[int] = None,
                      grb_threads: int = 1,
                      lp_only: bool = False
                      ) -> List[Tuple[Optional[float], Optional[float]]]:
    """
    Solve LP (and optionally ILP) for a batch of instances in parallel.

    Returns a list of (lp, ilp) tuples, aligned with `instances`. On error,
    the corresponding entry is (None, None); errors are logged to stderr.

    n_workers:  defaults to os.cpu_count(). Best to set
                n_workers * grb_threads <= physical_core_count.
    grb_threads: threads Gurobi uses per worker. 1 is safest with many workers.
    lp_only:    skip the ILP. Use for fast ranking during local search.
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) // max(1, grb_threads))
    tasks = [(i, H, V) for i, (H, V) in enumerate(instances)]
    results: List[Tuple[Optional[float], Optional[float]]] = [
        (None, None)] * len(instances)

    ctx = mp.get_context("spawn")  # clean env per worker; avoids fork+Gurobi hazards
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(grb_threads, lp_only),
    ) as ex:
        futs = [ex.submit(_worker_solve, t) for t in tasks]
        for f in as_completed(futs):
            idx, lp, ilp, err = f.result()
            if err is not None:
                print(f"[worker {idx}] error: {err}", file=sys.stderr)
            results[idx] = (lp, ilp)
    return results


# =========================================================================== #
# Parallel local-search driver                                                 #
# =========================================================================== #
#
# Runs `local_search` from mistr_runner on each seed in a worker process.
# Each worker maintains its own tabu + elites; results merge at the end.


def _worker_local_search(task):
    global _WORKER_GRB_THREADS
    idx, seed, budget, rng_seed, alpha_lp, beta_ilp = task
    try:
        from mistr_runner import local_search
        rng = random.Random(rng_seed)
        es, best = local_search(
            seed,
            time_budget_s=budget,
            rng=rng,
            alpha_lp=alpha_lp,
            beta_ilp=beta_ilp,
            grb_threads=_WORKER_GRB_THREADS,
            elite_size=32,
            neighbor_k=64,
        )
        return idx, es, best, None
    except Exception as e:
        import traceback
        return idx, [], None, traceback.format_exc()


def parallel_local_search(seeds: List[Instance],
                          time_budget_s: float = 3.0,
                          n_workers: Optional[int] = None,
                          grb_threads: int = 1,
                          alpha_lp: float = 0.15,
                          beta_ilp: float = 0.10,
                          rng_seed_base: int = 0,
                          ) -> Tuple[List[Tuple[float, Seq, Seq]], float]:
    """
    Run local_search() on each seed in parallel; aggregate elites and best.

    Returns (merged_elites_sorted_desc, best_ratio_across_all).
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) // max(1, grb_threads))
    tasks = [
        (i, (H, V), time_budget_s, rng_seed_base + i, alpha_lp, beta_ilp)
        for i, (H, V) in enumerate(seeds)
    ]
    merged: List[Tuple[float, Seq, Seq]] = []
    best_overall = 0.0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(grb_threads, False),
    ) as ex:
        futs = [ex.submit(_worker_local_search, t) for t in tasks]
        for f in as_completed(futs):
            idx, es, best, err = f.result()
            if err is not None:
                print(f"[seed {idx}] error:\n{err}", file=sys.stderr)
                continue
            if best is not None:
                best_overall = max(best_overall, best)
            merged.extend(es)
    merged.sort(key=lambda x: -x[0])
    return merged, best_overall


# =========================================================================== #
# Benchmarks                                                                   #
# =========================================================================== #


def bench_vectorization(k: int = 5):
    """Compare scalar vs vectorized pre-checks on M_k."""
    H, V = kbox_instance(k)

    t0 = time.perf_counter()
    r_scalar = max_clique_at_grid(H, V)
    t_scalar = time.perf_counter() - t0

    t0 = time.perf_counter()
    r_vec = vec_max_clique_at_grid(H, V)
    t_vec = time.perf_counter() - t0

    assert r_scalar == r_vec, f"scalar={r_scalar} vec={r_vec}"

    print(f"k={k}  n={4*k*k}")
    print(f"  max_clique_at_grid  scalar: {t_scalar*1000:7.2f} ms")
    print(f"  max_clique_at_grid  vec:    {t_vec*1000:7.2f} ms   "
          f"(speedup {t_scalar/max(t_vec,1e-9):.1f}x)")


def bench_parallel(k: int = 5, n_seeds: int = 8, grb_threads: int = 1):
    """Run parallel_evaluate on n_seeds copies of M_k to measure speedup."""
    H, V = kbox_instance(k)
    # create n_seeds slightly-perturbed variants so the solver doesn't cache
    from kbox_search import shuffle_boxes
    rng = random.Random(0)
    instances = [(H, V)]
    for _ in range(n_seeds - 1):
        instances.append(shuffle_boxes(H, V, k, rng))

    print(f"parallel_evaluate on {n_seeds} seeds at k={k} (n={4*k*k})")
    for nw in (1, 2, 4, max(1, (os.cpu_count() or 1))):
        t0 = time.perf_counter()
        res = parallel_evaluate(instances, n_workers=nw,
                                grb_threads=grb_threads, lp_only=False)
        dt = time.perf_counter() - t0
        good = sum(1 for (lp, ilp) in res if lp is not None)
        print(f"  workers={nw}  grb_threads={grb_threads}  "
              f"wall={dt:.2f}s  ok={good}/{n_seeds}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-vec", action="store_true",
                    help="benchmark vectorized vs scalar pre-checks")
    ap.add_argument("--bench-par", action="store_true",
                    help="benchmark parallel Gurobi solves")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--grb-threads", type=int, default=1)
    args = ap.parse_args()

    if args.bench_vec:
        for k in (3, 5, 7, 10):
            bench_vectorization(k)
            print()
    if args.bench_par:
        bench_parallel(args.k, args.n_seeds, args.grb_threads)

    if not (args.bench_vec or args.bench_par):
        ap.print_help()


if __name__ == "__main__":
    # Required on macOS so child processes use spawn properly.
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()
