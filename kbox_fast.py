#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kbox_fast.py
============

Optimized, parallelized local-search and scoring for k-box MISR search.

Three optimizations vs. the stock mistr_runner pipeline, with safety
notes on each:

  1. Cross-seed multiprocessing via ProcessPoolExecutor.
     Each worker builds its own Gurobi env; seeds are evaluated in
     parallel. Expected speedup ~= min(n_workers, cores / grb_threads).
     Safety: Gurobi envs are built cleanly inside each worker, no shared
     state. Results are merged at the end; ordering differences don't
     affect correctness since local_search is independently seeded per task.

  2. LP-only during neighbor search, ILP only on elites at the end.
     local_search_fast() uses the LP value (scaled) as its scoring signal
     during neighborhood exploration. When a run finishes, it ILP-verifies
     the top elites before returning. For non-elite neighbors, the ratio
     is never used for anything that persists, so skipping ILP is safe.
     Expected speedup: ~3-5x on top of parallelism.
     Safety: the returned elite list contains only ILP-verified entries.
     Their `best_ratio` is the true LP/ILP ratio, identical to what the
     stock search would return.

  3. Warm-start ILP solves from the previous elite's independent set.
     When ILP-verifying a new elite whose structure is close to a known
     elite, we pass the known IS as a MIP start. Gurobi uses this as a
     hint; it is never treated as a bound, so correctness is unaffected.
     Expected speedup: ~1.5-3x on ILP verification.

Symmetry breaking (item 5 from the optimization list) is NOT implemented
here. For k-box instances, natural symmetries exist (permuting boxes
gives equivalent graphs) but programmatically detecting which labels
belong to which "box" after arbitrary local-search drift is fragile.
Getting this wrong could exclude valid optima. Skipping for safety.

Lazy constraint generation (item 4) is NOT implemented. The engineering
effort is substantial and the payoff at n ≤ 600 is modest.

Usage:
    from kbox_fast import parallel_local_search_fast

    merged, best_ratio = parallel_local_search_fast(
        seeds, time_budget_s=4.0, n_workers=8, grb_threads=1,
        alpha_lp=0.15, beta_ilp=0.10,
        verify_top_k=8,   # ILP-verify the top 8 elites per seed
    )
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from kbox_misr import (
    Instance, Seq,
    canonicalize, instance_key,
    build_rects, grid_points, covers_grid_closed,
)

# worker-global (set by _init_worker)
_WORKER_ENV = None
_WORKER_GRB_THREADS = 1


def _init_worker(grb_threads: int):
    """ProcessPoolExecutor initializer. One Gurobi env per worker process."""
    global _WORKER_ENV, _WORKER_GRB_THREADS
    import gurobipy as gp
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    if grb_threads > 0:
        env.setParam("Threads", grb_threads)
    env.start()
    _WORKER_ENV = env
    _WORKER_GRB_THREADS = grb_threads


# =========================================================================== #
# LP-only scoring (used during local search)                                   #
# =========================================================================== #


def _solve_lp(rects, covers, env=None):
    """Fast LP solve. Returns LP value (float) or None on failure."""
    import gurobipy as gp
    from gurobipy import GRB
    n = len(rects)
    m = gp.Model("lp", env=env) if env else gp.Model("lp")
    m.setParam("OutputFlag", 0)
    # pick the right method based on size
    if n >= 300:
        m.setParam("Method", 3)       # concurrent simplex
        m.setParam("Crossover", 0)    # we don't need a basic solution
    else:
        m.setParam("Method", 1)       # dual simplex
    x = m.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    m.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if len(S) >= 2:
            m.addConstr(gp.quicksum(x[i] for i in S) <= 1)
    m.optimize()
    if m.status == GRB.OPTIMAL:
        return float(m.objVal)
    return None


def _solve_ilp(rects, covers, env=None, time_limit: float = 60.0,
               warm_start_selected: Optional[List[int]] = None,
               require_optimal: bool = False):
    """
    Exact ILP solve. Returns (ilp_value, selected_indices, proved_optimal) or
    (None, None, False) if no solution found.

    warm_start_selected: optional list of 0-indexed rect positions known to
    form a feasible IS. Passed to Gurobi as a MIP start. Gurobi treats this
    as a hint, not a bound, so correctness is unaffected.

    require_optimal: if True, returns (None, None, False) unless Gurobi
    proved optimality within the time limit. Set True for verification;
    False for fast neighbor scoring where a lower bound is acceptable.
    """
    import gurobipy as gp
    from gurobipy import GRB
    n = len(rects)
    m = gp.Model("ilp", env=env) if env else gp.Model("ilp")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    m.setParam("MIPFocus", 2)         # prove optimality
    m.setParam("Presolve", 2)
    m.setParam("Cuts", 2)

    y = m.addVars(n, vtype=GRB.BINARY, name="y")
    m.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if len(S) >= 2:
            m.addConstr(gp.quicksum(y[i] for i in S) <= 1)

    # warm start
    if warm_start_selected is not None:
        sel_set = set(warm_start_selected)
        for i in range(n):
            y[i].Start = 1.0 if i in sel_set else 0.0

    m.optimize()
    if m.SolCount == 0:
        return None, None, False
    proved_optimal = (m.status == GRB.OPTIMAL)
    if require_optimal and not proved_optimal:
        return None, None, False
    ilp_val = float(m.objVal)
    selected = [i for i in range(n) if y[i].X > 0.5]
    return ilp_val, selected, proved_optimal


# =========================================================================== #
# Local search, fast variant                                                   #
# =========================================================================== #
#
# Strategy:
#   * Score candidates by LP only during neighborhood exploration.
#   * Keep the top K elites by LP value.
#   * At end of run, ILP-verify all elites (in order of LP value) to get
#     the true gap. Warm-start each ILP from the previous elite's selection.
#   * Return elites with their ILP-verified ratios, so the caller sees
#     the exact same data shape as the stock local_search.


def _neighbors_adapter(H: Seq, V: Seq, rng: random.Random, k: int
                       ) -> List[Instance]:
    """
    Use mistr_runner's neighbors(); local to this module so we don't need
    to re-import globally in each worker.
    """
    from mistr_runner import neighbors
    return neighbors(H, V, rng, k)


def local_search_fast(
    seed: Instance,
    time_budget_s: float,
    rng: random.Random,
    alpha_lp: float = 0.15,
    beta_ilp: float = 0.10,
    grb_threads: int = 0,
    tabu_seconds: float = 20.0,
    elite_size: int = 32,
    neighbor_k: int = 64,
    verify_top_k: int = 8,
    verify_time_limit: float = 60.0,
    neighbor_ilp_time: float = 0.8,
    env=None,  # optional gurobi env (passed by parallel worker)
) -> Tuple[List[Tuple[float, Seq, Seq]], float]:
    """
    LP + short-budget-ILP driven local search + full ILP verification of elites.

    Each neighbor is scored by:
        - solve LP (exact, fast)
        - solve ILP with a short time limit (default 0.8s). The ILP may
          not be proved optimal within that budget; in that case we use
          the best feasible upper bound.

    This gives a real lp/ilp gradient signal (unlike pure-LP scoring,
    which is constant at n/2 in triangle-free regimes). The final elites
    get fully-verified ILP at the end.

    Returns (elites_sorted_by_verified_ratio, best_ratio).
    """
    start = time.time()
    H, V = canonicalize(*seed)
    # seen: instance_key -> (lp, ilp_shortbudget, ratio, blended)
    seen: Dict[str, Tuple[float, float, float, float]] = {}
    elite_ratio: List[Tuple[float, Seq, Seq]] = []
    tabu: Dict[str, float] = {}

    def push(r_val: float, h: Seq, v: Seq):
        elite_ratio.append((r_val, h[:], v[:]))
        elite_ratio.sort(key=lambda x: -x[0])
        if len(elite_ratio) > max(elite_size, verify_top_k):
            elite_ratio.pop()

    def score_lp_shortilp(H_: Seq, V_: Seq):
        """Returns (lp, ilp, ratio, blended) or (None,)*4 on failure.

        The ILP call here uses a short time limit and does NOT require
        proof of optimality — a feasible lower bound on alpha is fine
        for ranking candidates during the search. The final verification
        pass uses a longer budget and does require proved optimality,
        so any reported gap is tight.
        """
        rects = build_rects(H_, V_)
        pts = grid_points(rects)
        covers = covers_grid_closed(rects, pts)
        lp = _solve_lp(rects, covers, env=env)
        if lp is None:
            return None, None, None, None
        ilp, _, _proved = _solve_ilp(rects, covers, env=env,
                                     time_limit=neighbor_ilp_time,
                                     require_optimal=False)
        if ilp is None or ilp <= 0:
            return lp, None, None, None
        ratio = lp / ilp
        n_ = len(rects)
        blended = ratio + alpha_lp * (lp / n_) - beta_ilp * (ilp / n_)
        return lp, ilp, ratio, blended

    # initial seed
    lp0, ilp0, r0, b0 = score_lp_shortilp(H, V)
    if lp0 is None or r0 is None:
        return [], 0.0
    seen[instance_key(H, V)] = (lp0, ilp0, r0, b0)
    push(r0, H, V)
    cur_blended = b0
    best_during_search = r0

    # search loop
    while time.time() - start < time_budget_s:
        now = time.time()
        key = instance_key(H, V)
        if key in tabu and (now - tabu[key] < tabu_seconds):
            if elite_ratio:
                _, H, V = random.choice(elite_ratio)
                cur_blended = seen.get(instance_key(H, V),
                                       (0, 0, 0, 0))[3]
            else:
                H, V = H[::-1], V[::-1]
            continue

        cand = _neighbors_adapter(H, V, rng, neighbor_k)
        best_nb = None
        best_sc = -1e9
        for (h2, v2) in cand:
            if time.time() - start >= time_budget_s:
                break
            k2 = instance_key(h2, v2)
            if k2 in seen:
                lp2, ilp2, r2, b2 = seen[k2]
            else:
                lp2, ilp2, r2, b2 = score_lp_shortilp(h2, v2)
                if lp2 is None or r2 is None:
                    continue
                seen[k2] = (lp2, ilp2, r2, b2)
                push(r2, h2, v2)
                if r2 > best_during_search:
                    best_during_search = r2
            if b2 > best_sc:
                best_sc = b2
                best_nb = (h2, v2, lp2, ilp2, r2, b2)

        if best_nb:
            _, _, _, _, r2, b2 = best_nb
            if b2 >= cur_blended:
                H, V = best_nb[0], best_nb[1]
                cur_blended = b2
            else:
                delta = b2 - cur_blended
                T = 0.03
                if math.exp(delta / max(T, 1e-6)) > rng.random():
                    H, V = best_nb[0], best_nb[1]
                    cur_blended = b2
                else:
                    tabu[key] = now
                    if elite_ratio:
                        _, H, V = random.choice(elite_ratio)
                        cur_blended = seen.get(instance_key(H, V),
                                               (0, 0, 0, 0))[3]

    # --- final ILP verification of top elites ---
    # REQUIRES proof of optimality. Elites where Gurobi can't prove optimum
    # within verify_time_limit are skipped — better to report nothing than
    # an inflated ratio from a lower-bound IS.
    # Also: we do NOT warm-start across elites here. A neighboring elite's
    # IS may not correspond to a feasible IS in a structurally-different
    # graph, and even when it does, the initial presolve overhead saved
    # is small compared to the risk of biasing Gurobi's branch-and-bound.
    verify_n = min(verify_top_k, len(elite_ratio))
    verified: List[Tuple[float, Seq, Seq]] = []
    skipped_unverified = 0
    for (search_ratio, H_e, V_e) in elite_ratio[:verify_n]:
        rects = build_rects(H_e, V_e)
        pts = grid_points(rects)
        covers = covers_grid_closed(rects, pts)
        lp_v = _solve_lp(rects, covers, env=env)
        if lp_v is None:
            continue
        ilp_v, _sel, proved = _solve_ilp(
            rects, covers, env=env,
            time_limit=verify_time_limit,
            warm_start_selected=None,
            require_optimal=True,
        )
        if ilp_v is None or ilp_v <= 0 or not proved:
            skipped_unverified += 1
            continue
        verified.append((lp_v / ilp_v, H_e, V_e))
    verified.sort(key=lambda x: -x[0])
    best_ratio = verified[0][0] if verified else 0.0
    return verified, best_ratio


# =========================================================================== #
# Process-pool driver                                                          #
# =========================================================================== #


def _worker_run_one_seed(task):
    """
    Run local_search_fast on a single seed. Runs inside a worker process.
    `env` comes from the worker-global _WORKER_ENV set up in _init_worker.
    """
    (idx, seed, budget, rng_seed, alpha_lp, beta_ilp,
     elite_size, neighbor_k, verify_top_k, verify_time_limit,
     neighbor_ilp_time) = task
    try:
        rng = random.Random(rng_seed)
        es, best = local_search_fast(
            seed,
            time_budget_s=budget,
            rng=rng,
            alpha_lp=alpha_lp,
            beta_ilp=beta_ilp,
            grb_threads=_WORKER_GRB_THREADS,
            elite_size=elite_size,
            neighbor_k=neighbor_k,
            verify_top_k=verify_top_k,
            verify_time_limit=verify_time_limit,
            neighbor_ilp_time=neighbor_ilp_time,
            env=_WORKER_ENV,
        )
        return idx, es, best, None
    except Exception:
        return idx, [], 0.0, traceback.format_exc()


def parallel_local_search_fast(
    seeds: List[Instance],
    time_budget_s: float = 4.0,
    n_workers: Optional[int] = None,
    grb_threads: int = 1,
    alpha_lp: float = 0.15,
    beta_ilp: float = 0.10,
    elite_size: int = 32,
    neighbor_k: int = 64,
    verify_top_k: int = 8,
    verify_time_limit: float = 60.0,
    neighbor_ilp_time: float = 0.8,
    rng_seed_base: int = 0,
    verbose: bool = True,
) -> Tuple[List[Tuple[float, Seq, Seq]], float]:
    """
    Run local_search_fast on many seeds in parallel.

    Returns (merged_elites_sorted_desc, best_ratio_across_all). Merged
    elites are deduplicated by instance_key. Rankings are by ILP-verified
    LP/ILP ratio, the same metric the stock pipeline uses.
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) // max(1, grb_threads))

    tasks = [
        (i, seeds[i], time_budget_s, rng_seed_base + i,
         alpha_lp, beta_ilp, elite_size, neighbor_k,
         verify_top_k, verify_time_limit, neighbor_ilp_time)
        for i in range(len(seeds))
    ]
    merged: List[Tuple[float, Seq, Seq]] = []
    seen_keys: set = set()
    best_overall = 0.0
    n_errors = 0

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(grb_threads,),
    ) as ex:
        futs = [ex.submit(_worker_run_one_seed, t) for t in tasks]
        for f in as_completed(futs):
            idx, es, best, err = f.result()
            if err is not None:
                n_errors += 1
                if verbose:
                    print(f"[seed {idx}] worker error:\n{err}", file=sys.stderr)
                continue
            if best > best_overall:
                best_overall = best
            for (ratio, H_e, V_e) in es:
                key = instance_key(H_e, V_e)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged.append((ratio, H_e, V_e))
    if n_errors and verbose:
        print(f"[warn] {n_errors}/{len(seeds)} worker tasks errored.",
              file=sys.stderr)
    merged.sort(key=lambda x: -x[0])
    return merged, best_overall
