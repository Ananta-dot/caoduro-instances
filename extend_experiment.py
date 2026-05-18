#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extend_experiment.py
====================

Directed test of the "extend a thin segment across the grid" move on
pristine M_k. The diagnostic in geom_outliers.py established that the
verified +1/+2 results at k=9 and k=10 come from this move applied to
1–3 specific rectangles. The k=11 plateau result has zero such moves
applied.

This script does long-distance swaps on pristine M_k (rather than the
±1 nudges in perturb_segment_extent) and scores each candidate with
the same Gurobi LP/ILP used by the search. Anything that beats the
Caoduro baseline gets saved.

Two modes:
  --mode random:   pick a random label, swap one of its positions in
                   H or V with a random other position.
  --mode directed: enumerate height-1 (or width-1) thin segments and
                   extend their endpoints toward the bulk of the grid.
                   Mirrors what we observed at k=9 and k=10.

The candidate is the bare modification — no local search wraps around
it. We're testing whether the seed itself is good, not whether local
search around the seed is good.

Usage:
    python3 extend_experiment.py --k 11 --mode random --trials 200
    python3 extend_experiment.py --k 11 --mode directed
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time
from typing import List, Tuple

from kbox_misr import kbox_instance, build_rects, grid_points, covers_grid_closed
from kbox_parallel import vec_triangle_free, vec_max_clique_at_grid

# Use the fast solver configuration from verify_instance.py rather than
# mistr_runner.solve_lp_ilp (which doesn't set MIPFocus=2 / Method=3 and is
# much slower at n ≥ 200).
import gurobipy as gp
from gurobipy import GRB


def fast_lp(rects, covers, threads=0):
    m = gp.Model("clique_lp")
    m.setParam("OutputFlag", 0)
    if threads > 0:
        m.setParam("Threads", threads)
    m.setParam("Method", 3)
    m.setParam("Crossover", 0)
    n = len(rects)
    x = m.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    m.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if len(S) >= 2:
            m.addConstr(gp.quicksum(x[i] for i in S) <= 1)
    m.optimize()
    return float(m.objVal) if m.status == GRB.OPTIMAL else float("nan")


def fast_ilp(rects, covers, time_limit=60.0, threads=0):
    m = gp.Model("ilp")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    m.setParam("MIPFocus", 2)
    m.setParam("Presolve", 2)
    m.setParam("Cuts", 2)
    if threads > 0:
        m.setParam("Threads", threads)
    n = len(rects)
    z = m.addVars(n, vtype=GRB.BINARY, name="z")
    m.setObjective(gp.quicksum(z[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if len(S) >= 2:
            m.addConstr(gp.quicksum(z[i] for i in S) <= 1)
    m.optimize()
    if m.SolCount == 0:
        return None, False
    return float(m.objVal), (m.status == GRB.OPTIMAL)


def solve_lp_ilp(rects, time_limit=60.0, threads=0):
    """Drop-in compatible signature: returns (lp, ilp). Uses fast config."""
    pts = grid_points(rects)
    covers = covers_grid_closed(rects, pts)
    lp = fast_lp(rects, covers, threads=threads)
    ilp_val, _ = fast_ilp(rects, covers, time_limit=time_limit, threads=threads)
    return lp, ilp_val if ilp_val is not None else 0.0

Seq = List[int]


# ---------- mutations ----------

def random_long_swap(H: Seq, V: Seq, rng: random.Random) -> Tuple[Seq, Seq]:
    """Pick a label, swap one of its 2 positions in H or V with any other slot."""
    n = max(max(H), max(V))
    which = rng.choice(("H", "V"))
    seq = list(V if which == "V" else H)
    lab = rng.randint(1, n)
    positions = [p for p, x in enumerate(seq) if x == lab]
    if len(positions) != 2:
        return list(H), list(V)
    p_idx = rng.choice(positions)
    target = rng.randrange(len(seq))
    while seq[target] == lab:
        target = rng.randrange(len(seq))
    seq[p_idx], seq[target] = seq[target], seq[p_idx]
    if which == "V":
        return list(H), seq
    return seq, list(V)


def directed_extend_horizontal(H: Seq, V: Seq, rng: random.Random
                               ) -> Tuple[Seq, Seq]:
    """
    Find a height-1 horizontal rectangle and drag its bottom (or top)
    edge across part of the grid. Models the k=10 modification.
    """
    rects = build_rects(H, V)
    n = len(rects)
    candidates = []  # labels of height-1 horizontal rects
    for i, ((x1, x2), (y1, y2)) in enumerate(rects):
        if y2 - y1 == 1 and x2 - x1 >= 10:
            candidates.append(i + 1)  # 1-based label
    if not candidates:
        return list(H), list(V)
    lab = rng.choice(candidates)

    V2 = list(V)
    positions = [p for p, x in enumerate(V2) if x == lab]
    p_first, p_last = min(positions), max(positions)
    end = rng.choice(("first", "last"))
    if end == "first":
        # drag the bottom edge downward by a random distance
        p_old = p_first
        p_new = rng.randint(0, p_first - 1) if p_first > 0 else p_first
    else:
        # drag the top edge upward
        p_old = p_last
        p_new = rng.randint(p_last + 1, len(V2) - 1) if p_last < len(V2) - 1 else p_last
    if p_new == p_old:
        return list(H), list(V)
    V2[p_old], V2[p_new] = V2[p_new], V2[p_old]
    return list(H), V2


def directed_extend_vertical(H: Seq, V: Seq, rng: random.Random
                             ) -> Tuple[Seq, Seq]:
    """Same but for thin vertical rectangles (width 1, height ≥ 10), via H."""
    rects = build_rects(H, V)
    candidates = []
    for i, ((x1, x2), (y1, y2)) in enumerate(rects):
        if x2 - x1 == 1 and y2 - y1 >= 10:
            candidates.append(i + 1)
    if not candidates:
        return list(H), list(V)
    lab = rng.choice(candidates)

    H2 = list(H)
    positions = [p for p, x in enumerate(H2) if x == lab]
    p_first, p_last = min(positions), max(positions)
    end = rng.choice(("first", "last"))
    if end == "first":
        p_old = p_first
        p_new = rng.randint(0, p_first - 1) if p_first > 0 else p_first
    else:
        p_old = p_last
        p_new = rng.randint(p_last + 1, len(H2) - 1) if p_last < len(H2) - 1 else p_last
    if p_new == p_old:
        return list(H), list(V)
    H2[p_old], H2[p_new] = H2[p_new], H2[p_old]
    return H2, list(V)


# ---------- experiment driver ----------

def _run_trial(args_dict):
    """
    Worker function: generates one candidate, optionally rejects on triangle-
    freeness, runs LP+ILP, returns a result dict. Module-level so it's
    picklable for multiprocessing.Pool.
    """
    t = args_dict["t"]
    rng = random.Random(args_dict["trial_seed"])
    mode = args_dict["mode"]
    multi = args_dict["multi"]
    H = list(args_dict["H_seed"])
    V = list(args_dict["V_seed"])
    triangle_free_only = args_dict["triangle_free_only"]
    ilp_time = args_dict["ilp_time"]
    threads = args_dict.get("gurobi_threads", 1)

    for _ in range(multi):
        if mode == "random":
            H, V = random_long_swap(H, V, rng)
        else:
            if rng.random() < 0.5:
                H, V = directed_extend_horizontal(H, V, rng)
            else:
                H, V = directed_extend_vertical(H, V, rng)

    # Cheap pre-filter: triangle-freeness.
    if triangle_free_only:
        if not vec_triangle_free(H, V):
            return {"status": "rejected_tf", "trial": t}

    try:
        rects = build_rects(H, V)
        lp, ilp = solve_lp_ilp(rects, time_limit=ilp_time, threads=threads)
    except Exception as e:
        return {"status": "error", "trial": t, "error": f"{type(e).__name__}: {e}"}

    if ilp <= 0:
        return {"status": "error", "trial": t, "error": "ilp_zero"}

    ratio = lp / ilp

    # Verify-on-hit: if this candidate looks like a hit and verify time is set,
    # re-solve with longer time limit to filter phantoms.
    verify_time = args_dict.get("verify_ilp_time", 0.0)
    baseline = args_dict.get("verify_baseline", 0.0)
    verified = False
    if verify_time > 0 and ratio > baseline + 1e-6:
        try:
            lp2, ilp2 = solve_lp_ilp(rects, time_limit=verify_time, threads=threads)
            if ilp2 > 0 and ilp2 != ilp:
                # short ILP was wrong; use the longer-time result
                ilp = ilp2
                ratio = lp / ilp
            verified = True
        except Exception as e:
            return {"status": "error", "trial": t,
                    "error": f"verify_failed: {type(e).__name__}: {e}"}

    return {
        "status": "ok", "trial": t,
        "lp": lp, "ilp": ilp, "ratio": ratio,
        "verified": verified,
        "H": H, "V": V,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--mode", choices=("random", "directed"), default="directed")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--multi", type=int, default=1,
                    help="apply this many mutations per candidate (1=single move)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", default="extend_hits")
    ap.add_argument("--ilp-time", type=float, default=30.0,
                    help="time limit per ILP solve (seconds)")
    ap.add_argument("--verify-ilp-time", type=float, default=0.0,
                    help="if > 0, any candidate that beats baseline in the short ILP "
                         "will be re-solved with this longer time limit. Only hits "
                         "that survive verification are saved/reported.")
    ap.add_argument("--seed-pickle", default=None,
                    help="start from this pickle's (H, V) instead of pristine M_k")
    ap.add_argument("--triangle-free-only", action="store_true",
                    help="reject any candidate whose intersection graph contains a triangle "
                         "(checked cheaply via vectorized max-clique at grid points)")
    ap.add_argument("--workers", type=int, default=1,
                    help="number of parallel worker processes (1 = serial). Each worker "
                         "uses Gurobi with Threads=1 to avoid oversubscription.")
    ap.add_argument("--gurobi-threads", type=int, default=1,
                    help="threads per Gurobi solve (only used when workers=1)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    H_p, V_p = kbox_instance(args.k)
    rects_p = build_rects(H_p, V_p)
    print(f"Solving pristine M_{args.k}...")
    lp_p, ilp_p = solve_lp_ilp(rects_p, time_limit=args.ilp_time)
    baseline = lp_p / ilp_p
    print(f"  pristine: lp={lp_p:.2f}, ilp={ilp_p:.0f}, gap={baseline:.4f}")
    print()

    # If a seed pickle is given, start mutations from it rather than pristine.
    if args.seed_pickle:
        with open(args.seed_pickle, "rb") as f:
            d = pickle.load(f)
        H_seed = d.get("H") or d.get("best_instance_H")
        V_seed = d.get("V") or d.get("best_instance_V")
        rects_seed = build_rects(H_seed, V_seed)
        lp_s, ilp_s = solve_lp_ilp(rects_seed, time_limit=args.ilp_time)
        seed_gap = lp_s / ilp_s
        print(f"Seeding from {args.seed_pickle}")
        print(f"  seed:     lp={lp_s:.2f}, ilp={ilp_s:.0f}, gap={seed_gap:.4f}")
        print()
        # use the seed as starting point for every trial; bar to clear is seed_gap
        H_p, V_p = list(H_seed), list(V_seed)
        baseline = max(baseline, seed_gap)

    # known finite-n gap formula for Caoduro at this k
    formula = 2 * args.k * args.k / (args.k * args.k + 3 * args.k - 2)
    print(f"  Caoduro formula for k={args.k}: 2k^2/(k^2+3k-2) = {formula:.4f}")
    print(f"  We need gap > {baseline:.4f} for a +1 (= {lp_p / (ilp_p - 1):.4f} if LP unchanged)")
    print()

    os.makedirs(args.save_dir, exist_ok=True)

    # Build trial argument tuples for the worker.
    trial_args = []
    for t in range(args.trials):
        # each trial gets a deterministic but distinct sub-seed
        trial_args.append({
            "t": t,
            "trial_seed": args.seed * 100003 + t,
            "k": args.k,
            "mode": args.mode,
            "multi": args.multi,
            "ilp_time": args.ilp_time,
            "verify_ilp_time": args.verify_ilp_time,
            "verify_baseline": baseline,
            "H_seed": list(H_p),
            "V_seed": list(V_p),
            "triangle_free_only": args.triangle_free_only,
            "gurobi_threads": args.gurobi_threads if args.workers == 1 else 1,
        })

    best = (baseline, -1, list(H_p), list(V_p), lp_p, ilp_p)
    t0 = time.time()
    n_hits = 0
    n_rejected_tf = 0
    n_errors = 0

    if args.triangle_free_only:
        print(f"  [filter] triangle-free-only: candidates with max-clique > 2 "
              f"will be rejected before LP/ILP")

    if args.workers > 1:
        print(f"  [parallel] using {args.workers} worker processes")
        import multiprocessing as mp
        ctx = mp.get_context("spawn")  # safer with Gurobi than fork
        pool = ctx.Pool(processes=args.workers)
        result_iter = pool.imap_unordered(_run_trial, trial_args, chunksize=1)
    else:
        result_iter = (_run_trial(a) for a in trial_args)

    completed = 0
    try:
        for res in result_iter:
            completed += 1
            status = res["status"]
            t = res["trial"]

            if status == "rejected_tf":
                n_rejected_tf += 1
            elif status == "error":
                n_errors += 1
                print(f"  trial {t:>4}: skip ({res.get('error', '?')})")
            elif status == "ok":
                lp = res["lp"]
                ilp = res["ilp"]
                ratio = res["ratio"]
                if ratio > baseline + 1e-6:
                    n_hits += 1
                    tag = ""
                    if ratio > best[0]:
                        best = (ratio, t, res["H"], res["V"], lp, ilp)
                        tag = "  ⭐ NEW BEST"
                        # save it
                        fn = os.path.join(args.save_dir,
                                          f"k{args.k}_extend_t{t}_gap{ratio:.4f}"
                                          + ("_tf1" if args.triangle_free_only else "")
                                          + ".pkl")
                        with open(fn, "wb") as f:
                            pickle.dump({
                                "k": args.k, "H": res["H"], "V": res["V"],
                                "lp": lp, "ilp": ilp, "gap": ratio,
                                "trial": t, "mode": args.mode,
                                "triangle_free": args.triangle_free_only,
                            }, f)
                    print(f"  trial {t:>4}: lp={lp:>7.2f}  ilp={ilp:>4.0f}  "
                          f"gap={ratio:.4f}{tag}")

            # progress beacon every 25 completed trials
            if completed % 25 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (args.trials - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{args.trials} done, {elapsed:.0f}s elapsed, "
                      f"~{eta:.0f}s remaining, {n_hits} hits, "
                      f"{n_rejected_tf} tf-rejected, best={best[0]:.4f}]")
    finally:
        if args.workers > 1:
            pool.close()
            pool.join()

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"Done in {elapsed:.0f}s. {n_hits}/{args.trials} hits over baseline.")
    if args.triangle_free_only:
        print(f"  Triangle-free filter: {n_rejected_tf} candidates rejected.")
    if n_errors:
        print(f"  Errors: {n_errors} trials.")
    print(f"Best: gap={best[0]:.4f} at trial {best[1]} (lp={best[4]:.2f}, ilp={best[5]:.0f})")
    if best[1] >= 0:
        print(f"Saved to {args.save_dir}/")
    print()
    if best[0] > baseline + 1e-6:
        improvement = best[5] - ilp_p  # negative = α dropped
        print(f"  ILP changed by {improvement:+.0f} (negative = α dropped → gap up)")
        print(f"  At k={args.k}: pristine ratio {baseline:.4f} → {best[0]:.4f}")
    else:
        print(f"  No improvement. Either the move isn't reachable at k={args.k},")
        print(f"  or this random/directed mutation set isn't enough — try")
        print(f"  --multi 2 or --multi 3 for chained extensions.")


if __name__ == "__main__":
    main()
