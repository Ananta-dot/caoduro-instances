#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_reverify.py
=================

Re-verify a batch of saved MISR pickles with proven-optimality enforcement
and long ILP time budgets. Distinguishes real gaps from phantom gaps caused
by extend_experiment.py's "use whatever ILP we found" behavior.

For each pickle:
  1. Read reported clique_LP and ILP from the file (the values that produced
     the saved gap).
  2. Re-solve clique_LP exactly (sanity check).
  3. Re-solve ILP with MIPFocus=2 and a long time budget, REQUIRING
     proof-of-optimality. If Gurobi can't prove optimal, mark UNVERIFIED.
  4. Recompute gap, compare to reported, and classify:
        VERIFIED     — proved optimal, recomputed gap close to reported
        REVISED_DOWN — proved optimal, recomputed gap significantly LOWER
                       than reported (phantom hit)
        UNVERIFIED   — Gurobi couldn't prove optimal within budget
        ERROR        — something else went wrong

Writes per-pickle results to a CSV. The headline output is which (if any)
of your pickles represent verified, real improvements over Caoduro.

Usage:
    python3 batch_reverify.py --pickles elites_above_threshold/k13_*.pkl \\
        --ilp-time 7200 --csv-out reverify_k13.csv

    python3 batch_reverify.py --pickles elites_above_threshold/k15_*.pkl \\
        --ilp-time 14400 --csv-out reverify_k15.csv

    # Re-verify everything from the suspicious sweep range
    python3 batch_reverify.py --pickles \\
        elites_above_threshold/k13_*.pkl \\
        elites_above_threshold/k14_*.pkl \\
        elites_above_threshold/k15_*.pkl \\
        --ilp-time 7200 \\
        --csv-out reverify_all.csv

Honest expectations:
  - At n=676 (k=13), each ILP can take ~1 hour to prove optimal.
  - At n=784 (k=14), each ILP can take 2-4 hours.
  - At n=900 (k=15), each ILP can take 4+ hours; may not prove optimal at all.
  - Running 9 pickles end-to-end overnight: comfortable for k=13, tight
    for k=14, risky for k=15.

Run k=13 first. If everything is phantom, k=14 and k=15 almost certainly
are too — don't burn the larger budgets.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import sys
import time
from typing import List, Optional

from kbox_misr import build_rects, grid_points, covers_grid_closed

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: gurobipy required.", file=sys.stderr)
    sys.exit(2)


def caoduro_gap(k: int) -> float:
    return 2 * k * k / (k * k + 3 * k - 2)


def solve_clique_lp(rects, covers, threads: int = 0) -> Optional[float]:
    m = gp.Model("lp")
    m.setParam("OutputFlag", 0)
    if threads > 0:
        m.setParam("Threads", threads)
    m.setParam("Method", 3)
    m.setParam("Crossover", 0)
    n = len(rects)
    x = m.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
    m.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if len(S) >= 2:
            m.addConstr(gp.quicksum(x[i] for i in S) <= 1)
    m.optimize()
    if m.status == GRB.OPTIMAL:
        return float(m.objVal)
    return None


def solve_ilp_proved(rects, covers, time_limit: float, threads: int = 0):
    """
    Returns (ilp_value, proved_optimal, best_bound, mip_gap, sol_count).
    Uses MIPFocus=2 to prioritize proving optimality.
    """
    m = gp.Model("ilp")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    m.setParam("MIPFocus", 2)
    m.setParam("Presolve", 2)
    m.setParam("Cuts", 2)
    if threads > 0:
        m.setParam("Threads", threads)
    n = len(rects)
    y = m.addVars(n, vtype=GRB.BINARY)
    m.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if len(S) >= 2:
            m.addConstr(gp.quicksum(y[i] for i in S) <= 1)
    m.optimize()
    if m.SolCount == 0:
        return None, False, None, None, 0
    return (
        float(m.objVal),
        (m.status == GRB.OPTIMAL),
        float(m.ObjBound),
        float(m.MIPGap),
        m.SolCount,
    )


def reverify_one(pkl_path: str, ilp_time: float) -> dict:
    """Verify one pickle. Returns a dict suitable for CSV writing."""
    result = {
        "pickle": pkl_path,
        "k": None,
        "n": None,
        "reported_gap": None,
        "reported_lp": None,
        "reported_ilp": None,
        "recomputed_lp": None,
        "recomputed_ilp": None,
        "proved_optimal": None,
        "mip_gap": None,
        "best_bound": None,
        "recomputed_gap": None,
        "caoduro_baseline": None,
        "delta_caoduro": None,
        "delta_reported": None,
        "max_clique": None,
        "verdict": "ERROR",
        "wall_time_s": 0.0,
        "notes": "",
    }

    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
    except Exception as e:
        result["notes"] = f"load failed: {e}"
        return result

    H = d.get("H") or d.get("best_instance_H")
    V = d.get("V") or d.get("best_instance_V")
    if H is None or V is None:
        result["notes"] = "missing H/V"
        return result

    k = d.get("k")
    if k is None:
        try:
            n_lab = max(max(H), max(V))
            k = int(round((n_lab / 4) ** 0.5))
        except Exception:
            result["notes"] = "couldn't infer k"
            return result

    n = 4 * k * k
    result["k"] = k
    result["n"] = n
    result["reported_gap"] = d.get("gap")
    result["reported_lp"] = d.get("lp")
    result["reported_ilp"] = d.get("ilp")
    result["caoduro_baseline"] = caoduro_gap(k)

    rects = build_rects(H, V)
    pts = grid_points(rects)
    covers = covers_grid_closed(rects, pts)

    # max-clique check (cheap)
    mc = max((len(S) for S in covers), default=0)
    result["max_clique"] = mc

    t_start = time.time()

    # LP
    lp = solve_clique_lp(rects, covers)
    if lp is None:
        result["notes"] = "LP failed"
        result["wall_time_s"] = time.time() - t_start
        return result
    result["recomputed_lp"] = lp

    # ILP with proof-of-optimality required
    ilp, proved, best_bound, mip_gap, sol_count = solve_ilp_proved(
        rects, covers, time_limit=ilp_time
    )

    result["wall_time_s"] = time.time() - t_start

    if ilp is None:
        result["notes"] = "ILP found no feasible solution"
        result["verdict"] = "ERROR"
        return result

    result["recomputed_ilp"] = ilp
    result["proved_optimal"] = proved
    result["best_bound"] = best_bound
    result["mip_gap"] = mip_gap

    if ilp > 0:
        result["recomputed_gap"] = lp / ilp
        result["delta_caoduro"] = lp / ilp - result["caoduro_baseline"]
        if result["reported_gap"] is not None:
            result["delta_reported"] = lp / ilp - result["reported_gap"]

    # Verdict
    if not proved:
        result["verdict"] = "UNVERIFIED"
        result["notes"] = (
            f"ILP didn't prove optimal in {ilp_time}s "
            f"(bound={best_bound}, mip_gap={mip_gap:.4f})"
        )
    elif result["reported_gap"] is None or abs(result["delta_reported"]) <= 1e-4:
        result["verdict"] = "VERIFIED"
    elif result["delta_reported"] < -0.005:
        result["verdict"] = "REVISED_DOWN"
        result["notes"] = (
            f"reported {result['reported_gap']:.4f} → "
            f"recomputed {result['recomputed_gap']:.4f} "
            f"({result['delta_reported']:+.4f})"
        )
    else:
        result["verdict"] = "VERIFIED"
        result["notes"] = (
            f"reported {result['reported_gap']:.4f} "
            f"vs recomputed {result['recomputed_gap']:.4f}"
        )

    return result


def print_progress(result: dict, idx: int, total: int):
    """Print a one-line status for this pickle."""
    elapsed = result["wall_time_s"]
    verdict = result["verdict"]

    if verdict == "VERIFIED":
        symbol = "[OK]"
    elif verdict == "REVISED_DOWN":
        symbol = "[PHANTOM]"
    elif verdict == "UNVERIFIED":
        symbol = "[??]"
    else:
        symbol = "[ERR]"

    rep_gap = result["reported_gap"]
    rec_gap = result["recomputed_gap"]
    mc = result["max_clique"]
    delta_caoduro = result["delta_caoduro"]

    rep_str = f"{rep_gap:.4f}" if rep_gap else "?"
    rec_str = f"{rec_gap:.4f}" if rec_gap else "?"
    dC_str = f"{delta_caoduro:+.4f}" if delta_caoduro is not None else "?"

    name = os.path.basename(result["pickle"])
    print(
        f"[{idx}/{total}] {symbol:11} k={result['k']}  mc={mc}  "
        f"reported={rep_str}  recomputed={rec_str}  ΔCaoduro={dC_str}  "
        f"({elapsed:.0f}s)  {name}"
    )
    if result["notes"]:
        print(f"          {result['notes']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickles", nargs="+", required=True,
                    help="paths to pickles to re-verify (globs OK)")
    ap.add_argument("--ilp-time", type=float, default=7200.0,
                    help="time limit per ILP solve in seconds (default 7200)")
    ap.add_argument("--threads", type=int, default=0,
                    help="Gurobi threads per solve (0 = auto)")
    ap.add_argument("--csv-out", default="reverify_results.csv",
                    help="output CSV path")
    ap.add_argument("--skip-failed", action="store_true",
                    help="if ILP fails to prove optimal, skip subsequent "
                         "pickles of higher k (saves time)")
    args = ap.parse_args()

    # Expand globs in case the shell didn't
    expanded = []
    for p in args.pickles:
        if any(c in p for c in "*?["):
            expanded.extend(sorted(glob.glob(p)))
        else:
            expanded.append(p)
    pickles = [p for p in expanded if os.path.isfile(p)]
    if not pickles:
        print(f"No pickle files found matching: {args.pickles}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Re-verifying {len(pickles)} pickles with ilp-time={args.ilp_time:.0f}s")
    print(f"  CSV output: {args.csv_out}")
    print(f"  Threads:    {args.threads if args.threads > 0 else 'auto'}")
    print()

    results = []
    largest_unverified_k = None

    for idx, pkl in enumerate(pickles, 1):
        # Skip-failed heuristic: if a smaller-k pickle came back UNVERIFIED,
        # larger-k pickles will too (only more so).
        if (args.skip_failed
                and largest_unverified_k is not None):
            try:
                # peek at the pickle to find its k
                with open(pkl, "rb") as f:
                    d_peek = pickle.load(f)
                k_peek = d_peek.get("k")
                if k_peek is None:
                    H_peek = d_peek.get("H") or d_peek.get("best_instance_H")
                    if H_peek:
                        n_peek = max(max(H_peek),
                                     max(d_peek.get("V", [1])))
                        k_peek = int(round((n_peek / 4) ** 0.5))
                if k_peek and k_peek >= largest_unverified_k:
                    print(f"[{idx}/{len(pickles)}] SKIPPED  k={k_peek}  "
                          f"(--skip-failed; k={largest_unverified_k} "
                          f"already unverified)")
                    continue
            except Exception:
                pass

        result = reverify_one(pkl, args.ilp_time)
        results.append(result)
        print_progress(result, idx, len(pickles))

        if (result["verdict"] == "UNVERIFIED"
                and (largest_unverified_k is None
                     or result["k"] < largest_unverified_k)):
            largest_unverified_k = result["k"]

        # Write CSV incrementally so partial runs aren't lost
        with open(args.csv_out, "w", newline="") as f:
            if results:
                w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                w.writeheader()
                for r in results:
                    w.writerow(r)

    # Summary
    print()
    print("=" * 72)
    print("BATCH RE-VERIFICATION SUMMARY")
    print("=" * 72)

    by_verdict = {}
    for r in results:
        by_verdict.setdefault(r["verdict"], []).append(r)

    for verdict in ["VERIFIED", "REVISED_DOWN", "UNVERIFIED", "ERROR"]:
        if verdict in by_verdict:
            rows = by_verdict[verdict]
            print(f"\n{verdict}: {len(rows)} pickles")
            for r in rows:
                name = os.path.basename(r["pickle"])
                if r["recomputed_gap"]:
                    print(f"  k={r['k']}  recomputed={r['recomputed_gap']:.4f}  "
                          f"ΔCaoduro={r['delta_caoduro']:+.4f}  "
                          f"mc={r['max_clique']}  {name}")
                else:
                    print(f"  k={r['k']}  (no result)  {name}")

    print()
    print("=" * 72)
    print("WHICH RESULTS ARE PUBLISHABLE?")
    print("=" * 72)

    publishable = [r for r in results
                   if r["verdict"] == "VERIFIED"
                   and r["delta_caoduro"] is not None
                   and r["delta_caoduro"] > 0]
    if publishable:
        publishable.sort(key=lambda r: (-r["delta_caoduro"]))
        print(f"\n{len(publishable)} verified improvements over Caoduro:")
        for r in publishable:
            print(f"  k={r['k']}  gap={r['recomputed_gap']:.4f}  "
                  f"(Δ={r['delta_caoduro']:+.4f}, mc={r['max_clique']})")
    else:
        print("\nNone of the verified results improve on Caoduro.")
        print("Suggests the existing pipeline doesn't beat Caoduro at the k")
        print("values batched. Re-check the verification command and source")
        print("pickles, or run extend_experiment.py with longer per-trial ILP.")

    print()
    print(f"Full results CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
