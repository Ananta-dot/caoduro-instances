 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_instance.py — exhaustive verification of a saved MISR pickle.

Solves the instance three ways and reports whether the claimed gap is real:
  (a) clique LP  — the relaxation your search uses (intersection regions)
  (b) edge LP    — the paper's relaxation (pairwise intersection constraints),
                   equal to n/2 iff the instance is triangle-free
  (c) ILP        — integer optimum, with proof-of-optimality enabled

The output tells you:
  * Whether the instance is triangle-free (matches the paper's setting)
  * Whether edge_LP == n/2 (expected if triangle-free)
  * Whether the ILP was proved optimal (not just feasible)
  * The two different "gap" metrics — yours vs the paper's

Usage:
  python3 verify_instance.py <pickle_path> [time_limit_seconds]

Defaults to a 1800s (30 min) ILP time limit. For large n (>= 400), use 3600+.

Examples:
  python3 verify_instance.py elites_above_threshold/k9_r2_gap1.5429_tf1.pkl
  python3 verify_instance.py elites_above_threshold/k10_r3_gap1.5748_tf1.pkl 3600

Requires: gurobipy, kbox_misr.py, kbox_parallel.py (all in the same dir).
"""

from __future__ import annotations

import pickle
import sys
import time

from kbox_misr import build_rects, grid_points, covers_grid_closed
from kbox_parallel import vec_max_clique_at_grid

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: gurobipy required.", file=sys.stderr)
    sys.exit(2)


def solve_clique_lp(rects, covers, grb_threads: int = 0) -> float:
    """The same LP your search uses — one constraint per intersection region."""
    m = gp.Model("clique_lp")
    m.setParam("OutputFlag", 0)
    if grb_threads > 0:
        m.setParam("Threads", grb_threads)
    # large LP: concurrent simplex is often fastest
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


def solve_edge_lp(rects, grb_threads: int = 0) -> tuple:
    """Pairwise edge LP — equals n/2 iff the intersection graph is triangle-free."""
    m = gp.Model("edge_lp")
    m.setParam("OutputFlag", 0)
    if grb_threads > 0:
        m.setParam("Threads", grb_threads)
    m.setParam("Method", 3)
    m.setParam("Crossover", 0)
    n = len(rects)
    y = m.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")
    m.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)

    edges_added = 0
    for i in range(n):
        (xi1, xi2), (yi1, yi2) = rects[i]
        for j in range(i + 1, n):
            (xj1, xj2), (yj1, yj2) = rects[j]
            if (max(xi1, xj1) <= min(xi2, xj2)
                    and max(yi1, yj1) <= min(yi2, yj2)):
                m.addConstr(y[i] + y[j] <= 1)
                edges_added += 1

    m.optimize()
    val = float(m.objVal) if m.status == GRB.OPTIMAL else float("nan")
    return val, edges_added


def solve_ilp(rects, covers, time_limit: float, grb_threads: int = 0) -> dict:
    """ILP with proof-of-optimality focus. Returns value, proved_optimal, gap."""
    m = gp.Model("ilp")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    m.setParam("MIPFocus", 2)   # prove optimality (vs. MIPFocus=1 which chases feasible)
    m.setParam("Presolve", 2)
    m.setParam("Cuts", 2)
    if grb_threads > 0:
        m.setParam("Threads", grb_threads)
    n = len(rects)
    z = m.addVars(n, vtype=GRB.BINARY, name="z")
    m.setObjective(gp.quicksum(z[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if len(S) >= 2:
            m.addConstr(gp.quicksum(z[i] for i in S) <= 1)
    m.optimize()

    if m.SolCount == 0:
        return {"value": None, "proved_optimal": False, "mip_gap": None,
                "status": m.status}

    return {
        "value": float(m.objVal),
        "proved_optimal": (m.status == GRB.OPTIMAL),
        "mip_gap": float(m.MIPGap),
        "status": m.status,
        "best_bound": float(m.ObjBound),
    }


def main():
    if len(sys.argv) < 2:
        print("usage: python3 verify_instance.py <pickle_path> [time_limit_seconds]")
        sys.exit(1)

    path = sys.argv[1]
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 1800.0

    # --- load ---
    with open(path, "rb") as f:
        d = pickle.load(f)
    H = d.get("H") or d.get("best_instance_H")
    V = d.get("V") or d.get("best_instance_V")
    if H is None or V is None:
        print(f"ERROR: pickle missing H/V keys. Keys: {list(d.keys())}",
              file=sys.stderr)
        sys.exit(1)

    print(f"=" * 72)
    print(f"File:          {path}")
    print(f"Reported k:    {d.get('k', '?')}")
    print(f"Reported gap:  {d.get('gap', d.get('best_overall', '?'))}")
    print(f"Reported tf:   {d.get('triangle_free', d.get('best_triangle_free', '?'))}")
    print(f"Reported mc:   {d.get('max_clique', d.get('best_max_clique', '?'))}")
    print(f"=" * 72)

    rects = build_rects(H, V)
    n = len(rects)
    pts = grid_points(rects)
    covers = covers_grid_closed(rects, pts)
    print(f"n = {n} rectangles,  {len(pts)} grid points,  "
          f"{sum(1 for S in covers if len(S) >= 2)} non-trivial cover constraints")

    # --- recheck triangle-freeness ---
    mc = vec_max_clique_at_grid(H, V)
    print(f"\nMax clique at any grid point: {mc}  "
          f"(triangle-free iff <= 2)")

    # --- (a) clique LP ---
    print("\n[1/3] Solving clique LP (your search's relaxation)...")
    t0 = time.perf_counter()
    clique_lp = solve_clique_lp(rects, covers)
    print(f"      clique_LP = {clique_lp:.6f}  [{time.perf_counter()-t0:.1f}s]")

    # --- (b) edge LP ---
    print("\n[2/3] Solving edge LP (paper's alpha*)...")
    t0 = time.perf_counter()
    edge_lp, n_edges = solve_edge_lp(rects)
    print(f"      edge_LP   = {edge_lp:.6f}  ({n_edges} edges)  "
          f"[{time.perf_counter()-t0:.1f}s]")
    expected_nhalf = n / 2.0
    print(f"      n/2       = {expected_nhalf:.4f}  "
          f"({'MATCH' if abs(edge_lp - expected_nhalf) < 1e-4 else 'DIFFERS'})")

    # --- (c) ILP ---
    print(f"\n[3/3] Solving ILP (TimeLimit={time_limit:.0f}s, MIPFocus=2)...")
    print("      Warning: this can take a long time at large n.")
    t0 = time.perf_counter()
    ilp_res = solve_ilp(rects, covers, time_limit=time_limit)
    print(f"      [{time.perf_counter()-t0:.1f}s elapsed]")
    if ilp_res["value"] is None:
        print(f"      ILP: NO SOLUTION FOUND (status={ilp_res['status']})")
        return
    print(f"      ILP value     = {ilp_res['value']:.0f}")
    print(f"      best bound    = {ilp_res.get('best_bound', '?')}")
    print(f"      mip_gap       = {ilp_res['mip_gap']:.6f}")
    print(f"      proved optimal: {ilp_res['proved_optimal']}")

    # --- summary ---
    ilp = ilp_res["value"]
    print(f"\n" + "=" * 72)
    print(f"SUMMARY")
    print(f"=" * 72)
    print(f"  triangle-free              : {mc <= 2}")
    print(f"  edge_LP == n/2             : {abs(edge_lp - expected_nhalf) < 1e-4}")
    print(f"  ILP proved optimal         : {ilp_res['proved_optimal']}")
    print()
    print(f"  clique_LP / ILP            : {clique_lp/ilp:.6f}  "
          f"(your metric)")
    print(f"  edge_LP   / ILP            : {edge_lp/ilp:.6f}  "
          f"(paper's alpha*/alpha)")
    print()
    reported_gap = d.get("gap", d.get("best_overall"))
    if reported_gap is not None:
        print(f"  reported_gap               : {reported_gap:.6f}")
        diff = clique_lp / ilp - reported_gap
        print(f"  recomputed - reported      : {diff:+.6f}  "
              f"({'match' if abs(diff) < 1e-4 else 'differs!'})")

    if not ilp_res["proved_optimal"]:
        print()
        print("  NOTE: ILP did not prove optimality within time limit.")
        print(f"        True alpha may be higher than {ilp:.0f}, in which")
        print(f"        case the true gap is <= {clique_lp/ilp:.4f}.")
        print(f"        Re-run with a longer time_limit to resolve.")


if __name__ == "__main__":
    main()
