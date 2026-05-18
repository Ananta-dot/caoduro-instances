#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_instance.py — per-rectangle LP and ILP values from Gurobi.

For each rectangle in the saved (H, V) instance, prints:
    rect#   LP_value   ILP_selected   (x1,x2)   (y1,y2)

Also prints totals (sum LP, count ILP=1) as a sanity check.

Usage:
    python3 inspect_instance.py elites_above_threshold/k9_r0_gap1.5577_tf1.pkl
    python3 inspect_instance.py <pkl> --time-limit 600

Requires: gurobipy, kbox_misr.py.
"""

import argparse
import pickle
import sys

from kbox_misr import build_rects, grid_points, covers_grid_closed

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: gurobipy required.", file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle", help="path to the saved MISR pickle")
    ap.add_argument("--time-limit", type=float, default=1800.0,
                    help="ILP time limit in seconds (default 1800)")
    args = ap.parse_args()

    with open(args.pickle, "rb") as f:
        d = pickle.load(f)
    H = d.get("H") or d.get("best_instance_H")
    V = d.get("V") or d.get("best_instance_V")
    if H is None or V is None:
        print("ERROR: pickle missing H/V keys.", file=sys.stderr)
        sys.exit(1)

    rects = build_rects(H, V)
    n = len(rects)
    pts = grid_points(rects)
    covers = covers_grid_closed(rects, pts)

    print(f"File: {args.pickle}")
    print(f"  reported gap: {d.get('gap', '?')}")
    print(f"  reported tf:  {d.get('triangle_free', '?')}")
    print(f"  reported mc:  {d.get('max_clique', '?')}")
    print(f"  n = {n} rectangles, {len(pts)} grid points,"
          f" {sum(1 for S in covers if len(S) >= 2)} nontrivial cover constraints")
    print()

    # -------- clique LP --------
    print("Solving clique LP...")
    m_lp = gp.Model("lp")
    m_lp.setParam("OutputFlag", 0)
    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if len(S) >= 2:
            m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1)
    m_lp.optimize()
    if m_lp.status != GRB.OPTIMAL:
        print(f"ERROR: LP did not solve (status={m_lp.status})", file=sys.stderr)
        sys.exit(1)
    lp_total = m_lp.objVal
    lp_vals = [x[i].X for i in range(n)]
    print(f"  LP total = {lp_total:.4f}\n")

    # -------- ILP --------
    print(f"Solving ILP (TimeLimit={args.time_limit:.0f}s, MIPFocus=2)...")
    m_ilp = gp.Model("ilp")
    m_ilp.setParam("OutputFlag", 0)
    m_ilp.setParam("TimeLimit", args.time_limit)
    m_ilp.setParam("MIPFocus", 2)
    y = m_ilp.addVars(n, vtype=GRB.BINARY, name="y")
    m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if len(S) >= 2:
            m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1)
    m_ilp.optimize()
    if m_ilp.SolCount == 0:
        print(f"ERROR: ILP found no feasible solution (status={m_ilp.status})",
              file=sys.stderr)
        sys.exit(1)
    ilp_total = m_ilp.objVal
    ilp_vals = [int(round(y[i].X)) for i in range(n)]
    proved = (m_ilp.status == GRB.OPTIMAL)
    print(f"  ILP total = {ilp_total:.0f}  "
          f"(proved_optimal={proved}, mip_gap={m_ilp.MIPGap:.4f})\n")

    # -------- Per-rectangle dump --------
    print(f"{'rect':>4}  {'LP':>7}  {'ILP':>3}  {'x_span':>12}  {'y_span':>12}")
    print("-" * 50)
    for i in range(n):
        (x1, x2), (y1, y2) = rects[i]
        x_str = f"({x1},{x2})"
        y_str = f"({y1},{y2})"
        print(f"{i+1:>4}  {lp_vals[i]:>7.4f}  {ilp_vals[i]:>3}  "
              f"{x_str:>12}  {y_str:>12}")

    print()
    print(f"--- TOTALS ---")
    print(f"  sum(LP)        = {sum(lp_vals):.4f}")
    print(f"  sum(ILP)       = {sum(ilp_vals)}  (this is alpha)")
    print(f"  LP / ILP       = {lp_total/ilp_total:.6f}")
    print(f"  #rects LP > 0  = {sum(1 for v in lp_vals if v > 1e-6)}/{n}")
    print(f"  #rects ILP=1   = {sum(ilp_vals)}/{n}")
    print(f"  #rects LP=0.5  = {sum(1 for v in lp_vals if abs(v - 0.5) < 1e-4)}")
    print(f"  #rects LP<0.5  = {sum(1 for v in lp_vals if v < 0.5 - 1e-4)}")


if __name__ == "__main__":
    main()
