#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enumerate_perturbations.py
==========================

Brute-force enumerate every single-position perturbation of pristine M_k
and find the ones that produce alpha < Caoduro's value. Each perturbation
is a single transposition in H or V (the smallest possible move).

Three classes of perturbation tested per label i:
  - H-shrink-left:  move label i's earlier H position one step right
  - H-shrink-right: move label i's later H position one step left
  - V-shrink-left:  same for V
  - V-shrink-right: same for V

Plus the symmetric "extend" variants (move outward instead of inward).
Total: ~8 perturbations per label × n labels = 8n candidates per k.

Each candidate is canonicalized, deduped, then has its LP and ILP solved
in parallel. ILP uses MIPFocus=2 with a small time budget (most should
finish in seconds because the modification is local).

Output: a CSV of (label_modified, modification_type, mc, lp, ilp, gap)
for every candidate that beat or matched the Caoduro baseline.

Usage:
  python3 enumerate_perturbations.py --k 11 --workers 8 --ilp-time 30
  python3 enumerate_perturbations.py --k 12 --workers 8 --ilp-time 60 \\
      --save-best elites_above_threshold/

Honest expectations:
  - If a single-step modification produces +1 improvement, this finds it.
  - If only multi-step modifications work, this finds nothing.
  - Either result is informative for the structural question.
"""

from __future__ import annotations

# Force spawn before any torch-touching imports (defensive — this script
# doesn't use torch, but matches the rest of the pipeline)
import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import argparse
import csv
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

from kbox_misr import (kbox_instance, build_rects, grid_points,
                       covers_grid_closed, canonicalize, instance_key, Seq)


# Worker-global Gurobi env (set up in _init_worker, like in kbox_fast.py)
_ENV = None


def _init_worker(grb_threads: int):
    global _ENV
    import gurobipy as gp
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    if grb_threads > 0:
        env.setParam("Threads", grb_threads)
    env.start()
    _ENV = env


def _solve_one(payload):
    """
    Score one perturbation. Runs in a worker.

    payload: (candidate_id, modification_label, modification_type, H, V, ilp_time)
    returns: (id, mod_label, mod_type, mc, lp, ilp, proved, error_or_None)
    """
    (cand_id, mod_label, mod_type, H, V, ilp_time) = payload
    try:
        import gurobipy as gp
        from gurobipy import GRB

        rects = build_rects(H, V)
        n = len(rects)
        pts = grid_points(rects)
        covers = covers_grid_closed(rects, pts)
        mc = max((len(c) for c in covers), default=0)

        # --- LP ---
        m_lp = gp.Model("lp", env=_ENV)
        m_lp.setParam("OutputFlag", 0)
        if n >= 300:
            m_lp.setParam("Method", 3)
            m_lp.setParam("Crossover", 0)
        x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
        m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
        for S in covers:
            if len(S) >= 2:
                m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1)
        m_lp.optimize()
        lp_val = float(m_lp.objVal) if m_lp.status == GRB.OPTIMAL else None

        # --- ILP with proved-optimal requirement ---
        m_ilp = gp.Model("ilp", env=_ENV)
        m_ilp.setParam("OutputFlag", 0)
        m_ilp.setParam("TimeLimit", ilp_time)
        m_ilp.setParam("MIPFocus", 2)
        m_ilp.setParam("Presolve", 2)
        m_ilp.setParam("Cuts", 2)
        y = m_ilp.addVars(n, vtype=GRB.BINARY, name="y")
        m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
        for S in covers:
            if len(S) >= 2:
                m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1)
        m_ilp.optimize()
        proved = (m_ilp.status == GRB.OPTIMAL)
        ilp_val = float(m_ilp.objVal) if m_ilp.SolCount > 0 else None

        return (cand_id, mod_label, mod_type, mc, lp_val, ilp_val, proved, None)

    except Exception:
        return (cand_id, mod_label, mod_type, None, None, None, False,
                traceback.format_exc())


def gen_single_perturbations(H: Seq, V: Seq, n: int) -> List[Tuple[str, str, Seq, Seq]]:
    """
    Yield every single-transposition perturbation of (H, V).

    Returns: list of (mod_type, mod_label_str, H_new, V_new).
    mod_type is one of:
      'H_shrink_left', 'H_shrink_right', 'H_extend_left', 'H_extend_right',
      'V_shrink_left', 'V_shrink_right', 'V_extend_left', 'V_extend_right'.
    """
    out = []
    for which, seq_src in [('H', H), ('V', V)]:
        for lab in range(1, n + 1):
            positions = [i for i, x in enumerate(seq_src) if x == lab]
            if len(positions) != 2:
                continue
            p1, p2 = positions  # p1 < p2

            for direction, mod_name in [
                ('shrink_left', f'{which}_shrink_left'),
                ('shrink_right', f'{which}_shrink_right'),
                ('extend_left', f'{which}_extend_left'),
                ('extend_right', f'{which}_extend_right'),
            ]:
                S = list(seq_src)
                if direction == 'shrink_left' and p1 + 1 < p2:
                    # move p1 to p1+1 by swapping
                    S[p1], S[p1 + 1] = S[p1 + 1], S[p1]
                elif direction == 'shrink_right' and p2 - 1 > p1:
                    S[p2], S[p2 - 1] = S[p2 - 1], S[p2]
                elif direction == 'extend_left' and p1 > 0:
                    S[p1], S[p1 - 1] = S[p1 - 1], S[p1]
                elif direction == 'extend_right' and p2 + 1 < len(S):
                    S[p2], S[p2 + 1] = S[p2 + 1], S[p2]
                else:
                    continue  # at boundary; skip

                if which == 'H':
                    H_new, V_new = S, list(V)
                else:
                    H_new, V_new = list(H), S
                H_new, V_new = canonicalize(H_new, V_new)
                out.append((mod_name, f'lab_{lab}', H_new, V_new))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--workers", type=int, default=None,
                    help="number of parallel workers (default: CPU count)")
    ap.add_argument("--grb-threads", type=int, default=1)
    ap.add_argument("--ilp-time", type=float, default=30.0,
                    help="ILP time limit per candidate, seconds")
    ap.add_argument("--save-best", default=None,
                    help="directory to save .pkl for any candidate matching "
                         "or beating Caoduro (default: no save)")
    ap.add_argument("--csv-out", default=None,
                    help="path to write all-candidates CSV")
    args = ap.parse_args()

    k = args.k
    n = 4 * k * k
    caoduro_alpha = k * k + 3 * k - 2
    caoduro_lp = 2 * k * k
    caoduro_gap = caoduro_lp / caoduro_alpha

    if args.workers is None:
        args.workers = max(1, (os.cpu_count() or 1) // max(1, args.grb_threads))

    print(f"=== enumerate_perturbations  k={k}, n={n} ===")
    print(f"Caoduro baseline: alpha*={caoduro_lp}, alpha={caoduro_alpha}, "
          f"gap={caoduro_gap:.4f}")
    print(f"Workers: {args.workers}, Gurobi threads/worker: {args.grb_threads}")
    print(f"ILP time per candidate: {args.ilp_time:.0f}s")
    print()

    # Build pristine M_k and generate all single perturbations
    H_p, V_p = kbox_instance(k)
    print("Generating perturbations from pristine M_{}...".format(k))
    candidates = gen_single_perturbations(H_p, V_p, n)
    print(f"  Generated {len(candidates)} candidates (before dedup).")

    # Dedupe by instance_key
    seen = set()
    unique_candidates = []
    for (mt, ml, H, V) in candidates:
        key = instance_key(H, V)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append((mt, ml, H, V))
    print(f"  After dedup: {len(unique_candidates)} unique candidates.")

    # Verify pristine itself (sanity)
    print(f"\nVerifying pristine M_{k} first (sanity check)...")
    pristine_result = _check_one_inline(H_p, V_p, args.ilp_time, args.grb_threads)
    if pristine_result is None:
        print("  ERROR: could not solve pristine. Exiting.")
        return
    p_lp, p_ilp, p_mc, p_proved = pristine_result
    print(f"  Pristine: mc={p_mc}, LP={p_lp:.1f}, ILP={p_ilp:.0f}, "
          f"gap={p_lp/p_ilp:.4f}, proved={p_proved}")
    if abs(p_ilp - caoduro_alpha) > 0.5 or not p_proved:
        print(f"  WARNING: pristine didn't verify cleanly (expected ILP={caoduro_alpha}).")
    print()

    # Parallel scoring
    print(f"Scoring {len(unique_candidates)} candidates in parallel "
          f"({args.workers} workers)...")
    t0 = time.time()
    results = []
    payloads = [(i, ml, mt, H, V, args.ilp_time)
                for i, (mt, ml, H, V) in enumerate(unique_candidates)]

    ctx = _mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(args.grb_threads,),
    ) as ex:
        futs = [ex.submit(_solve_one, p) for p in payloads]
        n_done = 0
        for f in as_completed(futs):
            r = f.result()
            n_done += 1
            results.append(r)
            if n_done % 50 == 0 or n_done == len(payloads):
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 0.01)
                eta = (len(payloads) - n_done) / max(rate, 0.01)
                print(f"  done: {n_done}/{len(payloads)}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    # Sort by gap descending
    results_with_gap = []
    for (cid, ml, mt, mc, lp, ilp, proved, err) in results:
        if err is not None:
            continue
        if lp is None or ilp is None or ilp <= 0 or not proved:
            continue
        gap = lp / ilp
        results_with_gap.append((cid, ml, mt, mc, lp, ilp, gap))
    results_with_gap.sort(key=lambda x: -x[6])

    # Filter to "beats or matches Caoduro"
    interesting = [r for r in results_with_gap if r[6] >= caoduro_gap - 1e-6]

    # Report
    print(f"\n=== RESULTS ===")
    print(f"Total candidates solved with proved-optimal ILP: "
          f"{len(results_with_gap)} / {len(payloads)}")
    print(f"Candidates at/above Caoduro baseline ({caoduro_gap:.4f}): "
          f"{len(interesting)}")
    print()

    if interesting:
        print(f"{'rank':>4}  {'mod_label':>12}  {'mod_type':>20}  "
              f"{'mc':>3}  {'LP':>7}  {'ILP':>4}  {'gap':>8}  {'delta':>7}")
        print("-" * 80)
        for rank, (cid, ml, mt, mc, lp, ilp, gap) in enumerate(interesting[:30]):
            delta = gap - caoduro_gap
            print(f"{rank+1:>4}  {ml:>12}  {mt:>20}  "
                  f"{mc:>3}  {lp:>7.1f}  {ilp:>4.0f}  {gap:>8.4f}  "
                  f"{delta:>+7.4f}")
        if len(interesting) > 30:
            print(f"... and {len(interesting) - 30} more.")
    else:
        print("No candidates matched or beat Caoduro.")
        print(f"Best gap found: "
              f"{results_with_gap[0][6]:.4f}" if results_with_gap else "(none)")

    # Save CSV
    if args.csv_out:
        with open(args.csv_out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['rank', 'mod_label', 'mod_type', 'max_clique',
                        'LP', 'ILP', 'gap', 'delta_vs_caoduro'])
            for rank, (cid, ml, mt, mc, lp, ilp, gap) in enumerate(results_with_gap):
                w.writerow([rank + 1, ml, mt, mc,
                            f"{lp:.4f}", f"{ilp:.0f}", f"{gap:.6f}",
                            f"{gap - caoduro_gap:+.6f}"])
        print(f"\nFull results CSV: {args.csv_out}")

    # Save best pickles
    if args.save_best and interesting:
        os.makedirs(args.save_best, exist_ok=True)
        # find the candidate H, V for each top-N
        top_n = min(5, len(interesting))
        print(f"\nSaving top {top_n} pickles to {args.save_best}/...")
        for rank, (cid, ml, mt, mc, lp, ilp, gap) in enumerate(interesting[:top_n]):
            (orig_mt, orig_ml, H, V) = unique_candidates[cid]
            tf = (mc <= 2)
            fname = (f"enum_k{k}_{orig_mt}_{orig_ml}_"
                     f"gap{gap:.4f}_tf{int(tf)}.pkl")
            path = os.path.join(args.save_best, fname)
            data = {
                "k": k, "round": -2,  # sentinel for "enumeration"
                "H": list(H), "V": list(V),
                "gap": float(gap),
                "triangle_free": tf,
                "max_clique": int(mc),
                "n": n,
                "modification_type": orig_mt,
                "modification_label": orig_ml,
                "source": "enumerate_perturbations.py",
                "timestamp": time.time(),
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)
            print(f"  saved: {path}")


def _check_one_inline(H, V, ilp_time, grb_threads):
    """Verify one instance in-process (not in a worker) for the pristine sanity check."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        if grb_threads > 0:
            env.setParam("Threads", grb_threads)
        env.start()
        rects = build_rects(H, V)
        n = len(rects)
        pts = grid_points(rects)
        covers = covers_grid_closed(rects, pts)
        mc = max((len(c) for c in covers), default=0)

        m_lp = gp.Model("lp", env=env)
        m_lp.setParam("OutputFlag", 0)
        x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
        m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
        for S in covers:
            if len(S) >= 2:
                m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1)
        m_lp.optimize()
        lp = float(m_lp.objVal)

        m_ilp = gp.Model("ilp", env=env)
        m_ilp.setParam("OutputFlag", 0)
        m_ilp.setParam("TimeLimit", ilp_time)
        m_ilp.setParam("MIPFocus", 2)
        y = m_ilp.addVars(n, vtype=GRB.BINARY)
        m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
        for S in covers:
            if len(S) >= 2:
                m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1)
        m_ilp.optimize()
        proved = (m_ilp.status == GRB.OPTIMAL)
        ilp = float(m_ilp.objVal) if m_ilp.SolCount > 0 else None
        return (lp, ilp, mc, proved)
    except Exception as e:
        print(f"  pristine check error: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    main()
