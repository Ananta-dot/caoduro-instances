#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_k.py
==========

Overnight driver that runs extend_experiment.py at multiple k values back to
back, with per-k budgets scaled by n. Produces one CSV summarizing best
verified gap per k, lets you see the ceiling of "+1 family at each k" in
one place.

This is not parallel across k — it's *sequential* on a single machine,
because each extend_experiment run already maxes out cores. Running k=12
and k=20 simultaneously oversubscribes both. Sequential lets each k get
the full machine.

Default budgets are calibrated for an M4 Max with 8 performance cores.
Adjust --time-budget-hours if you have less time.

Usage:
    # Reasonable overnight sweep (k=9..15, ~10 hours)
    python3 sweep_k.py --k-list 9 10 11 12 13 14 15 --time-budget-hours 10

    # Aggressive multi-day attempt (k=9..20, ~36 hours total)
    python3 sweep_k.py --k-list 9 10 11 12 13 14 15 16 17 18 19 20 \\
        --time-budget-hours 36

    # Chain-mode at one k (try +1, +2, +3 ... +k iteratively)
    python3 sweep_k.py --chain-mode --k 11 --chain-target 6

Outputs:
  - sweep_k_results.csv          summary table
  - sweep_k_log/                 per-k stdout logs
  - elites from each k go to elites_above_threshold/ as usual

Honest expectations:
  - At k <= 13: ILP certifies reliably, expect verified +1 or better.
  - At k 14-17: certification gets fragile. Some trials will timeout.
  - At k 18+: most trials will not certify within reasonable budget.
    Results in this range are advisory, not publishable.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple


def caoduro_baseline(k: int) -> float:
    return 2 * k * k / (k * k + 3 * k - 2)


def per_trial_seconds_estimate(k: int) -> float:
    """
    Rough estimate of how long one trial (one mutation + LP + ILP) takes
    for proven-optimal certification at this k.

    Calibrated against measured times in your repo:
      k=9 (n=324): ~8s per trial with 30s ILP cap
      k=10 (n=400): ~80s
      k=11 (n=484): ~3 min
      k=12 (n=576): ~10 min
    Extrapolated above k=12 as roughly n^2.5 over a constant.
    """
    n = 4 * k * k
    # fit: t ≈ (n / 200) ^ 2.5 seconds, capped at 7200s = 2 hr
    est = (n / 200) ** 2.5
    return min(7200.0, max(8.0, est))


def ilp_time_for(k: int) -> float:
    """Per-ILP time limit to give Gurobi a fighting chance to certify."""
    n = 4 * k * k
    # 30s at k=9, 60s at k=11, scaling with n
    if n <= 400:
        return 30.0
    if n <= 600:
        return 60.0
    if n <= 1000:
        return 180.0
    if n <= 1600:
        return 900.0  # 15 min at k=20
    return 1800.0   # 30 min cap at very large k


def verify_time_for(k: int) -> float:
    """Verify-on-hit time limit, longer than in-loop ilp_time."""
    return ilp_time_for(k) * 4


def trials_for(k: int) -> int:
    """How many trials to attempt per k. Fewer at large k because each is slow."""
    n = 4 * k * k
    if n <= 400:
        return 100
    if n <= 600:
        return 50
    if n <= 1000:
        return 25
    if n <= 1600:
        return 10
    return 5


def estimate_total_seconds(k_list: List[int]) -> Dict[str, float]:
    """Return (total, per_k_dict)."""
    per_k = {}
    total = 0.0
    for k in k_list:
        t = trials_for(k) * per_trial_seconds_estimate(k)
        # workers in extend_experiment let trials run in parallel,
        # but each trial blocks on its own ILP, so divide by ~6
        # (parallel workers minus contention overhead)
        t = t / 6
        per_k[k] = t
        total += t
    return total, per_k


def run_one_k(k: int, args, out_dir: str, allow_unverified: bool = False
              ) -> Optional[Dict]:
    """
    Run extend_experiment.py for one k. Returns a result dict or None on error.
    Also writes per-k log under out_dir/.
    """
    log_path = os.path.join(out_dir, f"k{k:02d}.log")
    pickle_dir = "elites_above_threshold"

    cmd = [
        sys.executable, "extend_experiment.py",
        "--k", str(k),
        "--mode", args.mode,
        "--trials", str(trials_for(k)),
        "--multi", str(args.multi),
        "--seed", str(args.base_seed + k),
        "--ilp-time", str(ilp_time_for(k)),
        "--verify-ilp-time", str(verify_time_for(k)),
        "--workers", str(args.workers),
        "--save-dir", pickle_dir,
    ]
    if args.triangle_free_only:
        cmd.append("--triangle-free-only")

    print(f"\n{'='*70}")
    print(f"k={k}  (n={4*k*k}, Caoduro baseline {caoduro_baseline(k):.4f})")
    print(f"  trials={trials_for(k)}, ilp-time={ilp_time_for(k):.0f}s, "
          f"verify-ilp-time={verify_time_for(k):.0f}s")
    print(f"  estimated wall: ~{trials_for(k) * per_trial_seconds_estimate(k) / 60 / 6:.0f} min")
    print(f"  log → {log_path}")
    print(f"{'='*70}")

    t_start = time.time()
    try:
        with open(log_path, "w") as logf:
            result = subprocess.run(
                cmd, stdout=logf, stderr=subprocess.STDOUT,
                check=False, timeout=args.per_k_hard_timeout
            )
        elapsed = time.time() - t_start
        print(f"  done in {elapsed:.0f}s (exit={result.returncode})")
        if result.returncode != 0:
            print(f"  WARNING: nonzero exit code; check {log_path}",
                  file=sys.stderr)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t_start
        print(f"  HARD TIMEOUT after {elapsed:.0f}s. Moving on.")

    # Find best pickle for this k that was written during the run
    best = find_best_pickle_for_k(k, pickle_dir)
    return {
        "k": k,
        "n": 4 * k * k,
        "caoduro_baseline": caoduro_baseline(k),
        "trials_attempted": trials_for(k),
        "wall_time_seconds": elapsed,
        "best_pickle": best["path"] if best else None,
        "best_gap": best["gap"] if best else None,
        "best_lp": best["lp"] if best else None,
        "best_ilp": best["ilp"] if best else None,
        "best_tf": best["tf"] if best else None,
        "delta_vs_baseline": (best["gap"] - caoduro_baseline(k))
                              if best else None,
    }


def find_best_pickle_for_k(k: int, pickle_dir: str) -> Optional[Dict]:
    """Scan pickle_dir for the highest-gap pickle at this k."""
    if not os.path.isdir(pickle_dir):
        return None
    best = None
    for fname in os.listdir(pickle_dir):
        if not fname.endswith(".pkl"):
            continue
        try:
            with open(os.path.join(pickle_dir, fname), "rb") as f:
                d = pickle.load(f)
        except Exception:
            continue
        if d.get("k") != k:
            continue
        gap = d.get("gap")
        if gap is None or gap <= 0:
            continue
        if best is None or gap > best["gap"]:
            best = {
                "path": os.path.join(pickle_dir, fname),
                "gap": gap,
                "lp": d.get("lp"),
                "ilp": d.get("ilp"),
                "tf": d.get("triangle_free"),
            }
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-list", type=int, nargs="+",
                    default=[9, 10, 11, 12, 13, 14, 15],
                    help="list of k values to sweep")
    ap.add_argument("--mode", choices=("random", "directed"), default="random",
                    help="mutation mode passed to extend_experiment")
    ap.add_argument("--multi", type=int, default=1,
                    help="mutations per candidate (1=single move)")
    ap.add_argument("--triangle-free-only", action="store_true",
                    help="restrict to triangle-free candidates")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel workers within extend_experiment per k")
    ap.add_argument("--base-seed", type=int, default=1000,
                    help="random seed base; each k uses base_seed + k")
    ap.add_argument("--time-budget-hours", type=float, default=10.0,
                    help="estimated total budget; aborts if exceeded")
    ap.add_argument("--per-k-hard-timeout", type=float, default=14400.0,
                    help="kill any single-k subprocess after this many seconds")
    ap.add_argument("--out-dir", default="sweep_k_log",
                    help="directory for per-k logs")
    ap.add_argument("--results-csv", default="sweep_k_results.csv",
                    help="output CSV path")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan and estimated time, do not run")
    ap.add_argument("--no-budget-check", action="store_true",
                    help="skip the time-budget sanity check at start")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Estimate total wall time
    total_est, per_k_est = estimate_total_seconds(args.k_list)
    print(f"K-sweep plan: {args.k_list}")
    print(f"Per-k estimates (wall seconds):")
    for k in args.k_list:
        print(f"  k={k} (n={4*k*k}): ~{per_k_est[k]:.0f}s "
              f"(~{per_k_est[k]/60:.0f} min)")
    print(f"Total estimated wall: ~{total_est/3600:.1f} hours")
    print(f"Your budget: {args.time_budget_hours:.1f} hours")
    print()

    if total_est / 3600 > args.time_budget_hours and not args.no_budget_check:
        print(f"ABORT: estimated runtime exceeds budget.")
        print(f"  Options:")
        print(f"    - Shorten --k-list (drop large k)")
        print(f"    - Raise --time-budget-hours")
        print(f"    - Pass --no-budget-check to override")
        sys.exit(1)

    if args.dry_run:
        print("Dry run — exiting before any work.")
        return

    # Run each k in turn
    results = []
    overall_start = time.time()
    for k in args.k_list:
        elapsed_so_far = time.time() - overall_start
        if elapsed_so_far > args.time_budget_hours * 3600:
            print(f"\nBudget exhausted at k={k}. Stopping sweep.")
            break
        r = run_one_k(k, args, args.out_dir)
        if r is not None:
            results.append(r)
            # Save CSV after each k so partial runs aren't lost
            with open(args.results_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "k", "n", "caoduro_baseline",
                    "trials_attempted", "wall_time_seconds",
                    "best_gap", "best_lp", "best_ilp", "best_tf",
                    "delta_vs_baseline", "best_pickle",
                ])
                w.writeheader()
                for row in results:
                    w.writerow(row)

    # Final summary
    print("\n" + "=" * 72)
    print("SWEEP COMPLETE")
    print("=" * 72)
    print(f"{'k':>4} {'n':>5} {'Caoduro':>10} {'best_gap':>10} {'Δ':>8} "
          f"{'tf':>3} {'wall':>8}")
    print("-" * 72)
    for r in results:
        wall_min = r["wall_time_seconds"] / 60 if r["wall_time_seconds"] else 0
        best_gap_str = f"{r['best_gap']:.4f}" if r['best_gap'] else "-"
        delta_str = f"{r['delta_vs_baseline']:+.4f}" if r['delta_vs_baseline'] else "-"
        tf_str = str(r['best_tf']) if r['best_tf'] is not None else "-"
        print(f"{r['k']:>4} {r['n']:>5} {r['caoduro_baseline']:>10.4f} "
              f"{best_gap_str:>10} {delta_str:>8} {tf_str:>3} {wall_min:>6.1f}m")

    print(f"\nResults CSV: {args.results_csv}")
    print(f"Per-k logs:  {args.out_dir}/")


if __name__ == "__main__":
    main()
