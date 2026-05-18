#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geom_outliers.py
================

Pull the "real" structural modifications out of a target instance:
the rectangles whose extents are far from any pristine rectangle.
Filters out the 1-unit-wiggle passengers.

Also reports:
  - whether each outlier rectangle is in the target's IS
  - whether it intersects rectangles that are in the target's IS
  - which rectangles disappeared from pristine (potential paired changes)
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter

from kbox_misr import kbox_instance, build_rects


def rects_with_index(H, V):
    return list(enumerate(build_rects(H, V)))  # [(idx, ((x1,x2),(y1,y2))), ...]


def to_tuple(rect):
    (x1, x2), (y1, y2) = rect
    return (x1, x2, y1, y2)


def overlaps(r1, r2):
    (x1a, x2a, y1a, y2a) = r1
    (x1b, x2b, y1b, y2b) = r2
    return max(x1a, x1b) <= min(x2a, x2b) and max(y1a, y1b) <= min(y2a, y2b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle")
    ap.add_argument("--threshold", type=int, default=4,
                    help="L1 distance above which a rect is an outlier")
    args = ap.parse_args()

    with open(args.pickle, "rb") as f:
        d = pickle.load(f)
    H_t = d.get("H") or d.get("best_instance_H")
    V_t = d.get("V") or d.get("best_instance_V")
    IS_t = d.get("IS") or d.get("best_IS") or d.get("ilp_IS") or d.get("solution")
    k = d.get("k")
    if k is None:
        n = max(max(H_t), max(V_t))
        k = int(round((n / 4) ** 0.5))
    n = 4 * k * k

    H_p, V_p = kbox_instance(k)

    rects_t = build_rects(H_t, V_t)
    rects_p = build_rects(H_p, V_p)

    set_t = set(to_tuple(r) for r in rects_t)
    set_p = set(to_tuple(r) for r in rects_p)

    only_t = list(set_t - set_p)
    only_p = list(set_p - set_t)

    # For each rectangle in only_t, find its closest pristine rect (greedy is fine
    # for finding outliers — if a rect is far from EVERY pristine rect, it's an
    # outlier regardless of pairing strategy).
    outliers_t = []
    for rt in only_t:
        best_d = float("inf")
        best_rp = None
        for rp in rects_p:
            rp_t = to_tuple(rp)
            d_ = (abs(rp_t[0] - rt[0]) + abs(rp_t[1] - rt[1])
                  + abs(rp_t[2] - rt[2]) + abs(rp_t[3] - rt[3]))
            if d_ < best_d:
                best_d, best_rp = d_, rp_t
        if best_d >= args.threshold:
            outliers_t.append((rt, best_rp, best_d))

    # And vice versa
    outliers_p = []
    for rp in only_p:
        best_d = float("inf")
        best_rt = None
        for rt in rects_t:
            rt_t = to_tuple(rt)
            d_ = (abs(rt_t[0] - rp[0]) + abs(rt_t[1] - rp[1])
                  + abs(rt_t[2] - rp[2]) + abs(rt_t[3] - rp[3]))
            if d_ < best_d:
                best_d, best_rt = d_, rt_t
        if best_d >= args.threshold:
            outliers_p.append((rp, best_rt, best_d))

    # Sort by L1 descending
    outliers_t.sort(key=lambda t: -t[2])
    outliers_p.sort(key=lambda t: -t[2])

    print("=" * 72)
    print(f"File:  {args.pickle}")
    print(f"k = {k}, n = {n}, threshold L1 ≥ {args.threshold}")
    print("=" * 72)
    print()
    print(f"Outlier rectangles in TARGET (no near match in pristine): {len(outliers_t)}")
    for rt, rp, dist in outliers_t:
        wxt = rt[1] - rt[0]
        wyt = rt[3] - rt[2]
        ot = "H" if wxt > wyt else ("V" if wyt > wxt else "□")
        print(f"  L1={dist:>4} [{ot}] target=({rt[0]},{rt[1]})×({rt[2]},{rt[3]}) "
              f"w=({wxt},{wyt})")
        print(f"             closest pristine=({rp[0]},{rp[1]})×({rp[2]},{rp[3]}) "
              f"w=({rp[1]-rp[0]},{rp[3]-rp[2]})")

    print()
    print(f"Outlier rectangles in PRISTINE (no near match in target): {len(outliers_p)}")
    for rp, rt, dist in outliers_p:
        wxp = rp[1] - rp[0]
        wyp = rp[3] - rp[2]
        op = "H" if wxp > wyp else ("V" if wyp > wxp else "□")
        print(f"  L1={dist:>4} [{op}] pristine=({rp[0]},{rp[1]})×({rp[2]},{rp[3]}) "
              f"w=({wxp},{wyp})")
        print(f"             closest target =({rt[0]},{rt[1]})×({rt[2]},{rt[3]}) "
              f"w=({rt[1]-rt[0]},{rt[3]-rt[2]})")

    # IS membership of each outlier in target
    if IS_t is not None:
        print()
        print("--- IS context for target outliers ---")
        # IS_t is presumably a list of label indices (1-based or 0-based?)
        # Check format: if max element <= n, it's labels; otherwise it's a bitvector.
        is_set = set(IS_t)
        # Try both 1-based and 0-based by checking range
        rect_to_idx = {to_tuple(r): i for i, r in enumerate(rects_t)}
        # compatibility: also try i+1
        for rt, _, dist in outliers_t:
            idx0 = rect_to_idx.get(rt)
            if idx0 is None:
                continue
            idx1 = idx0 + 1
            in_is = (idx0 in is_set) or (idx1 in is_set)
            # count how many IS members it overlaps
            n_overlaps_with_is = 0
            for j, r2 in enumerate(rects_t):
                if j == idx0:
                    continue
                j1 = j + 1
                if (j in is_set) or (j1 in is_set):
                    if overlaps(rt, to_tuple(r2)):
                        n_overlaps_with_is += 1
            print(f"  outlier ({rt[0]},{rt[1]})×({rt[2]},{rt[3]}): "
                  f"in_IS={in_is}, overlaps {n_overlaps_with_is} IS members")
    else:
        print()
        print("(no IS in pickle — skipping IS-context analysis)")

    # Final framing
    print()
    print("=" * 72)
    print("WHAT THIS MEANS FOR THE GAP")
    print("=" * 72)
    if len(outliers_t) <= 4 and len(outliers_p) <= 4:
        print(f"Found {len(outliers_t)} target-side and {len(outliers_p)} pristine-side")
        print("outliers. The local change is small. Print these out, look at where")
        print("they sit in the k-box grid, and you have your structural description.")
    else:
        print(f"Found {len(outliers_t)} target-side and {len(outliers_p)} pristine-side")
        print("outliers — more than expected for a clean theorem. The +1 might be")
        print("from a less-local modification, or the threshold needs tuning.")


if __name__ == "__main__":
    main()
