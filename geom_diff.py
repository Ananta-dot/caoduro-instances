#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geom_diff.py
============

Geometry-only diff between a saved MISR pickle and pristine M_k.

Compares rectangles as a *multiset* of (x1, x2, y1, y2) tuples. Labels are
ignored entirely — two rectangles match iff they occupy the exact same
extent. This avoids the canonicalize() relabeling artifact in
diff_vs_pristine.py that makes a 1-rectangle change look like a 291-label
change.

Output:
    - how many rectangles are *identical* to pristine
    - the rectangles that are in target but not pristine (modifications)
    - the rectangles that are in pristine but not target (originals replaced)
    - nearest-neighbor pairing by L1 distance on (x1, x2, y1, y2),
      so each modified rectangle is shown next to its closest pristine match

Usage:
    python3 geom_diff.py elites_above_threshold/k9_r0_gap1.5577_tf1.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter

from kbox_misr import kbox_instance, build_rects


def rect_tuples(H, V):
    """Build rectangles and return them as (x1, x2, y1, y2) tuples."""
    return [(x1, x2, y1, y2) for ((x1, x2), (y1, y2)) in build_rects(H, V)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle", help="saved (H, V) pickle")
    ap.add_argument("--show", type=int, default=30,
                    help="how many modified rects to print (default 30)")
    args = ap.parse_args()

    with open(args.pickle, "rb") as f:
        d = pickle.load(f)
    H_t = d.get("H") or d.get("best_instance_H")
    V_t = d.get("V") or d.get("best_instance_V")
    if H_t is None or V_t is None:
        print("ERROR: pickle missing H/V keys", file=sys.stderr)
        sys.exit(1)
    k = d.get("k")
    if k is None:
        n = max(max(H_t), max(V_t))
        k = int(round((n / 4) ** 0.5))
    n = 4 * k * k

    H_p, V_p = kbox_instance(k)
    R_t = rect_tuples(H_t, V_t)
    R_p = rect_tuples(H_p, V_p)

    C_t = Counter(R_t)
    C_p = Counter(R_p)

    common = C_t & C_p
    only_t = C_t - C_p
    only_p = C_p - C_t

    n_common = sum(common.values())
    n_only_t = sum(only_t.values())
    n_only_p = sum(only_p.values())

    print("=" * 70)
    print(f"File:         {args.pickle}")
    print(f"k = {k}, n = {n}")
    print(f"reported gap: {d.get('gap', '?')}")
    print(f"reported tf:  {d.get('triangle_free', '?')}")
    print(f"reported mc:  {d.get('max_clique', '?')}")
    print("=" * 70)
    print()
    print(f"Total rectangles:                  {n}")
    print(f"Identical to pristine:             {n_common} / {n}")
    print(f"Modified (in target, not pristine): {n_only_t}")
    print(f"Replaced (in pristine, not target): {n_only_p}")
    print()

    if n_only_t == 0 and n_only_p == 0:
        print("Rectangle multisets are IDENTICAL.")
        print("If the IS still differs, the difference is in which IS the ILP")
        print("found, not in the rectangle set itself.")
        return

    # Limit listing if huge
    show = args.show

    print(f"--- Modified rectangles (up to {show}) ---")
    for r, c in sorted(only_t.items())[:show]:
        x1, x2, y1, y2 = r
        wx, wy = x2 - x1, y2 - y1
        orient = "H" if wx > wy else ("V" if wy > wx else "□")
        print(f"  [{orient}] x=({x1:>4},{x2:>4})  y=({y1:>4},{y2:>4})"
              f"  w=({wx},{wy})  ×{c}")

    print()
    print(f"--- Replaced rectangles (up to {show}) ---")
    for r, c in sorted(only_p.items())[:show]:
        x1, x2, y1, y2 = r
        wx, wy = x2 - x1, y2 - y1
        orient = "H" if wx > wy else ("V" if wy > wx else "□")
        print(f"  [{orient}] x=({x1:>4},{x2:>4})  y=({y1:>4},{y2:>4})"
              f"  w=({wx},{wy})  ×{c}")

    # Nearest-neighbor pairing: for each modified rect, find closest pristine.
    print()
    print(f"--- Nearest-neighbor pairing (target ↔ pristine, L1 on extents) ---")
    list_t = list(only_t.elements())
    list_p_remaining = list(only_p.elements())
    pairs = []
    for rt in list_t:
        if not list_p_remaining:
            break
        best_idx, best_d = -1, float("inf")
        for i, rp in enumerate(list_p_remaining):
            d_ = (abs(rp[0] - rt[0]) + abs(rp[1] - rt[1])
                  + abs(rp[2] - rt[2]) + abs(rp[3] - rt[3]))
            if d_ < best_d:
                best_d, best_idx = d_, i
        rp = list_p_remaining.pop(best_idx)
        pairs.append((rt, rp, best_d))

    pairs.sort(key=lambda t: t[2])
    for rt, rp, dist in pairs[:show]:
        dx1 = rt[0] - rp[0]
        dx2 = rt[1] - rp[1]
        dy1 = rt[2] - rp[2]
        dy2 = rt[3] - rp[3]
        deltas = []
        if dx1: deltas.append(f"Δx1={dx1:+d}")
        if dx2: deltas.append(f"Δx2={dx2:+d}")
        if dy1: deltas.append(f"Δy1={dy1:+d}")
        if dy2: deltas.append(f"Δy2={dy2:+d}")
        delta_str = " ".join(deltas) if deltas else "(none?)"
        print(f"  L1={dist:>3}  pristine=({rp[0]},{rp[1]})×({rp[2]},{rp[3]})"
              f"  →  target=({rt[0]},{rt[1]})×({rt[2]},{rt[3]})  [{delta_str}]")

    if len(pairs) > show:
        print(f"  ... and {len(pairs) - show} more pairs")

    # Distance histogram
    print()
    print("--- L1-distance histogram of modified rectangles vs pristine ---")
    from collections import Counter as _C
    hist = _C(d for _, _, d in pairs)
    for dist in sorted(hist):
        print(f"  L1={dist:>3}: {hist[dist]} rectangles")

    # Verdict
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    if n_only_t <= 4:
        print(f"VERY LOCAL change: only {n_only_t} rectangles differ from pristine.")
        print("This is theorem-shaped. Look at the modified rects above and write")
        print("the change down in one sentence.")
    elif n_only_t <= 16:
        print(f"LOCAL change: {n_only_t} rectangles differ. Likely affects one")
        print(f"or two k-boxes. Compute which boxes the modified rects belong to")
        print(f"and you have a structural description.")
    elif n_only_t <= n // 4:
        print(f"MODERATE change: {n_only_t}/{n} rectangles differ.")
        print(f"Search drifted noticeably. Still compact enough to characterize")
        print(f"if the modifications cluster in a few k-boxes.")
    else:
        print(f"LARGE change: {n_only_t}/{n} rectangles differ.")
        print(f"Instance is structurally far from pristine. Either local search")
        print(f"found a genuinely different family, or the modifications are")
        print(f"distributed and not theorem-shaped.")


if __name__ == "__main__":
    main()
