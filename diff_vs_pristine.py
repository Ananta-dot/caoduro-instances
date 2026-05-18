#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diff_vs_pristine.py
===================

Compare any saved MISR pickle against pristine M_k. Tells you exactly which
labels' positions differ — the structural modification that produced the
improvement over Caoduro.

Three angles of comparison:

  1. Position diff: for each label, do its H positions or V positions
     differ from pristine? Lists exactly which labels moved.

  2. Rectangle geometry diff: for each label, what is its (x1, x2, y1, y2)
     in your instance vs in pristine? Highlights segments that got
     shrunk, extended, or shifted.

  3. Intersection graph diff: which edges exist in your instance but not
     in pristine, and vice versa. Tells you what new intersections were
     created (and any that disappeared).

Usage:
  python3 diff_vs_pristine.py elites_above_threshold/k9_r0_gap1.5577_tf1.pkl
  python3 diff_vs_pristine.py <pickle> --verbose   (full per-rect dump)
"""

from __future__ import annotations

import argparse
import pickle
import sys
from typing import Dict, List, Set, Tuple

from kbox_misr import kbox_instance, build_rects, canonicalize


def position_diff(H_a: List[int], V_a: List[int],
                  H_b: List[int], V_b: List[int]) -> Dict:
    """For each label, compare its 2 positions in H and V across two instances."""
    n_a = max(max(H_a), max(V_a))
    n_b = max(max(H_b), max(V_b))
    if n_a != n_b:
        return {"error": f"n mismatch: {n_a} vs {n_b}"}

    def positions(seq, lab):
        return sorted([i for i, x in enumerate(seq) if x == lab])

    moved_in_h = []
    moved_in_v = []
    for lab in range(1, n_a + 1):
        pH_a = positions(H_a, lab)
        pH_b = positions(H_b, lab)
        pV_a = positions(V_a, lab)
        pV_b = positions(V_b, lab)
        if pH_a != pH_b:
            moved_in_h.append((lab, tuple(pH_a), tuple(pH_b)))
        if pV_a != pV_b:
            moved_in_v.append((lab, tuple(pV_a), tuple(pV_b)))
    return {
        "n_moved_in_h": len(moved_in_h),
        "n_moved_in_v": len(moved_in_v),
        "moved_in_h": moved_in_h,
        "moved_in_v": moved_in_v,
    }


def geometry_diff(H_a, V_a, H_b, V_b):
    """For each label, compare its (x1, x2, y1, y2) across two instances."""
    rects_a = build_rects(H_a, V_a)
    rects_b = build_rects(H_b, V_b)
    n = len(rects_a)
    diffs = []
    for i in range(n):
        ((xa1, xa2), (ya1, ya2)) = rects_a[i]
        ((xb1, xb2), (yb1, yb2)) = rects_b[i]
        if (xa1, xa2, ya1, ya2) != (xb1, xb2, yb1, yb2):
            diffs.append({
                "label": i + 1,
                "x_a": (xa1, xa2), "x_b": (xb1, xb2),
                "y_a": (ya1, ya2), "y_b": (yb1, yb2),
                "x_width_a": xa2 - xa1, "x_width_b": xb2 - xb1,
                "y_width_a": ya2 - ya1, "y_width_b": yb2 - yb1,
            })
    return diffs


def edges_of(H, V) -> Set[Tuple[int, int]]:
    """Return set of {(i,j): i<j} for intersecting rectangle indices."""
    rects = build_rects(H, V)
    n = len(rects)
    edges = set()
    for i in range(n):
        (xi1, xi2), (yi1, yi2) = rects[i]
        for j in range(i + 1, n):
            (xj1, xj2), (yj1, yj2) = rects[j]
            if (max(xi1, xj1) <= min(xi2, xj2)
                    and max(yi1, yj1) <= min(yi2, yj2)):
                edges.add((i, j))
    return edges


def edge_diff(H_a, V_a, H_b, V_b):
    """Return (only_a, only_b, common) edge counts and the asymmetric edge lists."""
    E_a = edges_of(H_a, V_a)
    E_b = edges_of(H_b, V_b)
    return {
        "n_edges_a": len(E_a),
        "n_edges_b": len(E_b),
        "n_common": len(E_a & E_b),
        "n_only_a": len(E_a - E_b),
        "n_only_b": len(E_b - E_a),
        "only_a": E_a - E_b,
        "only_b": E_b - E_a,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle", help="the saved pickle to compare against pristine")
    ap.add_argument("--verbose", action="store_true",
                    help="print every per-label diff (large output)")
    ap.add_argument("--max-show", type=int, default=15,
                    help="how many labels/edges to show in non-verbose mode")
    args = ap.parse_args()

    # ---- load the saved pickle ----
    with open(args.pickle, "rb") as f:
        d = pickle.load(f)
    H_target = d.get("H") or d.get("best_instance_H")
    V_target = d.get("V") or d.get("best_instance_V")
    if H_target is None or V_target is None:
        print("ERROR: pickle missing H/V keys.", file=sys.stderr)
        sys.exit(1)
    k = d.get("k")
    if k is None:
        # infer from n = 4k^2
        n = max(max(H_target), max(V_target))
        k = int(round((n / 4) ** 0.5))
        print(f"[info] inferred k={k} from n={n}", file=sys.stderr)
    n = 4 * k * k

    H_pristine, V_pristine = kbox_instance(k)

    # canonicalize both for a fair comparison
    # NOTE: canonicalize relabels by first-appearance in H, so the same
    # graph under different labelings will canonicalize to the same (H, V).
    H_a, V_a = canonicalize(H_target, V_target)
    H_b, V_b = canonicalize(H_pristine, V_pristine)

    print("=" * 70)
    print(f"File:           {args.pickle}")
    print(f"k:              {k} (n = {n})")
    print(f"reported gap:   {d.get('gap', '?')}")
    print(f"reported tf:    {d.get('triangle_free', '?')}")
    print(f"reported mc:    {d.get('max_clique', '?')}")
    print("=" * 70)

    # ---- 1. Position diff ----
    print("\n[1] LABEL POSITION DIFF (after canonicalization)")
    pd = position_diff(H_a, V_a, H_b, V_b)
    if "error" in pd:
        print(f"  ERROR: {pd['error']}")
        return
    print(f"  Labels with different H positions: {pd['n_moved_in_h']} / {n}")
    print(f"  Labels with different V positions: {pd['n_moved_in_v']} / {n}")

    if pd['n_moved_in_h'] == 0 and pd['n_moved_in_v'] == 0:
        print("  Both sequences are identical after canonicalization.")
        print("  → Your instance IS pristine M_k (just possibly relabeled).")
        print("    The reported gap improvement either came from a metric")
        print("    difference or was an artifact. Worth re-verifying.")
        return

    if args.verbose:
        if pd['moved_in_h']:
            print("\n  H position diffs (label → was → now):")
            for lab, was, now in pd['moved_in_h']:
                print(f"    {lab:>4}: {was} → {now}")
        if pd['moved_in_v']:
            print("\n  V position diffs (label → was → now):")
            for lab, was, now in pd['moved_in_v']:
                print(f"    {lab:>4}: {was} → {now}")
    else:
        if pd['moved_in_h']:
            shown = pd['moved_in_h'][:args.max_show]
            print(f"\n  First {len(shown)} H position diffs:")
            for lab, was, now in shown:
                print(f"    label {lab}: H positions {was} → {now}")
            if len(pd['moved_in_h']) > args.max_show:
                print(f"    ... and {len(pd['moved_in_h']) - args.max_show} more")
        if pd['moved_in_v']:
            shown = pd['moved_in_v'][:args.max_show]
            print(f"\n  First {len(shown)} V position diffs:")
            for lab, was, now in shown:
                print(f"    label {lab}: V positions {was} → {now}")
            if len(pd['moved_in_v']) > args.max_show:
                print(f"    ... and {len(pd['moved_in_v']) - args.max_show} more")

    # ---- 2. Geometry diff ----
    print("\n[2] RECTANGLE GEOMETRY DIFF")
    geo = geometry_diff(H_a, V_a, H_b, V_b)
    print(f"  Labels with different (x1, x2, y1, y2): {len(geo)} / {n}")

    if geo:
        shown = geo if args.verbose else geo[:args.max_show]
        print(f"\n  {'label':>5}  {'x_target':>10}  {'x_pristine':>10}  "
              f"{'y_target':>10}  {'y_pristine':>10}  {'changed':>9}")
        print("  " + "-" * 65)
        for g in shown:
            changes = []
            if g['x_a'] != g['x_b']: changes.append("x")
            if g['y_a'] != g['y_b']: changes.append("y")
            if g['x_width_a'] != g['x_width_b']: changes.append("wx")
            if g['y_width_a'] != g['y_width_b']: changes.append("wy")
            ch_str = ",".join(changes) if changes else "-"
            print(f"  {g['label']:>5}  "
                  f"{str(g['x_a']):>10}  {str(g['x_b']):>10}  "
                  f"{str(g['y_a']):>10}  {str(g['y_b']):>10}  "
                  f"{ch_str:>9}")
        if not args.verbose and len(geo) > args.max_show:
            print(f"  ... and {len(geo) - args.max_show} more")

    # Width-change summary
    shrunk_x = sum(1 for g in geo if g['x_width_a'] < g['x_width_b'])
    extended_x = sum(1 for g in geo if g['x_width_a'] > g['x_width_b'])
    shrunk_y = sum(1 for g in geo if g['y_width_a'] < g['y_width_b'])
    extended_y = sum(1 for g in geo if g['y_width_a'] > g['y_width_b'])
    print(f"\n  Width changes: x_shrunk={shrunk_x}, x_extended={extended_x}, "
          f"y_shrunk={shrunk_y}, y_extended={extended_y}")

    # ---- 3. Edge diff ----
    print("\n[3] INTERSECTION GRAPH EDGE DIFF")
    ed = edge_diff(H_a, V_a, H_b, V_b)
    print(f"  Edges in your instance: {ed['n_edges_a']}")
    print(f"  Edges in pristine M_{k}: {ed['n_edges_b']}")
    print(f"  Common: {ed['n_common']}")
    print(f"  Only in yours (new):    {ed['n_only_a']}")
    print(f"  Only in pristine (lost): {ed['n_only_b']}")
    print(f"  Symmetric difference:    {ed['n_only_a'] + ed['n_only_b']}")
    if ed['n_only_a'] > 0 and (args.verbose or ed['n_only_a'] <= args.max_show):
        print(f"\n  New edges (label_i, label_j) in your instance:")
        for (i, j) in sorted(ed['only_a'])[:args.max_show if not args.verbose else 9999]:
            print(f"    ({i+1}, {j+1})")
        if not args.verbose and ed['n_only_a'] > args.max_show:
            print(f"    ... and {ed['n_only_a'] - args.max_show} more")
    if ed['n_only_b'] > 0 and (args.verbose or ed['n_only_b'] <= args.max_show):
        print(f"\n  Lost edges (label_i, label_j) from pristine:")
        for (i, j) in sorted(ed['only_b'])[:args.max_show if not args.verbose else 9999]:
            print(f"    ({i+1}, {j+1})")
        if not args.verbose and ed['n_only_b'] > args.max_show:
            print(f"    ... and {ed['n_only_b'] - args.max_show} more")

    # ---- summary verdict ----
    print("\n" + "=" * 70)
    print("STRUCTURAL SUMMARY")
    print("=" * 70)
    n_moved = max(pd['n_moved_in_h'], pd['n_moved_in_v'])
    edge_sym = ed['n_only_a'] + ed['n_only_b']
    if n_moved <= 4 and edge_sym <= 20:
        print(f"  Small local modification: ~{n_moved} labels moved,")
        print(f"  ~{edge_sym} edges differ. This is a tractable structural")
        print(f"  change you can describe in a few sentences.")
    elif n_moved <= 20 and edge_sym <= 100:
        print(f"  Moderate modification: {n_moved} labels moved,")
        print(f"  {edge_sym} edges differ. Local search drifted significantly")
        print(f"  but the structure is still close to pristine.")
    else:
        print(f"  Large modification: {n_moved} labels moved,")
        print(f"  {edge_sym} edges differ. The instance is structurally")
        print(f"  far from pristine M_{k}. Hard to describe combinatorially.")


if __name__ == "__main__":
    main()
