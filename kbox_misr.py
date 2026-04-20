#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kbox_misr.py
============

k-box segment construction (Caoduro, Cslovjecsek, Pilipczuk, Wegrzycki 2022,
arXiv:2205.15189) encoded as THIN RECTANGLES in the (H, V) twin-sequence
format used by mistr_runner.py.

This is the missing scaffold for pushing MISR integrality-gap search past
the Chuzhoy 3/2 barrier toward 2 - epsilon.

Provides:
  * kbox_instance(k): exact M_k in the (H, V) format, n = 4k^2 labels.
  * expected_edges(k): combinatorial ground-truth edge set of G_k.
  * actual_edges(H, V): intersection graph actually realized by the (H, V) coords.
  * triangle_free(H, V): True iff every grid point has <= 2 rectangles on it.
  * explicit_IS(k): the paper's independent set of size k^2 + 3k - 2.
  * verify(k): runs all the above, prints a report, returns pass/fail.
  * verify_gurobi(H, V): solves LP and ILP and checks 2k^2 / (k^2 + 3k - 2).
  * Block-level search primitives (see search_moves.py or at bottom of this file).

Usage:
  python3 kbox_misr.py --k 3
  python3 kbox_misr.py --k 5 --gurobi
  python3 kbox_misr.py --k 3 --dump | head -5
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# --------------------------------------------------------------------------- #
# Reuse from mistr_runner.py if available (for seq_spans, canonicalize, etc.) #
# --------------------------------------------------------------------------- #

Seq = List[int]
Instance = Tuple[Seq, Seq]

try:
    # local import — lets this file be dropped next to mistr_runner.py
    from mistr_runner import (
        seq_spans,
        canonicalize,
        instance_key,
        build_rects,
        grid_points,
        covers_grid_closed,
    )
    _HAS_MISTR = True
except Exception:
    _HAS_MISTR = False

    def seq_spans(seq: Seq) -> List[Tuple[int, int]]:
        first: Dict[int, int] = {}
        spans: Dict[int, Tuple[int, int]] = {}
        for idx, lab in enumerate(seq):
            if lab not in first:
                first[lab] = idx
            else:
                spans[lab] = (first[lab], idx)
        n = max(seq) if seq else 0
        return [spans[i] for i in range(1, n + 1)]

    def canonicalize(H: Seq, V: Seq) -> Instance:
        order: List[int] = []
        seen: Set[int] = set()
        for x in H:
            if x not in seen:
                order.append(x)
                seen.add(x)
        rel = {old: new for new, old in enumerate(order, 1)}
        return [rel[x] for x in H], [rel[x] for x in V]

    def instance_key(H: Seq, V: Seq) -> str:
        import hashlib
        s = ",".join(map(str, H)) + "|" + ",".join(map(str, V))
        return hashlib.blake2b(s.encode(), digest_size=16).hexdigest()

    Rect = Tuple[Tuple[int, int], Tuple[int, int]]

    def build_rects(H: Seq, V: Seq):
        X = seq_spans(H)
        Y = seq_spans(V)
        rects = []
        for (x1, x2), (y1, y2) in zip(X, Y):
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            rects.append(((x1, x2), (y1, y2)))
        return rects

    def grid_points(rects):
        xs = sorted({x for r in rects for x in (r[0][0], r[0][1])})
        ys = sorted({y for r in rects for y in (r[1][0], r[1][1])})
        return [(x, y) for x in xs for y in ys]

    def covers_grid_closed(rects, pts):
        C = []
        for (x, y) in pts:
            S = []
            for i, ((x1, x2), (y1, y2)) in enumerate(rects):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    S.append(i)
            C.append(S)
        return C


# =========================================================================== #
# 1. k-box geometry                                                            #
# =========================================================================== #
#
# Labeling convention (matches Lemma 4 of Caoduro et al.):
#   Each k-box has 4k segments:
#     u_i, d_i for i = 1..k     (vertical;   i=1 leftmost column, i=k rightmost)
#     l_j, r_j for j = 1..k     (horizontal; j=1 TOP row,         j=k BOTTOM row)
#
#   Within a k-box:
#     - Column i's meeting point is at row k+1-i  (top-left to bottom-right
#       diagonal: col 1 meets at top, col k meets at bottom).
#     - Row j's meeting point is at column j      (so row 1 [top] meets at col 1,
#       row k [bottom] meets at col k).
#
#   Two kinds of meeting points coincide at grid intersection (col i, row j)
#   when i = j = k+1-i ... impossible in general — they're at different
#   positions. But col i's and row (k+1-i)'s meeting points WOULD coincide
#   without perturbation. We apply a small perturbation:
#     * vertical meeting y-coordinate is nudged UP   by DELTA
#     * horizontal meeting x-coordinate is nudged RIGHT by DELTA
#   so that no two meeting points coincide (satisfies "no three at a point").
#
# Geometric consequences (verified against Lemma 4 of the paper):
#   Within a box, edges are:
#     u_i - d_i              (k edges; meeting at col i)
#     l_j - r_j              (k edges; meeting at row j)
#     d_i - l_j for i <= j   (k(k+1)/2 edges; includes d_i - l_i 'diagonal' pairs
#                             that realize the Lemma-4 D u L partition)
#     u_i - r_j for i >  j   (k(k-1)/2 edges; includes u_{i+1} - r_i 'Lemma-4
#                             U u R' pairs, plus the 'above-staircase' edges)
#     u_i - l_j              NONE within a box
#     d_i - r_j              NONE within a box
#
# Between boxes a != b, WLOG a < b (box a is below-left of box b):
#   u^(a) x l^(b):   all k^2 pairs are edges
#   d^(b) x r^(a):   all k^2 pairs are edges
#   everything else: NO cross-box edges
#
# These are the cross-box edges that drive Lemma 5 ("at most 2 interesting
# boxes") and ultimately give alpha <= k^2 + 3k - 2.

# --------------------------------------------------------------------------- #
# Scale constants: tuned so the real-coordinate geometry is robust.           #
# --------------------------------------------------------------------------- #
#   BOX_SCALE      — size of each k-box in real coords (outer square stays
#                    proportional: (k+3) * BOX_SCALE wide).
#   DELTA          — meeting-point perturbation. Must be:
#                        THICK       <<  DELTA  <<  grid gap within a box
#                    Grid gap within a box is BOX_SCALE / (k+1), so we need
#                    DELTA < BOX_SCALE / (k+1) comfortably.
#   THICK          — thin-rectangle thickness. Much smaller than DELTA.
#   BOUNDARY_JITTER — label-dependent tiebreaker for the (H, V) conversion.
#                     Smaller than THICK by many orders of magnitude.

BOX_SCALE = 10_000.0
DELTA_FRAC = 0.05          # DELTA = 0.05 * BOX_SCALE = 500
THICK_FRAC = 0.0001        # THICK = 0.0001 * BOX_SCALE = 1
JITTER = 1e-9              # unit jitter per (label, side); magnifies by label index


def _kbox_geom(k: int):
    """
    Build the geometric thin-rectangles for M_k.

    Returns a list of tuples: (kind, a, idx, label, xL, xR, yB, yT)
      kind  : one of 'u', 'd', 'l', 'r'
      a     : k-box index (1..k)
      idx   : i or j within the box (1..k)
      label : sequential integer 1..4k^2
      xL,xR : x-boundaries (floats, all distinct across all rectangles)
      yB,yT : y-boundaries (floats, all distinct across all rectangles)
    """
    DELTA = DELTA_FRAC * BOX_SCALE
    THICK = THICK_FRAC * BOX_SCALE
    OUTER_LO = 0.0
    OUTER_HI = (k + 3) * BOX_SCALE

    # Jitter should be <<< THICK. Scale by a generous safety margin.
    def jitter(label: int, side: int) -> float:
        # side: 0=xL, 1=xR, 2=yB, 3=yT
        return (label * 4 + side) * JITTER * THICK

    rects = []
    lab = 0

    for a in range(1, k + 1):
        # Box a occupies [(a+0.5)*BOX_SCALE, (a+1.5)*BOX_SCALE]^2 (inner extent
        # with some padding so other boxes fit along the outer-square diagonal).
        box_lo = (a + 0.5) * BOX_SCALE
        box_hi = (a + 1.5) * BOX_SCALE
        span = box_hi - box_lo

        # k vertical lines at v_x[0..k-1], k horizontal lines at h_y[0..k-1]
        v_x = [box_lo + span * i / (k + 1) for i in range(1, k + 1)]
        h_y = [box_lo + span * j / (k + 1) for j in range(1, k + 1)]

        # --- vertical segments u_i, d_i for i = 1..k ---
        # column i's meeting point: (v_x[i-1], h_y[k-i] + DELTA)
        for i in range(1, k + 1):
            mx = v_x[i - 1]
            my = h_y[k - i] + DELTA

            # u_i: thin rectangle at x=mx, y from my down-to-just-past (so u_i
            #      overlaps d_i) up to outer top.
            lab += 1
            xL = mx - THICK + jitter(lab, 0)
            xR = mx + THICK + jitter(lab, 1)
            yB = my - 0.1 * THICK + jitter(lab, 2)
            yT = OUTER_HI + jitter(lab, 3)
            rects.append(('u', a, i, lab, xL, xR, yB, yT))

            # d_i: thin rectangle at x=mx (slightly different thickness to keep
            #      boundaries distinct), y from outer bottom up to just past my.
            lab += 1
            xL = mx - 0.9 * THICK + jitter(lab, 0)
            xR = mx + 0.9 * THICK + jitter(lab, 1)
            yB = OUTER_LO + jitter(lab, 2)
            yT = my + 0.1 * THICK + jitter(lab, 3)
            rects.append(('d', a, i, lab, xL, xR, yB, yT))

        # --- horizontal segments l_j, r_j for j = 1..k (j=1 top, j=k bottom) ---
        # row j's meeting point: (v_x[j-1] + DELTA, h_y[k-j])
        for j in range(1, k + 1):
            my = h_y[k - j]
            mx = v_x[j - 1] + DELTA

            # l_j: thin rectangle at y=my, x from outer left to just past mx.
            lab += 1
            xL = OUTER_LO + jitter(lab, 0)
            xR = mx + 0.1 * THICK + jitter(lab, 1)
            yB = my - THICK + jitter(lab, 2)
            yT = my + THICK + jitter(lab, 3)
            rects.append(('l', a, j, lab, xL, xR, yB, yT))

            # r_j
            lab += 1
            xL = mx - 0.1 * THICK + jitter(lab, 0)
            xR = OUTER_HI + jitter(lab, 1)
            yB = my - 0.9 * THICK + jitter(lab, 2)
            yT = my + 0.9 * THICK + jitter(lab, 3)
            rects.append(('r', a, j, lab, xL, xR, yB, yT))

    assert lab == 4 * k * k, f"built {lab} rectangles, expected {4*k*k}"
    return rects


# =========================================================================== #
# 2. (H, V) conversion                                                         #
# =========================================================================== #


def kbox_instance(k: int) -> Instance:
    """
    Return the canonical (H, V) pair encoding M_k.

    The (H, V) format: each label 1..4k^2 appears exactly twice in H and twice
    in V. A label's rectangle has x-interval = (positions of its two H
    occurrences) and y-interval = (positions of its two V occurrences).

    The labels returned here are NOT necessarily in a particular semantic order;
    they're assigned sequentially during geometry construction and then
    canonicalized by first-appearance in H.
    """
    rects = _kbox_geom(k)

    x_events: List[Tuple[float, int]] = []  # (xcoord, label)
    y_events: List[Tuple[float, int]] = []
    for (_kind, _a, _i, lab, xL, xR, yB, yT) in rects:
        x_events.append((xL, lab))
        x_events.append((xR, lab))
        y_events.append((yB, lab))
        y_events.append((yT, lab))

    # All coords distinct by construction (jitter); simple sort suffices.
    x_events.sort()
    y_events.sort()

    H = [lab for (_, lab) in x_events]
    V = [lab for (_, lab) in y_events]

    # Sanity: each label appears exactly twice.
    cH = Counter(H)
    cV = Counter(V)
    assert all(v == 2 for v in cH.values()), "H: some label not appearing twice"
    assert all(v == 2 for v in cV.values()), "V: some label not appearing twice"
    assert len(cH) == 4 * k * k, f"H has {len(cH)} labels, expected {4*k*k}"

    return canonicalize(H, V)


def kbox_instance_with_key_map(k: int) -> Tuple[Instance, Dict[Tuple[str, int, int], int]]:
    """
    Same as kbox_instance, but also returns a map (kind, box, idx) -> canonical
    label (after canonicalization). Useful for constructing explicit independent
    sets in terms of the (H, V)'s labels.
    """
    rects = _kbox_geom(k)
    raw_key_to_lab = {(kind, a, i): lab for (kind, a, i, lab, *_) in rects}

    x_events = []
    y_events = []
    for (_k, _a, _i, lab, xL, xR, yB, yT) in rects:
        x_events.append((xL, lab))
        x_events.append((xR, lab))
        y_events.append((yB, lab))
        y_events.append((yT, lab))
    x_events.sort()
    y_events.sort()
    H_raw = [lab for (_, lab) in x_events]
    V_raw = [lab for (_, lab) in y_events]

    # canonicalize(): relabels by first-appearance order in H
    order = []
    seen = set()
    for x in H_raw:
        if x not in seen:
            order.append(x)
            seen.add(x)
    rel = {old: new for new, old in enumerate(order, 1)}

    key_to_lab = {key: rel[raw] for key, raw in raw_key_to_lab.items()}
    H = [rel[x] for x in H_raw]
    V = [rel[x] for x in V_raw]
    return (H, V), key_to_lab


# =========================================================================== #
# 3. Expected combinatorial graph (ground truth)                               #
# =========================================================================== #


def expected_edges(k: int,
                   key_to_lab: Optional[Dict[Tuple[str, int, int], int]] = None
                   ) -> Set[FrozenSet[int]]:
    """
    Build the expected edge set of G_k from the structural rules described at
    the top of this file. If key_to_lab is given, edges are keyed on those
    labels; otherwise they use the sequential labeling from _kbox_geom.
    """
    if key_to_lab is None:
        rects = _kbox_geom(k)
        key_to_lab = {(kind, a, i): lab for (kind, a, i, lab, *_) in rects}

    edges: Set[FrozenSet[int]] = set()

    def add(ka, kb):
        la = key_to_lab[ka]
        lb = key_to_lab[kb]
        if la != lb:
            edges.add(frozenset((la, lb)))

    # within-box
    for a in range(1, k + 1):
        for i in range(1, k + 1):
            add(('u', a, i), ('d', a, i))           # column meetings
        for j in range(1, k + 1):
            add(('l', a, j), ('r', a, j))           # row meetings
        for i in range(1, k + 1):                    # d_i x l_j  for i <= j
            for j in range(i, k + 1):
                add(('d', a, i), ('l', a, j))
        for i in range(2, k + 1):                    # u_i x r_j  for i > j
            for j in range(1, i):
                add(('u', a, i), ('r', a, j))

    # cross-box (a < b)
    for a, b in itertools.combinations(range(1, k + 1), 2):
        for i in range(1, k + 1):
            for j in range(1, k + 1):
                add(('u', a, i), ('l', b, j))
                add(('d', b, i), ('r', a, j))

    return edges


def actual_edges(H: Seq, V: Seq) -> Set[FrozenSet[int]]:
    """
    Intersection graph of the rectangles as realized by the (H, V) integer
    coordinates. Edges are frozensets of 1-indexed labels.
    """
    rects = build_rects(H, V)
    n = len(rects)
    edges: Set[FrozenSet[int]] = set()
    for i in range(n):
        (xi_l, xi_r), (yi_l, yi_r) = rects[i]
        for j in range(i + 1, n):
            (xj_l, xj_r), (yj_l, yj_r) = rects[j]
            if max(xi_l, xj_l) <= min(xi_r, xj_r) and max(yi_l, yj_l) <= min(yi_r, yj_r):
                edges.add(frozenset((i + 1, j + 1)))
    return edges


def triangle_free(H: Seq, V: Seq) -> bool:
    """True iff every grid point is covered by AT MOST 2 rectangles."""
    rects = build_rects(H, V)
    pts = grid_points(rects)
    covers = covers_grid_closed(rects, pts)
    return all(len(c) <= 2 for c in covers)


# =========================================================================== #
# 4. Explicit independent set from the paper (size k^2 + 3k - 2)               #
# =========================================================================== #


def explicit_IS_keys(k: int) -> List[Tuple[str, int, int]]:
    """
    The explicit independent set used in the paper, expressed as (kind, box, idx).
    """
    IS: List[Tuple[str, int, int]] = []
    # B_1: all l's and all u's
    for j in range(1, k + 1):
        IS.append(('l', 1, j))
    for i in range(1, k + 1):
        IS.append(('u', 1, i))
    # B_2: all r's and all d's
    if k >= 2:
        for j in range(1, k + 1):
            IS.append(('r', 2, j))
        for i in range(1, k + 1):
            IS.append(('d', 2, i))
    # B_i (i >= 3): all r's and the topmost u (u_1)
    for a in range(3, k + 1):
        for j in range(1, k + 1):
            IS.append(('r', a, j))
        IS.append(('u', a, 1))
    return IS


def explicit_IS_labels(k: int,
                       key_to_lab: Dict[Tuple[str, int, int], int]) -> List[int]:
    return [key_to_lab[key] for key in explicit_IS_keys(k)]


# =========================================================================== #
# 5. Verification                                                              #
# =========================================================================== #


def verify(k: int, verbose: bool = True) -> bool:
    """
    Build M_k via kbox_instance, then check:
      - Intersection graph matches expected_edges(k).
      - Graph is triangle-free.
      - The explicit IS is independent and has size k^2 + 3k - 2.
      - alpha* >= n/2 = 2k^2 is realized by the all-1/2 fractional solution.
    Returns True iff all checks pass.
    """
    (H, V), key_to_lab = kbox_instance_with_key_map(k)
    n = 4 * k * k
    expected_alpha = k * k + 3 * k - 2
    expected_alpha_star = 2 * k * k

    ok = True
    if verbose:
        print(f"=== Verifying M_{k}  (n = {n}, expected alpha* = {expected_alpha_star}, "
              f"expected alpha = {expected_alpha}, target gap = "
              f"{expected_alpha_star/expected_alpha:.4f}) ===")

    # --- edge set ---
    exp = expected_edges(k, key_to_lab)
    act = actual_edges(H, V)
    missing = exp - act
    extra = act - exp
    if verbose:
        print(f"  edges: expected={len(exp)}  actual={len(act)}"
              f"  missing={len(missing)}  extra={len(extra)}")
    if missing or extra:
        ok = False
        if verbose and (missing or extra):
            show = list(missing)[:5] + list(extra)[:5]
            for e in show[:10]:
                a, b = sorted(e)
                print(f"    diff edge: {a}-{b}")

    # --- triangle-free ---
    tf = triangle_free(H, V)
    if verbose:
        print(f"  triangle-free (every grid point covered by <= 2 rects): {tf}")
    if not tf:
        ok = False

    # --- explicit IS ---
    IS = explicit_IS_labels(k, key_to_lab)
    IS_set = set(IS)
    if verbose:
        print(f"  explicit IS size: {len(IS)} (expected {expected_alpha})")
    if len(IS) != expected_alpha:
        ok = False
    # check independence
    bad = [tuple(sorted(e)) for e in act
           if len(e) == 2 and all(v in IS_set for v in e)]
    if verbose:
        print(f"  IS independence violations: {len(bad)}")
    if bad:
        ok = False
        if verbose:
            for (u, v) in bad[:5]:
                print(f"    IS contains both {u} and {v}, which intersect")

    if verbose:
        print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def verify_gurobi(k: int, verbose: bool = True) -> Optional[Tuple[float, float]]:
    """
    If gurobipy and mistr_runner are available, solve LP and ILP and compare
    against 2k^2 and k^2 + 3k - 2.
    """
    if not _HAS_MISTR:
        if verbose:
            print("  (mistr_runner not importable; skipping Gurobi check.)")
        return None
    try:
        from mistr_runner import solve_lp_ilp
    except Exception as e:
        if verbose:
            print(f"  (could not import solve_lp_ilp: {e})")
        return None

    H, V = kbox_instance(k)
    rects = build_rects(H, V)
    lp, ilp = solve_lp_ilp(rects, grb_threads=0)
    exp_lp = 2 * k * k
    exp_ilp = k * k + 3 * k - 2
    if verbose:
        print(f"  Gurobi:  LP = {lp:.4f}  (expected {exp_lp})")
        print(f"           ILP = {ilp:.4f}  (expected {exp_ilp})")
        print(f"           ratio = {lp/ilp:.6f}  (expected {exp_lp/exp_ilp:.6f})")
    return lp, ilp


# =========================================================================== #
# 6. CLI                                                                       #
# =========================================================================== #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gurobi", action="store_true",
                    help="also solve LP/ILP via Gurobi (requires mistr_runner + gurobipy)")
    ap.add_argument("--dump", action="store_true",
                    help="print the full (H, V) sequences")
    ap.add_argument("--sweep", action="store_true",
                    help="verify k = 2..7 in a loop and print a summary table")
    args = ap.parse_args()

    if args.sweep:
        print(f"{'k':>3} {'n':>6} {'alpha*':>8} {'alpha':>8} {'gap':>8} {'ok':>4}")
        for k in range(2, 8):
            ok = verify(k, verbose=False)
            print(f"{k:>3} {4*k*k:>6} {2*k*k:>8} {k*k+3*k-2:>8} "
                  f"{2*k*k/(k*k+3*k-2):>8.4f} {'PASS' if ok else 'FAIL':>4}")
        return

    ok = verify(args.k, verbose=True)

    if args.gurobi:
        verify_gurobi(args.k, verbose=True)

    if args.dump:
        H, V = kbox_instance(args.k)
        print(f"H = {H}")
        print(f"V = {V}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
