#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lift_elite.py — extend a verified k-pickle to a larger k as a seed for search.

The idea: if we have a verified instance at k=9 with gap > Caoduro's 1.5283
(meaning our modification somewhere in the 9 k-boxes produced a smaller alpha),
we try to preserve that modification by *embedding* the k=9 instance into a
k=10 layout. We do this by appending one extra k-box's worth of rectangles
(the 10th box from pristine M_10), placed geometrically past the end of the
k=9 structure.

If the k=9 modification contributed -1 or -2 to alpha at n=324, and the
appended 10th box contributes its usual Caoduro count (+18 to IS, +36 rects),
the combined k=10 instance might have alpha around 106+18-1 = 123 instead of
Caoduro's 128. That would give gap 200/123 = 1.6260, a +5 improvement.

This is speculative — the geometric layout of the appended box may not
interact cleanly with the extended k=9 rectangles, so Gurobi may still
find a higher alpha. But as a seed for local search, it's strictly better
than starting from pristine M_10.

Usage:
  python3 lift_elite.py elites_above_threshold/k9_r0_gap1.5577_tf1.pkl \
      --to-k 10 --out lifted_seeds/k10_lifted_from_k9.pkl

Then feed the output pickle directly to kbox_search by placing it in
the elites_above_threshold directory (renamed to have the correct k
in the filename so elite-reuse picks it up).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import List, Tuple

from kbox_misr import kbox_instance, build_rects, canonicalize
from kbox_misr import Instance, Seq


def lift_instance(H_src: Seq, V_src: Seq,
                  k_src: int, k_dst: int) -> Instance:
    """
    Take an instance at n_src = 4*k_src^2 and extend it to an instance at
    n_dst = 4*k_dst^2 by appending rectangles from the pristine M_{k_dst}'s
    last (k_dst - k_src) boxes, placed to the right and below the existing
    structure.

    The appended rectangles are re-labeled to avoid collision with source
    labels, and the resulting sequences are canonicalized.
    """
    if k_dst <= k_src:
        raise ValueError(f"k_dst={k_dst} must be > k_src={k_src}")

    n_src = 4 * k_src * k_src
    n_dst = 4 * k_dst * k_dst

    # Sanity check source
    assert max(max(H_src), max(V_src)) == n_src, \
        f"source instance has max label {max(max(H_src), max(V_src))}, expected {n_src}"

    # Build pristine M_{k_dst}
    H_dst_pristine, V_dst_pristine = kbox_instance(k_dst)
    rects_dst_pristine = build_rects(H_dst_pristine, V_dst_pristine)

    # We need to figure out which rectangles in pristine M_{k_dst} correspond
    # to boxes k_src+1, k_src+2, ..., k_dst. We'll use their geometry: a
    # rectangle is "in the tail boxes" if its coordinates are all >= some
    # threshold determined by M_{k_src}'s extent.
    #
    # Simpler approach: build pristine M_{k_src} separately, find its extent,
    # then take rectangles from M_{k_dst} whose centroids lie outside that extent.
    H_src_pristine, V_src_pristine = kbox_instance(k_src)
    rects_src_pristine = build_rects(H_src_pristine, V_src_pristine)
    max_x_src = max(r[0][1] for r in rects_src_pristine)
    max_y_src = max(r[1][1] for r in rects_src_pristine)

    # Pick out rectangles in M_{k_dst} that are strictly outside M_{k_src}'s extent
    tail_rect_indices = []
    for i, r in enumerate(rects_dst_pristine):
        (x1, x2), (y1, y2) = r
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if cx > max_x_src and cy > max_y_src:
            tail_rect_indices.append(i)

    n_tail_expected = 4 * (k_dst - k_src) * (k_src + k_dst)  # heuristic
    # Actually for Caoduro: n_dst - n_src = 4*(k_dst^2 - k_src^2)
    n_tail_expected = n_dst - n_src

    if len(tail_rect_indices) < n_tail_expected:
        # Fall back: use all rectangles past the src extent, relaxed threshold
        tail_rect_indices = []
        for i, r in enumerate(rects_dst_pristine):
            (x1, x2), (y1, y2) = r
            if x2 > max_x_src or y2 > max_y_src:
                tail_rect_indices.append(i)

    # Take exactly n_tail_expected of them (may be slightly imperfect; the
    # embedding doesn't need to be Caoduro-structurally-identical, just
    # geometrically disjoint from the source).
    tail_rect_indices = tail_rect_indices[:n_tail_expected]
    if len(tail_rect_indices) != n_tail_expected:
        # Degenerate fallback: just use the last (n_dst - n_src) rectangles
        # from pristine M_{k_dst}, arbitrarily offset so they don't collide.
        print(f"[warn] could not find {n_tail_expected} tail rects via geometry; "
              f"using last {n_tail_expected} labels of pristine M_{k_dst} "
              f"instead", file=sys.stderr)
        tail_rect_indices = list(range(len(rects_dst_pristine) - n_tail_expected,
                                       len(rects_dst_pristine)))

    # Now construct the lifted (H, V). The source labels are 1..n_src. The
    # new labels will be n_src+1..n_dst, mapped to the tail rectangles. We
    # need to insert H and V positions for these new labels.
    #
    # Simplest correct approach: rebuild H, V from scratch in 2n_dst positions.
    # The source rectangles keep their original x/y intervals (shifted so that
    # their positions are within [0, 2*n_src)). The tail rectangles get placed
    # strictly after, at positions [2*n_src, 2*n_dst).

    # Source: find each label's two H positions and two V positions.
    def positions_of(seq, lab):
        return [i for i, x in enumerate(seq) if x == lab]

    H_out = list(H_src)  # copy; we'll extend
    V_out = list(V_src)

    # For each tail rectangle, we need to give it two H positions and two V
    # positions, all strictly past the existing ones (so the appended rects
    # are geometrically separated from the source).
    base_H = len(H_out)  # = 2 * n_src
    base_V = len(V_out)

    # Get the tail rects' relative H/V positions inside pristine M_{k_dst}
    # and offset them.
    tail_H_pos = {}  # label_in_src -> [pos1, pos2]
    tail_V_pos = {}
    dst_labels_used = set()
    for tail_idx, r_idx in enumerate(tail_rect_indices):
        # rectangle at index r_idx in pristine M_{k_dst} uses some label L.
        # We need to find that label's two positions in H_dst_pristine.
        label_dst = None
        cnt_h = 0
        for pos, lab in enumerate(H_dst_pristine):
            if lab not in dst_labels_used:
                # haven't picked this one yet; check if its rectangle matches our index
                pass
        # Direct approach: label = index+1 in pristine construction?
        # Can't assume that. Use the geometry: find the label whose rectangle
        # matches rects_dst_pristine[r_idx].
        # But labels are 1-indexed in canonical form...
        # Actually, rects_dst_pristine[i] corresponds to label i+1 in canonical
        # order because build_rects iterates labels 1..n.
        label_dst = r_idx + 1
        dst_labels_used.add(label_dst)
        # Get its H and V positions in pristine M_{k_dst}
        h_positions_dst = positions_of(H_dst_pristine, label_dst)
        v_positions_dst = positions_of(V_dst_pristine, label_dst)
        assert len(h_positions_dst) == 2 and len(v_positions_dst) == 2
        # New label in lifted instance: n_src + tail_idx + 1
        new_label = n_src + tail_idx + 1
        # Assign it positions strictly past all source positions.
        # A simple scheme: put the 2 H positions at base_H + 2*tail_idx and +1,
        # and similarly for V. This makes each new rect have H-span of width 1
        # and V-span of width 1, disjoint from all others (but also from each
        # other, which means they won't intersect at all).
        #
        # That's too trivial - the 10th box's rectangles should still intersect
        # each other in the pristine k-box pattern. To preserve the Caoduro
        # pattern inside the tail, we need to use the tail rects' *relative*
        # positions (how they intersect each other in pristine M_{k_dst})
        # and offset them as a block.

        # Simple but correct approach: use the relative positions within the
        # tail, offset by base_H / base_V.
        # First, find the min H-position among all tail rects in pristine M_dst.
        pass

    # Actually, reworking: the easiest correct approach is to take the
    # interleaved sequence of H (V) positions from pristine M_{k_dst},
    # restricted to tail labels, and append those positions (shifted) after
    # the source sequence.

    # Re-do: get the subsequence of H_dst_pristine consisting only of tail labels.
    tail_label_set = set(r_idx + 1 for r_idx in tail_rect_indices)
    H_tail_subseq = [lab for lab in H_dst_pristine if lab in tail_label_set]
    V_tail_subseq = [lab for lab in V_dst_pristine if lab in tail_label_set]

    # Relabel the tail: the kth distinct tail label (by first appearance in
    # H_tail_subseq) becomes n_src + k.
    tail_order = []
    tail_seen = set()
    for lab in H_tail_subseq:
        if lab not in tail_seen:
            tail_order.append(lab)
            tail_seen.add(lab)
    relabel = {old: n_src + i + 1 for i, old in enumerate(tail_order)}

    H_tail_final = [relabel[lab] for lab in H_tail_subseq]
    V_tail_final = [relabel[lab] for lab in V_tail_subseq]

    # Append to source.
    H_out = list(H_src) + H_tail_final
    V_out = list(V_src) + V_tail_final

    # Sanity
    from collections import Counter
    cH = Counter(H_out); cV = Counter(V_out)
    assert len(cH) == n_dst, f"expected {n_dst} distinct H labels, got {len(cH)}"
    assert all(v == 2 for v in cH.values()), "some H label not appearing twice"
    assert all(v == 2 for v in cV.values()), "some V label not appearing twice"

    return canonicalize(H_out, V_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle", help="path to source pickle (smaller k)")
    ap.add_argument("--to-k", type=int, required=True,
                    help="target k (must be larger than source k)")
    ap.add_argument("--out", required=True,
                    help="output pickle path")
    args = ap.parse_args()

    with open(args.pickle, "rb") as f:
        d = pickle.load(f)
    H_src = d.get("H") or d.get("best_instance_H")
    V_src = d.get("V") or d.get("best_instance_V")
    k_src = d.get("k")
    if H_src is None or V_src is None:
        print("ERROR: source pickle missing H/V", file=sys.stderr)
        sys.exit(1)
    if k_src is None:
        k_src = int(round((max(max(H_src), max(V_src)) / 4) ** 0.5))
        print(f"[info] inferred k_src={k_src} from source size",
              file=sys.stderr)

    print(f"Lifting {args.pickle}")
    print(f"  k={k_src} -> k={args.to_k}")
    print(f"  n={4*k_src*k_src} -> n={4*args.to_k*args.to_k}")
    print(f"  source gap: {d.get('gap', '?')}")

    H_out, V_out = lift_instance(H_src, V_src, k_src, args.to_k)
    n_out = max(max(H_out), max(V_out))
    print(f"  lifted n={n_out}")

    # Compute max_clique on the result
    from kbox_parallel import vec_max_clique_at_grid
    mc = vec_max_clique_at_grid(H_out, V_out)
    tf = mc <= 2
    print(f"  lifted max_clique={mc}  triangle_free={tf}")

    data_out = {
        "k": args.to_k,
        "round": -1,  # sentinel for "lifted seed"
        "H": H_out, "V": V_out,
        "gap": 0.0,   # unknown until scored
        "triangle_free": tf,
        "max_clique": mc,
        "n": n_out,
        "source_pickle": args.pickle,
        "source_k": k_src,
        "source_gap": d.get("gap"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(data_out, f)
    print(f"  saved: {args.out}")


if __name__ == "__main__":
    main()
