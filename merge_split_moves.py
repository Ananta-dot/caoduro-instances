#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_split_moves.py
====================

Larger-step structural moves for the k-box MISR search:

  - merge_adjacent_boxes(H, V, rng): collapse two spatially-adjacent diagonal
    blocks into a single interleaved region. Produces meaningful structural
    change in one move.

  - split_box(H, V, rng): take one diagonal block and split its labels into
    two interleaved sub-blocks at different geometric offsets. Inverse of
    merge.

  - swap_random_chunks(H, V, rng): take two large contiguous chunks of H (or V)
    and swap them. Larger neighborhood than swap_within_box.

These are designed to be drop-in alternatives to perturb_segment_extent
and shuffle_boxes — same input/output signature, same canonicalize-on-exit.

Notes on safety:
  - All moves preserve the multiplicity invariant (each label appears exactly
    twice in H and twice in V). Verified by assertion before returning.
  - Moves can produce non-triangle-free instances when the original was
    triangle-free; the caller (or the sanity filter) is responsible for
    deciding whether that's acceptable.
  - Moves never increase n. The label set is preserved.

These moves are NOT guaranteed to improve LP/ILP ratio. They're search
proposals; local search will accept or reject them based on scoring.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np


Seq = List[int]
Instance = Tuple[Seq, Seq]


def _verify_valid(H: Seq, V: Seq) -> None:
    """Assert that (H, V) is a valid twin-sequence instance."""
    cH = Counter(H)
    cV = Counter(V)
    n = max(max(H), max(V))
    assert all(cH[i] == 2 for i in range(1, n + 1)), \
        f"H multiplicity invariant violated: {cH}"
    assert all(cV[i] == 2 for i in range(1, n + 1)), \
        f"V multiplicity invariant violated: {cV}"
    assert len(H) == 2 * n and len(V) == 2 * n


def _canonicalize_local(H: Seq, V: Seq) -> Instance:
    """Local copy of canonicalize (relabel by first-appearance in H)."""
    order: List[int] = []
    seen = set()
    for x in H:
        if x not in seen:
            order.append(x)
            seen.add(x)
    rel = {old: new for new, old in enumerate(order, 1)}
    H2 = [rel[x] for x in H]
    V2 = [rel[x] for x in V]
    return H2, V2


def _spatial_cluster_labels(H: Seq, V: Seq, n_clusters: int = 2,
                            ) -> List[List[int]]:
    """
    Partition the labels into n_clusters groups by 1D projection of centroids
    onto the main diagonal. Returns a list of n_clusters lists of labels.

    For an instance whose structure is roughly a k-box-along-diagonal, this
    recovers boxes reasonably well. For drifted instances it produces
    something that's at least geometrically coherent.
    """
    n = max(max(H), max(V))
    # rectangle for each label: (xL, xR, yL, yR)
    rect = {}
    h_pos: Dict[int, List[int]] = defaultdict(list)
    v_pos: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(H):
        h_pos[lab].append(i)
    for i, lab in enumerate(V):
        v_pos[lab].append(i)
    for lab in range(1, n + 1):
        xs = sorted(h_pos[lab]) if h_pos[lab] else [0, 0]
        ys = sorted(v_pos[lab]) if v_pos[lab] else [0, 0]
        rect[lab] = (xs[0], xs[1], ys[0], ys[1])

    centroids = np.array([
        [(rect[lab][0] + rect[lab][1]) / 2.0,
         (rect[lab][2] + rect[lab][3]) / 2.0]
        for lab in range(1, n + 1)
    ])
    proj = centroids.sum(axis=1)
    order = np.argsort(proj)
    chunks = np.array_split(order, n_clusters)
    return [[int(idx) + 1 for idx in chunk] for chunk in chunks]


def merge_adjacent_boxes(H: Seq, V: Seq, rng: random.Random,
                         k_hint: int = None) -> Instance:
    """
    Pick two spatially-adjacent diagonal blocks and interleave their positions
    in H and V, so the two blocks effectively merge into one with mixed
    geometry.

    Mechanics:
      1. Spatial-cluster all labels into k_hint blocks (or sqrt(n/4) if not given).
      2. Pick two adjacent blocks (i, i+1) along the diagonal.
      3. For each of H and V, replace the positions of the union of those
         block labels with a new ordering where the two blocks' positions
         are interleaved (alternating).

    Returns canonicalized (H, V). If the move can't apply (k_hint=1 etc),
    returns canonicalized input unchanged.
    """
    n = max(max(H), max(V))
    if k_hint is None:
        # heuristic: assume the instance is k-boxish with k = sqrt(n/4)
        k_hint = max(2, int(round((n / 4) ** 0.5)))
    if k_hint < 2:
        return _canonicalize_local(H, V)

    blocks = _spatial_cluster_labels(H, V, n_clusters=k_hint)
    if len(blocks) < 2:
        return _canonicalize_local(H, V)

    # pick adjacent blocks
    i = rng.randint(0, len(blocks) - 2)
    block_a = set(blocks[i])
    block_b = set(blocks[i + 1])
    union = block_a | block_b

    def interleave_positions(seq: Seq) -> Seq:
        """Find positions of union labels in seq, interleave a/b alternately."""
        # positions of union labels in seq, in order of appearance
        positions = [(idx, lab) for idx, lab in enumerate(seq)
                     if lab in union]
        if not positions:
            return list(seq)
        # split into a-list and b-list preserving original order
        a_list = [(idx, lab) for idx, lab in positions if lab in block_a]
        b_list = [(idx, lab) for idx, lab in positions if lab in block_b]
        # interleave: take from a, then b, alternately
        # if lists are uneven, the longer one tail gets appended at end
        merged_labels: List[int] = []
        ai = bi = 0
        while ai < len(a_list) or bi < len(b_list):
            if ai < len(a_list):
                merged_labels.append(a_list[ai][1])
                ai += 1
            if bi < len(b_list):
                merged_labels.append(b_list[bi][1])
                bi += 1
        # write merged_labels back into the original positions
        new_seq = list(seq)
        for (idx, _), new_lab in zip(positions, merged_labels):
            new_seq[idx] = new_lab
        return new_seq

    H2 = interleave_positions(H)
    V2 = interleave_positions(V)

    _verify_valid(H2, V2)
    return _canonicalize_local(H2, V2)


def split_box(H: Seq, V: Seq, rng: random.Random,
              k_hint: int = None) -> Instance:
    """
    Take one spatial block and reorder its labels into two interleaved
    sub-blocks. Inverse of merge_adjacent_boxes for matched parameters.
    """
    n = max(max(H), max(V))
    if k_hint is None:
        k_hint = max(2, int(round((n / 4) ** 0.5)))
    if k_hint < 1:
        return _canonicalize_local(H, V)

    blocks = _spatial_cluster_labels(H, V, n_clusters=k_hint)
    candidate_blocks = [b for b in blocks if len(b) >= 4]
    if not candidate_blocks:
        return _canonicalize_local(H, V)

    block = rng.choice(candidate_blocks)
    # split the block into two halves randomly
    split_labels = list(block)
    rng.shuffle(split_labels)
    half = len(split_labels) // 2
    half_a = set(split_labels[:half])
    half_b = set(split_labels[half:])

    def split_positions(seq: Seq) -> Seq:
        """Reorder positions of block labels: all half-a first, then half-b."""
        positions = [(idx, lab) for idx, lab in enumerate(seq)
                     if lab in (half_a | half_b)]
        if not positions:
            return list(seq)
        # collect labels in order: half-a labels first (in original order),
        # then half-b labels
        a_labs = [lab for (_, lab) in positions if lab in half_a]
        b_labs = [lab for (_, lab) in positions if lab in half_b]
        new_labels = a_labs + b_labs
        new_seq = list(seq)
        for (idx, _), new_lab in zip(positions, new_labels):
            new_seq[idx] = new_lab
        return new_seq

    H2 = split_positions(H)
    V2 = split_positions(V)

    _verify_valid(H2, V2)
    return _canonicalize_local(H2, V2)


def swap_random_chunks(H: Seq, V: Seq, rng: random.Random,
                       chunk_size_frac: float = 0.05) -> Instance:
    """
    Pick two non-overlapping contiguous chunks of either H or V and swap them.
    Chunk size is chunk_size_frac of the sequence length.

    This is a much larger move than perturb_segment_extent (one position swap).
    Per move can permute ~5% of one sequence's positions at once.
    """
    n = max(max(H), max(V))
    L = 2 * n
    chunk = max(2, int(L * chunk_size_frac))

    if rng.random() < 0.5:
        S = list(H)
        which_h = True
    else:
        S = list(V)
        which_h = False

    if 2 * chunk + 1 >= L:
        return _canonicalize_local(H, V)

    # pick two non-overlapping ranges
    a_start = rng.randint(0, L - 2 * chunk - 1)
    b_start = rng.randint(a_start + chunk, L - chunk)
    # swap
    a_segment = S[a_start:a_start + chunk]
    b_segment = S[b_start:b_start + chunk]
    S[a_start:a_start + chunk] = b_segment
    S[b_start:b_start + chunk] = a_segment

    if which_h:
        H2, V2 = S, list(V)
    else:
        H2, V2 = list(H), S

    _verify_valid(H2, V2)
    return _canonicalize_local(H2, V2)


# ---------------------------------------------------------------- self-test


def _smoke_test():
    """Build a pristine k=4 instance, apply each move, verify validity."""
    import sys
    sys.path.insert(0, '.')
    from kbox_misr import kbox_instance

    rng = random.Random(0)
    H, V = kbox_instance(4)
    n = max(max(H), max(V))
    print(f"Pristine k=4: n={n}")
    _verify_valid(H, V)

    for move_name, move_fn in [
        ("merge_adjacent_boxes", lambda: merge_adjacent_boxes(H, V, rng)),
        ("split_box", lambda: split_box(H, V, rng)),
        ("swap_random_chunks", lambda: swap_random_chunks(H, V, rng)),
    ]:
        H2, V2 = move_fn()
        n2 = max(max(H2), max(V2))
        # check moves changed something
        diff = sum(1 for a, b in zip(H, H2) if a != b)
        diff += sum(1 for a, b in zip(V, V2) if a != b)
        print(f"  {move_name}: n={n2}, positions changed: {diff}")
    print("smoke test PASS")


if __name__ == "__main__":
    _smoke_test()
