#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kbox_search.py — search integration for k-box MISR instances.

CORRECTED VERSION WITH ROBUST SAVING.

Every round's best (if >= SAVE_THRESHOLD) is pickled to elites_above_threshold/.
A final summary pickle is always written to run_outputs/, even on Ctrl+C.

Usage:
    python3 kbox_search.py --smoke
    python3 kbox_search.py --save-smoke
    python3 kbox_search.py --verify-all
    python3 kbox_search.py --run --k-start 9 --k-end 10 --rounds 5

Output locations (relative to cwd):
    elites_above_threshold/  — every elite with ratio >= SAVE_THRESHOLD (1.40)
    run_outputs/             — final summary + full elite pool
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from kbox_misr import (
    kbox_instance,
    kbox_instance_with_key_map,
    _kbox_geom,
    verify,
    expected_edges,
    actual_edges,
    triangle_free,
    explicit_IS_labels,
    Seq, Instance,
)

try:
    from mistr_runner import (
        canonicalize,
        instance_key,
        seq_spans,
        build_rects,
        grid_points,
        covers_grid_closed,
        solve_lp_ilp,
        random_valid_seq,
        motif_seeds,
    )
    _HAS_MISTR = True
except Exception:
    _HAS_MISTR = False
    from kbox_misr import (  # noqa: F401
        canonicalize, instance_key, seq_spans, build_rects,
        grid_points, covers_grid_closed,
    )

    def random_valid_seq(n: int, rng: random.Random) -> Seq:
        seq = [i for i in range(1, n + 1) for _ in range(2)]
        rng.shuffle(seq)
        return seq

    def motif_seeds(n: int) -> List[Seq]:
        rainbow = list(range(1, n + 1)) + list(range(n, 0, -1))
        doubled = [x for i in range(1, n + 1) for x in (i, i)]
        return [rainbow, doubled]

    solve_lp_ilp = None


# =========================================================================== #
# Save configuration                                                           #
# =========================================================================== #

SAVE_THRESHOLD = 1.40
ELITES_DIR = "elites_above_threshold"
FINAL_DIR = "run_outputs"

# Pathology guard: reject any instance with max_clique >= this fraction of n
# or any instance with absolute max_clique >= MAX_CLIQUE_ABSOLUTE.
# Catches degenerate "everything-stacked-on-one-point" instances that
# produce absurd lp/ilp ratios without being meaningful rectangle graphs.
MAX_CLIQUE_ABSOLUTE = 8      # reject if mc > this
MAX_CLIQUE_FRACTION = 0.03   # reject if mc/n > this (e.g., 3% of n at a point)


# =========================================================================== #
# 1. Seed pool generator                                                       #
# =========================================================================== #


def _box_membership(k: int) -> Dict[int, List[int]]:
    (_H, _V), key_to_lab = kbox_instance_with_key_map(k)
    by_box: Dict[int, List[int]] = defaultdict(list)
    for (kind, a, i), lab in key_to_lab.items():
        by_box[a].append(lab)
    for a in by_box:
        by_box[a].sort()
    return dict(by_box)


def shuffle_boxes(H: Seq, V: Seq, k: int, rng: random.Random) -> Instance:
    """Randomly permute the k boxes along the diagonal."""
    by_box = _box_membership(k)
    boxes = list(range(1, k + 1))
    perm = boxes[:]
    rng.shuffle(perm)
    relabel: Dict[int, int] = {}
    for a_idx, a_new in enumerate(perm):
        a_old = boxes[a_idx]
        old_labs = by_box[a_old]
        new_labs = by_box[a_new]
        for o, n in zip(old_labs, new_labs):
            relabel[o] = n
    H2 = [relabel[x] for x in H]
    V2 = [relabel[x] for x in V]
    return canonicalize(H2, V2)


def swap_within_box(H: Seq, V: Seq, k: int, box: int,
                    rng: random.Random) -> Instance:
    """Swap two same-kind segments inside a single k-box."""
    (_H0, _V0), key_to_lab = kbox_instance_with_key_map(k)
    by_kind: Dict[str, List[int]] = defaultdict(list)
    for (kind, a, i), lab in key_to_lab.items():
        if a == box:
            by_kind[kind].append(lab)
    candidates = [kind for kind, labs in by_kind.items() if len(labs) >= 2]
    if not candidates:
        return canonicalize(H, V)
    kind = rng.choice(candidates)
    labs = by_kind[kind]
    a, b = rng.sample(labs, 2)
    relabel = {a: b, b: a}
    H2 = [relabel.get(x, x) for x in H]
    V2 = [relabel.get(x, x) for x in V]
    return canonicalize(H2, V2)


def perturb_segment_extent(H: Seq, V: Seq, rng: random.Random,
                           which: str = "auto") -> Instance:
    """Slide one of a label's positions by +/- 1 step."""
    n = max(max(H), max(V))
    if which == "auto":
        which = rng.choice(("H", "V"))
    S = list(H if which == "H" else V)
    lab = rng.randint(1, n)
    positions = [i for i, x in enumerate(S) if x == lab]
    if len(positions) != 2:
        return canonicalize(H, V)
    i = rng.choice(positions)
    direction = rng.choice((-1, 1))
    j = i + direction
    if not (0 <= j < len(S)):
        return canonicalize(H, V)
    if S[j] == lab:
        return canonicalize(H, V)
    S[i], S[j] = S[j], S[i]
    if which == "H":
        return canonicalize(S, V)
    return canonicalize(H, S)


def load_pickle_elites(k: int, elites_dir: str = "elites_above_threshold",
                       max_elites: int = 8,
                       min_gap: float = 0.0,
                       require_triangle_free: bool = False
                       ) -> List[Tuple[float, Seq, Seq]]:
    """
    Scan elites_dir for pickle files matching target k and load the top
    max_elites by gap. Returns a list of (gap, H, V) sorted descending.

    Filters by:
      - k matches (from the pickle's 'k' field or filename kN)
      - n matches 4*k*k (sanity check)
      - gap >= min_gap
      - optionally, triangle_free if require_triangle_free=True

    Silently skips malformed pickles; prints a summary of what was loaded.
    """
    if not os.path.isdir(elites_dir):
        return []

    target_n = 4 * k * k
    loaded: List[Tuple[float, Seq, Seq]] = []
    scanned = 0
    skipped = 0

    for fname in os.listdir(elites_dir):
        if not fname.endswith(".pkl"):
            continue
        path = os.path.join(elites_dir, fname)
        scanned += 1
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
        except Exception:
            skipped += 1
            continue

        pk = d.get("k", None)
        # if k field missing, try parsing from filename "kN_r..."
        if pk is None and fname.startswith("k"):
            try:
                pk = int(fname.split("_")[0][1:])
            except Exception:
                skipped += 1
                continue
        if pk != k:
            continue

        H, V = d.get("H"), d.get("V")
        if H is None or V is None:
            skipped += 1
            continue
        if max(max(H), max(V)) != target_n:
            skipped += 1
            continue

        gap = float(d.get("gap", 0.0))
        if gap < min_gap:
            continue

        if require_triangle_free:
            tf = d.get("triangle_free", None)
            if tf is False:
                continue
            if tf is None and max_clique_at_grid(H, V) > 2:
                continue

        loaded.append((gap, list(H), list(V)))

    loaded.sort(key=lambda t: -t[0])
    loaded = loaded[:max_elites]
    if loaded:
        print(f"  [elite-reuse] k={k}: loaded {len(loaded)} prior elites "
              f"(scanned {scanned}, skipped {skipped}, top gap={loaded[0][0]:.4f})")
    elif scanned > 0:
        print(f"  [elite-reuse] k={k}: no matching elites found "
              f"(scanned {scanned}, skipped {skipped})")

    return loaded


def kbox_seeded_pool(k: int, rng: random.Random, count: int,
                     include_perturbations: bool = True,
                     elite_reuse_dir: Optional[str] = None,
                     elite_reuse_count: int = 8,
                     elite_reuse_min_gap: float = 0.0,
                     elite_reuse_require_tf: bool = False
                     ) -> List[Instance]:
    """Build a pool of seeds at n = 4*k*k mixing k-boxes, variants, motifs."""
    pool: List[Instance] = []
    base_H, base_V = kbox_instance(k)
    pool.append((base_H, base_V))

    # --- NEW: inject saved elites from prior runs ---
    if elite_reuse_dir:
        prior = load_pickle_elites(
            k, elites_dir=elite_reuse_dir,
            max_elites=elite_reuse_count,
            min_gap=elite_reuse_min_gap,
            require_triangle_free=elite_reuse_require_tf,
        )
        for (_gap, H_p, V_p) in prior:
            pool.append((H_p, V_p))
            # and a few perturbations of each prior elite (they've already
            # escaped the basin of attraction of pristine M_k; local search
            # from their neighborhood should find nearby improvements)
            if include_perturbations:
                for _ in range(2):
                    Hx, Vx = H_p, V_p
                    for _step in range(rng.randint(1, 3)):
                        Hx, Vx = perturb_segment_extent(Hx, Vx, rng)
                    pool.append((Hx, Vx))

    if include_perturbations:
        for _ in range(max(2, count // 6)):
            pool.append(shuffle_boxes(base_H, base_V, k, rng))
        for _ in range(max(2, count // 6)):
            box = rng.randint(1, k)
            pool.append(swap_within_box(base_H, base_V, k, box, rng))
        for _ in range(max(2, count // 6)):
            H, V = base_H, base_V
            for _step in range(rng.randint(1, 3)):
                H, V = perturb_segment_extent(H, V, rng)
            pool.append((H, V))

    n = 4 * k * k
    while len(pool) < count:
        if rng.random() < 0.3:
            motifs = motif_seeds(n)
            m = rng.choice(motifs)
            other = random_valid_seq(n, rng)
            if rng.random() < 0.5:
                pool.append(canonicalize(m, other))
            else:
                pool.append(canonicalize(other, m))
        else:
            pool.append(canonicalize(random_valid_seq(n, rng),
                                     random_valid_seq(n, rng)))
    seen: Set[str] = set()
    uniq: List[Instance] = []
    for H, V in pool[:count]:
        key = instance_key(H, V)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((H, V))
    return uniq


# =========================================================================== #
# 2. Triangle-free filter and LP-only scoring                                  #
# =========================================================================== #


def max_clique_at_grid(H: Seq, V: Seq) -> int:
    rects = build_rects(H, V)
    pts = grid_points(rects)
    covers = covers_grid_closed(rects, pts)
    return max((len(c) for c in covers), default=0)


def is_sane_instance(H: Seq, V: Seq,
                     max_clique_abs: int = MAX_CLIQUE_ABSOLUTE,
                     max_clique_frac: float = MAX_CLIQUE_FRACTION
                     ) -> Tuple[bool, int]:
    """
    Return (is_sane, max_clique). Used to filter pathological instances
    (e.g. mc=133 at n=484) before saving, training on, or trusting
    the ratio of. Catches "all rectangles stacked on one point" cases.

    Filter rules (an instance is insane if ANY applies):
      - mc > MAX_CLIQUE_ABSOLUTE (default 8) — unambiguously pathological
      - n >= 100 AND mc/n > MAX_CLIQUE_FRACTION — scaled rule, only bites
        at larger n where small-constant mc values are expected.

    At small n (n < 100), many legitimate instances have mc values that
    would trip the fraction rule (e.g. pristine M_3 has mc=2 at n=36,
    which is mc/n=0.056). Those are not pathological.
    """
    mc = max_clique_at_grid(H, V)
    n = max(max(H), max(V)) if H else 1
    if mc > max_clique_abs:
        return False, mc
    if n >= 100 and mc / n > max_clique_frac:
        return False, mc
    return True, mc


def tfilter_evaluator(H: Seq, V: Seq, grb_threads: int = 0
                      ) -> Optional[Tuple[float, float, float]]:
    if max_clique_at_grid(H, V) > 2:
        return None
    if solve_lp_ilp is None:
        return None
    rects = build_rects(H, V)
    lp, ilp = solve_lp_ilp(rects, grb_threads=grb_threads)
    if ilp <= 0:
        return lp, ilp, 0.0
    return lp, ilp, lp / ilp


def lp_only_score(H: Seq, V: Seq, grb_threads: int = 0) -> Optional[float]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return None
    rects = build_rects(H, V)
    pts = grid_points(rects)
    covers = covers_grid_closed(rects, pts)
    m = gp.Model("misr_lp_only")
    m.setParam("OutputFlag", 0)
    if grb_threads > 0:
        m.setParam("Threads", grb_threads)
    n = len(rects)
    x = m.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    m.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if len(S) >= 2:
            m.addConstr(gp.quicksum(x[i] for i in S) <= 1)
    m.optimize()
    return float(m.objVal) if m.status == GRB.OPTIMAL else None


# =========================================================================== #
# 3. Padding helper                                                            #
# =========================================================================== #


def assemble_at_n(n: int, rng: random.Random,
                  with_kbox: bool = True) -> Instance:
    if not with_kbox:
        H = random_valid_seq(n, rng)
        V = random_valid_seq(n, rng)
        return canonicalize(H, V)
    k = 0
    while 4 * (k + 1) * (k + 1) <= n:
        k += 1
    if k < 1:
        return canonicalize(random_valid_seq(n, rng),
                            random_valid_seq(n, rng))
    base_H, base_V = kbox_instance(k)
    m = 4 * k * k
    if m == n:
        return canonicalize(base_H, base_V)
    dummies = list(range(m + 1, n + 1))
    H = list(base_H)
    V = list(base_V)
    for d in dummies:
        H.extend([d, d])
        V.extend([d, d])
    return canonicalize(H, V)


# =========================================================================== #
# 4. SAVE HELPERS (robust against silent failure)                              #
# =========================================================================== #


def _safe_save_pickle(path: str, data: dict) -> bool:
    """Pickle dump with explicit error reporting. Returns True on success."""
    try:
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        size = os.path.getsize(path)
        if size == 0:
            print(f"  [SAVE ERROR] wrote 0 bytes to {path}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"  [SAVE ERROR] {path}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False


def save_elite(k: int, r: int, score: float, H: Seq, V: Seq,
               elites_dir: str = ELITES_DIR,
               threshold: float = SAVE_THRESHOLD,
               verbose: bool = True,
               enforce_sanity: bool = True) -> Optional[str]:
    """Save an elite if its score meets the threshold. Returns filename or None.

    With enforce_sanity=True (default), rejects pathological instances where
    a single grid point is covered by too many rectangles (e.g. the mc=133
    at n=484 case). Such instances have meaningless lp/ilp ratios.
    """
    if score < threshold:
        return None

    if enforce_sanity:
        sane, mc_check = is_sane_instance(H, V)
        if not sane:
            n_check = max(max(H), max(V)) if H else 0
            if verbose:
                print(f"  [REJECTED: pathological] k={k} r={r} "
                      f"gap={score:.4f} mc={mc_check} n={n_check} "
                      f"(mc/n={mc_check/max(n_check,1):.3f})",
                      file=sys.stderr)
            return None

    try:
        mc = max_clique_at_grid(H, V)
        tf = mc <= 2
        n = max(max(H), max(V)) if H else 0
    except Exception as e:
        print(f"  [save_elite] could not compute tf/mc: {e}", file=sys.stderr)
        tf, mc, n = None, None, None
    fname = os.path.join(
        elites_dir,
        f"k{k}_r{r}_gap{score:.4f}_tf{int(tf) if tf is not None else 'NA'}.pkl"
    )
    data = {
        "k": k, "round": r, "H": list(H), "V": list(V),
        "gap": float(score), "triangle_free": tf, "max_clique": mc,
        "n": n, "timestamp": time.time(),
    }
    if _safe_save_pickle(fname, data):
        if verbose:
            print(f"  [SAVED] {fname}  tf={tf}  mc={mc}  n={n}")
        return fname
    return None


def save_final(best_overall: float,
               best_instance: Optional[Instance],
               all_elites: List[Tuple[float, Seq, Seq, int, int]],
               k_start: int, k_end: int,
               final_dir: str = FINAL_DIR) -> None:
    """End-of-run summary. Runs unconditionally."""
    os.makedirs(final_dir, exist_ok=True)
    summary_path = os.path.join(
        final_dir,
        f"final_k{k_start}-{k_end}_gap{best_overall:.4f}.pkl"
    )
    tf = mc = n = None
    if best_instance is not None:
        H_b, V_b = best_instance
        try:
            mc = max_clique_at_grid(H_b, V_b)
            tf = mc <= 2
            n = max(max(H_b), max(V_b))
        except Exception:
            pass
    data = {
        "k_start": k_start, "k_end": k_end,
        "best_overall": float(best_overall),
        "best_instance_H": list(best_instance[0]) if best_instance else None,
        "best_instance_V": list(best_instance[1]) if best_instance else None,
        "best_triangle_free": tf,
        "best_max_clique": mc,
        "best_n": n,
        "all_elites": [
            {"score": float(s), "H": list(H), "V": list(V), "k": k, "round": r}
            for (s, H, V, k, r) in all_elites
        ],
        "timestamp": time.time(),
    }
    if _safe_save_pickle(summary_path, data):
        print(f"\n[FINAL SAVE] {summary_path}")
        print(f"             best_overall={best_overall:.4f}  tf={tf}  mc={mc}  n={n}")
        print(f"             {len(all_elites)} elites recorded")


# =========================================================================== #
# 5. Patched PatternBoost loop                                                 #
# =========================================================================== #


def run_kbox_patternboost(
    k_start: int = 3,
    k_end: int = 6,
    rounds_per_k: int = 10,
    seeds_per_round: int = 32,
    local_time_per_seed: float = 3.0,
    require_triangle_free: bool = True,
    rng_seed: int = 123,
    save_threshold: float = SAVE_THRESHOLD,
    elites_dir: str = ELITES_DIR,
    final_dir: str = FINAL_DIR,
    reuse_elites: bool = True,
    reuse_count: int = 8,
    reuse_min_gap: float = 0.0,
    reuse_require_tf: bool = False,
    use_transformer: bool = False,
    transformer_samples_per_round: int = 8,
    transformer_train_steps_per_round: int = 40,
    transformer_batch_size: int = 16,
    transformer_temperature: float = 1.0,
    transformer_top_p: float = 0.9,
    transformer_elite_pool_size: int = 128,
    fast_parallel: bool = False,
    fast_n_workers: Optional[int] = None,
    fast_grb_threads: int = 1,
    fast_verify_top_k: int = 8,
    fast_verify_time_limit: float = 60.0,
    fast_neighbor_ilp_time: float = 0.8,
):
    """
    PatternBoost-style driver with three parallel learning channels:

      1. Pristine k-box seeds (structural scaffolding).
      2. Elite reuse from previously saved pickles (if reuse_elites=True).
      3. Transformer proposer (if use_transformer=True) — trained on the
         accumulated elite pool, samples novel candidate (H, V) each round.

    If use_transformer=True, all three channels feed seeds into each round.
    The transformer is trained incrementally on the full elite pool,
    including any instances loaded from prior pickles.
    """
    if not _HAS_MISTR:
        print("mistr_runner not importable; cannot run.", file=sys.stderr)
        return None, None

    from mistr_runner import local_search

    # --- optional transformer setup ---
    model = None
    opt = None
    transformer_training_pool: List[Tuple[float, Seq, Seq]] = []
    # These get set if the transformer init succeeds. We rebuild the model
    # and its helpers locally so we can size MAX_N for the target k (the
    # stock mistr_runner has MAX_N=128 which caps at n=128).
    _xf_sample_model = None
    _xf_make_batch = None
    _xf_train_one_step = None

    if use_transformer:
        try:
            import math
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from mistr_runner import DEVICE

            # Size vocabulary for the max n we'll encounter.
            target_max_n = 4 * k_end * k_end + 16  # safety margin
            xf_BASE_VOCAB = 3  # {BOS, SEP, EOS}
            xf_MAX_N = max(target_max_n, 128)
            xf_SPECIAL = {"BOS": 0, "SEP": 1, "EOS": 2}

            print(f"\nTransformer enabled (device={DEVICE}).")
            print(f"  samples/round={transformer_samples_per_round}  "
                  f"train steps/round={transformer_train_steps_per_round}  "
                  f"pool size={transformer_elite_pool_size}")
            print(f"  vocab sized for MAX_N={xf_MAX_N} (target n up to {target_max_n})")

            # --- Local copies of TinyGPT + helpers, sized for xf_MAX_N ---
            class _PositionalEncoding(nn.Module):
                def __init__(self, d_model, max_len=8192):
                    super().__init__()
                    pe = torch.zeros(max_len, d_model)
                    pos = torch.arange(0, max_len).unsqueeze(1)
                    div = torch.exp(torch.arange(0, d_model, 2) *
                                    (-math.log(10000.0) / d_model))
                    pe[:, 0::2] = torch.sin(pos * div)
                    pe[:, 1::2] = torch.cos(pos * div)
                    self.register_buffer("pe", pe)

                def forward(self, x):
                    return x + self.pe[:x.size(1)]

            class _TinyGPT(nn.Module):
                def __init__(self, d=192, nhead=6, nlayers=3, dropout=0.1,
                             max_n=xf_MAX_N, base_vocab=xf_BASE_VOCAB):
                    super().__init__()
                    self.base_vocab = base_vocab
                    self.max_n = max_n
                    self.label_embed = nn.Embedding(base_vocab + max_n, d)
                    self.n_embed = nn.Embedding(max_n + 1, d)
                    self.pos = _PositionalEncoding(d, max_len=4 * max_n + 32)
                    layer = nn.TransformerEncoderLayer(
                        d_model=d, nhead=nhead, dim_feedforward=4 * d,
                        dropout=dropout, batch_first=True,
                    )
                    self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
                    self.out = nn.Linear(d, base_vocab + max_n)

                def forward(self, tokens, n_scalar):
                    tok_emb = self.label_embed(tokens)
                    n_emb = self.n_embed(n_scalar).unsqueeze(1)
                    n_emb = n_emb.expand(-1, tok_emb.size(1), -1)
                    x = self.pos(tok_emb + n_emb)
                    L = x.size(1)
                    causal = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
                    h = self.enc(x, mask=causal)
                    return self.out(h)

            def _seq_to_tokens(seq):
                return [xf_BASE_VOCAB + (i - 1) for i in seq]

            def _tokens_to_seq(tokens):
                return [t - xf_BASE_VOCAB + 1 for t in tokens]

            def _make_batch(elites, B, rng_):
                tlist, tglist, ns = [], [], []
                for _ in range(B):
                    _, H_e, V_e = rng_.choice(elites)
                    n_e = max(H_e)
                    tok = ([xf_SPECIAL["BOS"]] + _seq_to_tokens(H_e)
                           + [xf_SPECIAL["SEP"]] + _seq_to_tokens(V_e)
                           + [xf_SPECIAL["EOS"]])
                    tgt = tok[1:] + [xf_SPECIAL["EOS"]]
                    tlist.append(torch.tensor(tok, dtype=torch.long))
                    tglist.append(torch.tensor(tgt, dtype=torch.long))
                    ns.append(n_e)
                L = max(len(t) for t in tlist)
                pad = xf_SPECIAL["EOS"]
                tokens = torch.full((B, L), pad, dtype=torch.long)
                targets = torch.full((B, L), pad, dtype=torch.long)
                for i, (t, tt) in enumerate(zip(tlist, tglist)):
                    tokens[i, :len(t)] = t
                    targets[i, :len(tt)] = tt
                return {
                    "tokens": tokens.to(DEVICE),
                    "n_scalar": torch.tensor(ns, dtype=torch.long, device=DEVICE),
                    "targets": targets.to(DEVICE),
                }

            @torch.no_grad()
            def _sample_model(m, n_target, temperature=1.0, top_p=0.9):
                m.eval()
                vocab = xf_BASE_VOCAB + xf_MAX_N
                toks = [xf_SPECIAL["BOS"]]

                def step_(mask_valid):
                    inp = torch.tensor(toks, dtype=torch.long,
                                       device=DEVICE).unsqueeze(0)
                    nvec = torch.tensor([n_target], dtype=torch.long,
                                        device=DEVICE)
                    logits = m(inp, nvec)[0, -1]
                    mask = torch.tensor(mask_valid, device=DEVICE)
                    logits = logits.masked_fill(~mask, -1e9)
                    probs = F.softmax(logits / temperature, dim=-1)
                    sorted_probs, idx = torch.sort(probs, descending=True)
                    csum = torch.cumsum(sorted_probs, dim=-1)
                    keep = csum <= top_p
                    if not torch.any(keep):
                        keep[0] = True
                    p = torch.zeros_like(probs).scatter(
                        0, idx[keep], sorted_probs[keep])
                    p = p / p.sum()
                    return int(torch.multinomial(p, 1).item())

                counts = [0] * (n_target + 1)
                while sum(counts) < 2 * n_target:
                    mask = [False] * vocab
                    for i in range(1, n_target + 1):
                        if counts[i] < 2:
                            mask[xf_BASE_VOCAB + (i - 1)] = True
                    toks.append(step_(mask))
                    lab = toks[-1] - xf_BASE_VOCAB + 1
                    counts[lab] += 1

                toks.append(xf_SPECIAL["SEP"])
                counts = [0] * (n_target + 1)
                max_len = 4 * n_target + 32
                while sum(counts) < 2 * n_target and len(toks) < max_len:
                    mask = [False] * vocab
                    for i in range(1, n_target + 1):
                        if counts[i] < 2:
                            mask[xf_BASE_VOCAB + (i - 1)] = True
                    toks.append(step_(mask))
                    lab = toks[-1] - xf_BASE_VOCAB + 1
                    counts[lab] += 1

                sep_idx = toks.index(xf_SPECIAL["SEP"])
                H_tok = toks[1:sep_idx]
                V_tok = toks[sep_idx + 1:]
                H_out = _tokens_to_seq(H_tok)
                V_out = _tokens_to_seq(V_tok)
                return canonicalize(H_out, V_out)

            def _train_one_step(m, optim, batch):
                m.train()
                logits = m(batch["tokens"], batch["n_scalar"])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch["targets"].reshape(-1),
                    ignore_index=xf_SPECIAL["EOS"],
                )
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                optim.step()
                return float(loss.item())

            model = _TinyGPT(d=192, nhead=6, nlayers=3).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(),
                                    lr=2e-4, weight_decay=1e-2)
            _xf_sample_model = _sample_model
            _xf_make_batch = _make_batch
            _xf_train_one_step = _train_one_step

        except Exception as e:
            import traceback
            print(f"[WARN] transformer setup failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("       falling back to search-only mode.", file=sys.stderr)
            model = None
            opt = None

    rng = random.Random(rng_seed)
    best_overall = 0.0
    best_instance: Optional[Instance] = None
    all_elites: List[Tuple[float, Seq, Seq, int, int]] = []

    # Verify output dirs are writable BEFORE the long run starts
    for d in (elites_dir, final_dir):
        try:
            os.makedirs(d, exist_ok=True)
            test_file = os.path.join(d, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
        except Exception as e:
            print(f"[ERROR] cannot write to {d}: {e}", file=sys.stderr)

    print(f"Save settings: threshold={save_threshold}")
    print(f"  elites_dir: {os.path.abspath(elites_dir)}")
    print(f"  final_dir:  {os.path.abspath(final_dir)}")

    if fast_parallel:
        import os as _os
        eff_workers = (fast_n_workers
                       if fast_n_workers is not None
                       else max(1, (_os.cpu_count() or 1) // max(1, fast_grb_threads)))
        print(f"Fast parallel path ENABLED:")
        print(f"  workers={eff_workers}  grb_threads={fast_grb_threads}")
        print(f"  verify_top_k={fast_verify_top_k}  "
              f"verify_time_limit={fast_verify_time_limit}s")
        print(f"  LP-only during search; ILP only on top elites.")
    else:
        print(f"Fast parallel path DISABLED (serial mistr_runner.local_search).")

    # try/finally ensures final save runs on Ctrl+C or exception
    try:
        for k in range(k_start, k_end + 1):
            n = 4 * k * k
            print(f"\n=== k={k}  (n={n}  target_gap={2*k*k/(k*k+3*k-2):.4f}) ===")

            # Reset transformer training pool across k boundaries: instances
            # at a different n are distributionally wrong for this k, and
            # keeping them causes the transformer to generate degenerate
            # samples (e.g., stacking labels producing max_clique=133 at n=484).
            if model is not None and transformer_training_pool:
                old_pool_size = len(transformer_training_pool)
                transformer_training_pool = [
                    (s, H_, V_) for (s, H_, V_) in transformer_training_pool
                    if H_ and max(H_) == n
                ]
                if len(transformer_training_pool) != old_pool_size:
                    print(f"  [transformer pool reset] kept "
                          f"{len(transformer_training_pool)}/{old_pool_size} "
                          f"entries matching n={n}")

            seeds = kbox_seeded_pool(
                k, rng, seeds_per_round,
                include_perturbations=True,
                elite_reuse_dir=(elites_dir if reuse_elites else None),
                elite_reuse_count=reuse_count,
                elite_reuse_min_gap=reuse_min_gap,
                elite_reuse_require_tf=reuse_require_tf,
            )

            # If transformer is enabled, seed its training pool with the
            # pickle-reused seeds that pass basic validity checks AND sanity.
            if model is not None:
                for (H, V) in seeds:
                    if max(H) != n:
                        continue
                    sane, _ = is_sane_instance(H, V)
                    if not sane:
                        continue
                    # use a neutral score (will get rescored by local_search)
                    transformer_training_pool.append((1.0, list(H), list(V)))

            best_at_k = 0.0
            for r in range(rounds_per_k):
                round_best = 0.0
                round_best_instance: Optional[Instance] = None

                # --- NEW: sample from transformer, add to seeds ---
                transformer_seeds: List[Instance] = []
                if (model is not None
                        and _xf_sample_model is not None
                        and len(transformer_training_pool) >= transformer_batch_size):
                    try:
                        for _ in range(transformer_samples_per_round):
                            h_s, v_s = _xf_sample_model(
                                model, n,
                                temperature=transformer_temperature,
                                top_p=transformer_top_p,
                            )
                            if (h_s and v_s
                                    and len(h_s) == 2 * n
                                    and len(v_s) == 2 * n
                                    and max(h_s) == n):
                                transformer_seeds.append((h_s, v_s))
                    except Exception as e:
                        import traceback
                        print(f"  [transformer sampling error] {e}",
                              file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)

                # Combine the pool: pristine+reuse seeds from kbox_seeded_pool,
                # plus transformer-generated ones. Transformer seeds go first
                # so they get evaluated even if the budget runs short.
                round_seeds = transformer_seeds + seeds

                # filter out non-triangle-free seeds up front (same as serial path)
                if require_triangle_free:
                    round_seeds = [
                        (H_s, V_s) for (H_s, V_s) in round_seeds
                        if max_clique_at_grid(H_s, V_s) <= 2
                    ]

                if fast_parallel:
                    # ---------- FAST PARALLEL PATH ----------
                    from kbox_fast import parallel_local_search_fast
                    round_best_at_this_k = 0.0
                    try:
                        merged_elites, pbest = parallel_local_search_fast(
                            round_seeds,
                            time_budget_s=local_time_per_seed,
                            n_workers=fast_n_workers,
                            grb_threads=fast_grb_threads,
                            alpha_lp=0.15,
                            beta_ilp=0.10,
                            elite_size=32,
                            neighbor_k=64,
                            verify_top_k=fast_verify_top_k,
                            verify_time_limit=fast_verify_time_limit,
                            neighbor_ilp_time=fast_neighbor_ilp_time,
                            rng_seed_base=rng.randint(0, 10**9),
                            verbose=False,
                        )
                    except Exception as e:
                        import traceback
                        print(f"  [parallel search error] {e}; "
                              f"falling back to serial for this round.",
                              file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)
                        merged_elites = []
                        pbest = 0.0

                    if pbest > round_best:
                        round_best = pbest
                        if merged_elites:
                            round_best_instance = (
                                merged_elites[0][1], merged_elites[0][2])
                    if pbest > best_at_k:
                        best_at_k = pbest
                    if pbest > best_overall and merged_elites:
                        best_overall = pbest
                        best_instance = (merged_elites[0][1],
                                         merged_elites[0][2])

                    # feed top elites into transformer training pool
                    if model is not None and merged_elites:
                        for (score, h_e, v_e) in merged_elites[:16]:
                            sane, _ = is_sane_instance(h_e, v_e)
                            if not sane:
                                continue
                            transformer_training_pool.append(
                                (score, list(h_e), list(v_e)))

                else:
                    # ---------- SERIAL PATH (original) ----------
                    for (H, V) in round_seeds:
                        es, best = local_search(
                            (H, V),
                            time_budget_s=local_time_per_seed,
                            rng=rng,
                            alpha_lp=0.15,
                            beta_ilp=0.10,
                            grb_threads=0,
                            elite_size=32,
                            neighbor_k=64,
                        )
                        if best is not None:
                            if best > round_best and es:
                                round_best = best
                                round_best_instance = (es[0][1], es[0][2])
                            if best > best_at_k:
                                best_at_k = best
                            if best > best_overall and es:
                                best_overall = best
                                best_instance = (es[0][1], es[0][2])

                            if model is not None and es:
                                for (score, h_e, v_e) in es[:8]:
                                    sane, _ = is_sane_instance(h_e, v_e)
                                    if not sane:
                                        continue
                                    transformer_training_pool.append(
                                        (score, list(h_e), list(v_e)))

                # Save every round's best if above threshold
                if (round_best_instance is not None
                        and round_best >= save_threshold):
                    H_rb, V_rb = round_best_instance
                    save_elite(k, r, round_best, H_rb, V_rb,
                               elites_dir=elites_dir,
                               threshold=save_threshold)
                    all_elites.append(
                        (round_best, list(H_rb), list(V_rb), k, r))

                # --- NEW: train transformer on top elites ---
                if model is not None and _xf_train_one_step is not None:
                    # keep only top-pool_size by score; larger pool dilutes signal
                    transformer_training_pool.sort(key=lambda t: -t[0])
                    if len(transformer_training_pool) > transformer_elite_pool_size:
                        transformer_training_pool = \
                            transformer_training_pool[:transformer_elite_pool_size]

                    if len(transformer_training_pool) >= transformer_batch_size:
                        last_loss = None
                        try:
                            for _ in range(transformer_train_steps_per_round):
                                batch = _xf_make_batch(
                                    transformer_training_pool,
                                    min(transformer_batch_size,
                                        len(transformer_training_pool)),
                                    rng,
                                )
                                last_loss = _xf_train_one_step(model, opt, batch)
                        except Exception as e:
                            import traceback
                            print(f"  [transformer training error] {e}",
                                  file=sys.stderr)
                            traceback.print_exc(file=sys.stderr)
                        if last_loss is not None:
                            extra = (f"  xf_loss={last_loss:.3f}  "
                                     f"pool={len(transformer_training_pool)}")
                        else:
                            extra = ""
                    else:
                        extra = (f"  xf_pool={len(transformer_training_pool)}"
                                 f"/{transformer_batch_size}")
                else:
                    extra = ""

                print(f"  round {r+1}/{rounds_per_k}  "
                      f"best_this_round={round_best:.4f}"
                      f"  best_at_k={best_at_k:.4f}"
                      f"  best_overall={best_overall:.4f}"
                      f"{extra}")

        print(f"\n=== DONE ===\nbest_overall={best_overall:.4f}")
    finally:
        save_final(best_overall, best_instance, all_elites,
                   k_start, k_end, final_dir=final_dir)

    return best_overall, best_instance


# =========================================================================== #
# 6. Standalone smoke tests                                                    #
# =========================================================================== #


def _smoke():
    rng = random.Random(0)
    for k in (2, 3, 4, 5):
        pool = kbox_seeded_pool(k, rng, count=8, include_perturbations=True)
        print(f"k={k}  pool_size={len(pool)}")
        H, V = pool[0]
        mc = max_clique_at_grid(H, V)
        print(f"  pristine triangle-free: {mc <= 2}  max_clique_at_grid={mc}")
        shuffled = shuffle_boxes(H, V, k, rng)
        mc2 = max_clique_at_grid(*shuffled)
        print(f"  shuffled triangle-free: {mc2 <= 2}  max_clique_at_grid={mc2}")
        swapped = swap_within_box(H, V, k, box=1, rng=rng)
        mc3 = max_clique_at_grid(*swapped)
        print(f"  swapped-within-box(1) max_clique_at_grid={mc3}")


def _save_smoke():
    """Verify save logic works end-to-end."""
    print("Save smoke test...")
    H, V = kbox_instance(3)
    tmp_dir = "/tmp/save_smoke_elites"
    fname = save_elite(k=3, r=0, score=1.5000, H=H, V=V,
                       elites_dir=tmp_dir, threshold=1.0, verbose=True)
    print(f"  save_elite returned: {fname}")
    if fname and os.path.exists(fname):
        with open(fname, "rb") as f:
            d = pickle.load(f)
        print(f"  re-loaded: gap={d['gap']} tf={d['triangle_free']} "
              f"mc={d['max_clique']} n={d['n']}")
        os.remove(fname)
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
        print("  save logic works.")
    else:
        print("  save FAILED.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--save-smoke", action="store_true")
    ap.add_argument("--verify-all", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--k-start", type=int, default=3)
    ap.add_argument("--k-end", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seeds-per-round", type=int, default=16)
    ap.add_argument("--local-time", type=float, default=2.0)
    ap.add_argument("--save-threshold", type=float, default=SAVE_THRESHOLD)
    ap.add_argument("--elites-dir", default=ELITES_DIR)
    ap.add_argument("--final-dir", default=FINAL_DIR)
    ap.add_argument("--no-reuse", action="store_true",
                    help="disable loading prior elites from --elites-dir")
    ap.add_argument("--reuse-count", type=int, default=8,
                    help="max prior elites to load per k")
    ap.add_argument("--reuse-min-gap", type=float, default=0.0,
                    help="only reuse prior elites with gap >= this")
    ap.add_argument("--reuse-tf-only", action="store_true",
                    help="only reuse triangle-free prior elites")
    ap.add_argument("--use-transformer", action="store_true",
                    help="enable TinyGPT proposer (trained on elite pool)")
    ap.add_argument("--xf-samples", type=int, default=8,
                    help="transformer samples per round")
    ap.add_argument("--xf-steps", type=int, default=40,
                    help="transformer training steps per round")
    ap.add_argument("--xf-batch", type=int, default=16,
                    help="transformer training batch size")
    ap.add_argument("--xf-temp", type=float, default=1.0,
                    help="transformer sampling temperature")
    ap.add_argument("--xf-top-p", type=float, default=0.9,
                    help="transformer sampling top-p")
    ap.add_argument("--xf-pool", type=int, default=128,
                    help="max elites kept in transformer training pool")
    ap.add_argument("--fast-parallel", action="store_true",
                    help="use process-pool parallel + LP-only local search "
                         "(recommended for large n; requires kbox_fast.py)")
    ap.add_argument("--fast-workers", type=int, default=None,
                    help="process-pool size (default: auto)")
    ap.add_argument("--fast-grb-threads", type=int, default=1,
                    help="Gurobi threads per worker (default 1 so cores "
                         "× workers ~= physical cores)")
    ap.add_argument("--fast-verify-top-k", type=int, default=8,
                    help="ILP-verify this many top LP elites per seed")
    ap.add_argument("--fast-verify-time-limit", type=float, default=60.0,
                    help="per-ILP verification time limit in seconds")
    ap.add_argument("--fast-neighbor-ilp-time", type=float, default=0.8,
                    help="per-ILP time limit during neighbor scoring "
                         "(short budget; verification uses the longer limit)")
    ap.add_argument("--allow-nontf", action="store_true",
                    help="allow non-triangle-free instances during search "
                         "(default is triangle-free only). Use this to let "
                         "local search drift off the 2-box manifold, which "
                         "is where clique-LP improvements like 1.5529 live.")
    args = ap.parse_args()

    if args.save_smoke:
        _save_smoke()
        return
    if args.smoke:
        _smoke()
        return
    if args.verify_all:
        print(f"{'k':>3} {'n':>6} {'alpha*':>8} {'alpha':>8} {'gap':>8} {'ok':>6}")
        for k in range(2, 8):
            ok = verify(k, verbose=False)
            print(f"{k:>3} {4*k*k:>6} {2*k*k:>8} {k*k+3*k-2:>8} "
                  f"{2*k*k/(k*k+3*k-2):>8.4f} {'PASS' if ok else 'FAIL':>6}")
        return
    if args.run:
        run_kbox_patternboost(
            k_start=args.k_start,
            k_end=args.k_end,
            rounds_per_k=args.rounds,
            seeds_per_round=args.seeds_per_round,
            local_time_per_seed=args.local_time,
            save_threshold=args.save_threshold,
            elites_dir=args.elites_dir,
            final_dir=args.final_dir,
            reuse_elites=not args.no_reuse,
            reuse_count=args.reuse_count,
            reuse_min_gap=args.reuse_min_gap,
            reuse_require_tf=args.reuse_tf_only,
            use_transformer=args.use_transformer,
            transformer_samples_per_round=args.xf_samples,
            transformer_train_steps_per_round=args.xf_steps,
            transformer_batch_size=args.xf_batch,
            transformer_temperature=args.xf_temp,
            transformer_top_p=args.xf_top_p,
            transformer_elite_pool_size=args.xf_pool,
            fast_parallel=args.fast_parallel,
            fast_n_workers=args.fast_workers,
            fast_grb_threads=args.fast_grb_threads,
            fast_verify_top_k=args.fast_verify_top_k,
            fast_verify_time_limit=args.fast_verify_time_limit,
            fast_neighbor_ilp_time=args.fast_neighbor_ilp_time,
            require_triangle_free=not args.allow_nontf,
        )
        return
    _smoke()


if __name__ == "__main__":
    main()
