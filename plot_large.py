#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_large.py — clean visualization of large MISR instances (n >= 100).

Produces a 4-panel figure designed to make a dense rectangle instance
readable at n=324:

  Panel A: "Anatomy" — all rectangles, selected in red, rest light gray.
           Selected rectangles get thick outlines; non-selected are hair-thin.
  Panel B: "Selected only" — just the independent set on its own, so you can
           see which rectangles won without clutter.
  Panel C: "Horizontal vs Vertical" — rectangles split by orientation
           (tall-thin = vertical segments; wide-flat = horizontal), in two
           sub-plots stacked side by side within the panel.
  Panel D: "Box structure" — rectangles colored by inferred k-box membership.
           Selected ones outlined bold. Reveals whether search drifted off
           the k-box structure or only perturbed it locally.

Usage:
  python3 plot_large.py elites_above_threshold/k9_r2_gap1.5429_tf1.pkl
  python3 plot_large.py path/to.pkl --out myplot.pdf --format pdf
  python3 plot_large.py path/to.pkl --no-ilp  # skip ILP, no red highlights

The ILP solve identifies selected rectangles. At n=324 a good ILP takes
~30s; at n=400 a few minutes. Use --ilp-time to cap it.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from typing import List, Optional, Sequence, Set, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

from kbox_misr import build_rects, grid_points, covers_grid_closed


# =========================================================================== #
# Loading and ILP                                                              #
# =========================================================================== #


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        d = pickle.load(f)
    if "H" not in d or "V" not in d:
        raise KeyError(f"{path}: pickle missing H or V")
    return d


def solve_ilp_selection(H: List[int], V: List[int],
                        time_limit: float = 120.0
                        ) -> Tuple[Optional[float], Optional[List[int]]]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        print(f"[warn] gurobipy unavailable ({e}); skipping ILP.", file=sys.stderr)
        return None, None

    rects = build_rects(H, V)
    pts = grid_points(rects)
    covers = covers_grid_closed(rects, pts)
    n = len(rects)

    m = gp.Model("misr_ilp")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    m.setParam("MIPFocus", 1)  # focus on feasible solutions
    y = m.addVars(n, vtype=GRB.BINARY, name="y")
    m.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if len(S) >= 2:
            m.addConstr(gp.quicksum(y[i] for i in S) <= 1)

    print(f"  solving ILP (n={n}, time_limit={time_limit}s)...")
    t0 = time.perf_counter()
    m.optimize()
    dt = time.perf_counter() - t0

    if m.SolCount == 0:
        return None, None

    ilp_val = float(m.objVal)
    selected = [i for i in range(n) if y[i].X > 0.5]
    status_str = "optimal" if m.status == 2 else f"status={m.status}"
    print(f"  ILP: value={ilp_val:.0f}, |S|={len(selected)}, {status_str} [{dt:.1f}s]")
    return ilp_val, selected


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #


def classify_orientation(rects) -> np.ndarray:
    """
    Return an array: 0 = horizontal (wider than tall), 1 = vertical, 2 = square-ish.
    """
    out = np.empty(len(rects), dtype=int)
    for i, ((x1, x2), (y1, y2)) in enumerate(rects):
        w = x2 - x1
        h = y2 - y1
        if w > 1.5 * h:
            out[i] = 0  # horizontal
        elif h > 1.5 * w:
            out[i] = 1  # vertical
        else:
            out[i] = 2  # square-ish
    return out


def infer_box_groups(rects, k_hint: Optional[int] = None) -> np.ndarray:
    """
    Cluster rectangles into k groups along the main diagonal. Uses projection
    onto the (1,1) direction and equal-size bins.
    """
    n = len(rects)
    if n == 0:
        return np.zeros(0, dtype=int)
    centroids = np.array([
        [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        for ((x1, x2), (y1, y2)) in rects
    ])
    proj = centroids.sum(axis=1)
    order = np.argsort(proj)

    if k_hint is None or k_hint < 1:
        k = max(1, int(round(np.sqrt(n / 4))))
    else:
        k = k_hint
    k = max(1, min(k, n))

    groups = np.zeros(n, dtype=int)
    chunk = n // k
    remainder = n - chunk * k
    idx = 0
    for g in range(k):
        size = chunk + (1 if g < remainder else 0)
        groups[order[idx:idx + size]] = g
        idx += size
    return groups


def _draw_rects(ax, rects, selected_set, *,
                non_selected_color=(0.70, 0.70, 0.70),
                non_selected_alpha=0.35,
                non_selected_lw=0.4,
                selected_color=(0.85, 0.10, 0.10),
                selected_alpha=1.0,
                selected_lw=1.8,
                selected_fill_alpha=0.08,
                group_colors=None,
                group_labels=None):
    """Core drawing routine. group_colors: optional (n,) array of (r,g,b,a)."""
    patches_selected = []
    patches_other = []
    fc_selected = []
    fc_other = []
    ec_selected = []
    ec_other = []

    for i, ((x1, x2), (y1, y2)) in enumerate(rects):
        w = x2 - x1
        h = y2 - y1
        rect = mpatches.Rectangle((x1, y1), w, h)
        if i in selected_set:
            patches_selected.append(rect)
            if group_colors is not None:
                r, g, b, _ = group_colors[i]
                fc_selected.append((r, g, b, selected_fill_alpha))
                ec_selected.append(selected_color)
            else:
                fc_selected.append((1, 0, 0, selected_fill_alpha))
                ec_selected.append(selected_color)
        else:
            patches_other.append(rect)
            if group_colors is not None:
                r, g, b, _ = group_colors[i]
                fc_other.append((r, g, b, 0.10))
                ec_other.append((r, g, b, 0.60))
            else:
                fc_other.append((*non_selected_color, 0.08))
                ec_other.append((*non_selected_color, non_selected_alpha))

    # non-selected first (so selected sit on top)
    if patches_other:
        pc = PatchCollection(patches_other, match_original=False)
        pc.set_facecolor(fc_other)
        pc.set_edgecolor(ec_other)
        pc.set_linewidth(non_selected_lw)
        pc.set_zorder(1)
        ax.add_collection(pc)

    if patches_selected:
        pc = PatchCollection(patches_selected, match_original=False)
        pc.set_facecolor(fc_selected)
        pc.set_edgecolor(ec_selected)
        pc.set_linewidth(selected_lw)
        pc.set_zorder(3)
        ax.add_collection(pc)


def _set_limits(ax, rects, pad_frac=0.02):
    if not rects:
        return
    all_x = [r[0][0] for r in rects] + [r[0][1] for r in rects]
    all_y = [r[1][0] for r in rects] + [r[1][1] for r in rects]
    lo_x, hi_x = min(all_x), max(all_x)
    lo_y, hi_y = min(all_y), max(all_y)
    px = pad_frac * max(1, hi_x - lo_x)
    py = pad_frac * max(1, hi_y - lo_y)
    ax.set_xlim(lo_x - px, hi_x + px)
    ax.set_ylim(lo_y - py, hi_y + py)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(True, linestyle=":", alpha=0.3)


# =========================================================================== #
# Panels                                                                       #
# =========================================================================== #


def panel_anatomy(ax, rects, selected_set, title):
    _draw_rects(ax, rects, selected_set)
    _set_limits(ax, rects)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("H index"); ax.set_ylabel("V index")
    n_sel = len(selected_set)
    n_not = len(rects) - n_sel
    leg = [
        Line2D([0], [0], color=(0.85, 0.10, 0.10), lw=1.8,
               label=f"Selected ({n_sel})"),
        Line2D([0], [0], color=(0.55, 0.55, 0.55), lw=0.4,
               label=f"Not selected ({n_not})"),
    ]
    ax.legend(handles=leg, loc="upper right", fontsize=9)


def panel_selected_only(ax, rects, selected_set, title):
    """Show ONLY the selected rectangles. Everything else at very low alpha."""
    # First pass: all in light gray background
    for i, ((x1, x2), (y1, y2)) in enumerate(rects):
        if i in selected_set:
            continue
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            facecolor="none", edgecolor=(0.85, 0.85, 0.85, 0.4),
            linewidth=0.25, zorder=1))
    # Second pass: selected in bold red
    for i, ((x1, x2), (y1, y2)) in enumerate(rects):
        if i not in selected_set:
            continue
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            facecolor=(1, 0, 0, 0.15), edgecolor=(0.75, 0.05, 0.05),
            linewidth=1.6, zorder=2))
    _set_limits(ax, rects)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("H index"); ax.set_ylabel("V index")


def panel_orientation(ax, rects, selected_set, title):
    """Color-code rectangles by orientation."""
    orient = classify_orientation(rects)
    colors = np.zeros((len(rects), 4))
    # horizontal: teal; vertical: orange; square: purple
    palette = {
        0: (0.10, 0.55, 0.55, 1.0),
        1: (0.85, 0.45, 0.10, 1.0),
        2: (0.55, 0.25, 0.65, 1.0),
    }
    for i, o in enumerate(orient):
        colors[i] = palette[o]
    _draw_rects(ax, rects, selected_set, group_colors=colors)
    _set_limits(ax, rects)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("H index"); ax.set_ylabel("V index")
    n_h = int((orient == 0).sum())
    n_v = int((orient == 1).sum())
    n_s = int((orient == 2).sum())
    leg = [
        Line2D([0], [0], color=palette[0][:3], lw=2, label=f"Horizontal ({n_h})"),
        Line2D([0], [0], color=palette[1][:3], lw=2, label=f"Vertical ({n_v})"),
        Line2D([0], [0], color=palette[2][:3], lw=2, label=f"Square-ish ({n_s})"),
        Line2D([0], [0], color=(0.85, 0.10, 0.10), lw=2, ls="--",
               label=f"Selected outlined red"),
    ]
    ax.legend(handles=leg, loc="upper right", fontsize=8)


def panel_boxes(ax, rects, selected_set, k_hint, title):
    """Color-code by inferred k-box membership."""
    groups = infer_box_groups(rects, k_hint=k_hint)
    k_eff = int(groups.max()) + 1
    cmap = plt.get_cmap("tab10" if k_eff <= 10 else "tab20")
    colors = np.array([cmap(g % cmap.N) for g in groups])
    _draw_rects(ax, rects, selected_set, group_colors=colors)
    _set_limits(ax, rects)
    ax.set_title(title + f" (inferred {k_eff} boxes)", fontsize=11)
    ax.set_xlabel("H index"); ax.set_ylabel("V index")
    # count selected per inferred box
    sel_per_box = np.zeros(k_eff, dtype=int)
    for i in selected_set:
        sel_per_box[groups[i]] += 1
    leg = []
    for g in range(k_eff):
        col = cmap(g % cmap.N)
        leg.append(Line2D([0], [0], color=col, lw=2,
                          label=f"Box {g+1} (IS count: {sel_per_box[g]})"))
    ax.legend(handles=leg, loc="upper right", fontsize=7)


# =========================================================================== #
# Driver                                                                       #
# =========================================================================== #


def make_figure(data: dict, out: str, dpi: int = 180,
                do_ilp: bool = True, ilp_time_limit: float = 120.0):
    H, V = data["H"], data["V"]
    rects = build_rects(H, V)
    n = len(rects)
    k = data.get("k", None)
    gap = data.get("gap", None)
    tf = data.get("triangle_free", None)
    mc = data.get("max_clique", None)
    r = data.get("round", None)

    # solve ILP
    selected: Optional[List[int]] = None
    ilp_val: Optional[float] = None
    if do_ilp:
        ilp_val, selected = solve_ilp_selection(H, V, time_limit=ilp_time_limit)
    selected_set = set(selected) if selected else set()

    # build header
    hdr_bits = [f"n={n}"]
    if k is not None: hdr_bits.append(f"k={k}")
    if r is not None: hdr_bits.append(f"round={r}")
    if gap is not None: hdr_bits.append(f"gap={gap:.4f}")
    if tf is not None: hdr_bits.append(f"triangle_free={tf}")
    if mc is not None: hdr_bits.append(f"max_clique={mc}")
    if ilp_val is not None: hdr_bits.append(f"ILP={ilp_val:.0f}")
    header = "  ·  ".join(hdr_bits)

    # figure with 4 panels: 2x2
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    panel_anatomy(ax_a, rects, selected_set,
                  "A. Anatomy — all rectangles, IS highlighted")
    panel_selected_only(ax_b, rects, selected_set,
                        "B. Selected-only — just the independent set")
    panel_orientation(ax_c, rects, selected_set,
                      "C. Orientation — horizontal (teal) / vertical (orange)")
    panel_boxes(ax_d, rects, selected_set, k_hint=k,
                title="D. Box structure — colored by inferred k-box")

    fig.suptitle(header, fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle", help="path to saved MISR pickle")
    ap.add_argument("--out", default=None,
                    help="output path (default: <pickle>.png)")
    ap.add_argument("--format", choices=["pdf", "png", "svg"], default="png")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--no-ilp", action="store_true",
                    help="skip ILP (no red IS highlights)")
    ap.add_argument("--ilp-time", type=float, default=120.0,
                    help="ILP time limit in seconds")
    args = ap.parse_args()

    data = load_pickle(args.pickle)

    if args.out is None:
        base = os.path.splitext(args.pickle)[0]
        args.out = f"{base}_4panel.{args.format}"

    make_figure(data, args.out, dpi=args.dpi,
                do_ilp=not args.no_ilp,
                ilp_time_limit=args.ilp_time)


if __name__ == "__main__":
    main()
