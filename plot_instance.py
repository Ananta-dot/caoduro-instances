#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_instance.py
================

Visualize a saved MISR instance pickle: rectangles, independent-set
highlighting, and optionally the intersection graph as a node-link diagram.

Reads a pickle of the form written by the patched run_kbox_patternboost:
    {"k": int, "round": int, "H": list[int], "V": list[int],
     "gap": float, "triangle_free": bool, "max_clique": int}

Usage:
    python3 plot_instance.py elites_above_1.5/k9_r1_gap1.5381.pkl
    python3 plot_instance.py path/to/file.pkl --out my_plot.pdf
    python3 plot_instance.py path/to/file.pkl --views rects selected graph
    python3 plot_instance.py path/to/file.pkl --format png --dpi 200
    python3 plot_instance.py path/to/file.pkl --no-ilp    # skip ILP (fast)

Requires: matplotlib, gurobipy (for the ILP to identify selected rects).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

from kbox_misr import build_rects, grid_points, covers_grid_closed


# =========================================================================== #
# Loading                                                                      #
# =========================================================================== #


def load_pickle(path: str) -> Dict:
    with open(path, "rb") as f:
        d = pickle.load(f)
    required = {"H", "V"}
    missing = required - set(d.keys())
    if missing:
        raise KeyError(f"pickle missing keys: {missing}")
    return d


# =========================================================================== #
# ILP (to identify selected rectangles)                                        #
# =========================================================================== #


def solve_ilp_selection(H: List[int], V: List[int],
                        time_limit: float = 120.0
                        ) -> Tuple[Optional[float], Optional[List[int]]]:
    """
    Solve the integer clique-constrained MISR to obtain both alpha and the
    set of selected rectangles. Returns (ilp_value, selected_indices_0based).

    Returns (None, None) if Gurobi is unavailable or fails.
    """
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
    y = m.addVars(n, vtype=GRB.BINARY, name="y")
    m.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if len(S) >= 2:
            m.addConstr(gp.quicksum(y[i] for i in S) <= 1)
    m.optimize()

    if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        print(f"[warn] ILP status = {m.status}", file=sys.stderr)
        return None, None
    if m.SolCount == 0:
        return None, None

    ilp_value = float(m.objVal)
    selected = [i for i in range(n) if y[i].X > 0.5]
    return ilp_value, selected


# =========================================================================== #
# Heuristic: guess box membership via spatial clustering                       #
# =========================================================================== #
#
# For a pristine M_k the 4k^2 rectangles cluster into k groups along the
# diagonal. We can find them by k-means on centroids (with k unknown).
# This is purely decorative — it assigns colors in the "rects" plot.


def guess_box_groups(rects, k_hint: Optional[int] = None, n_max: int = 20
                     ) -> np.ndarray:
    """
    Return an array of integer group labels, one per rectangle, derived from
    spatial clustering of centroids along the main diagonal.

    If k_hint is provided and fits the number of rectangles, use it.
    Otherwise pick the k that minimizes within-group variance among 1..n_max
    via the elbow heuristic on 1D projection onto the main diagonal.
    """
    centroids = np.array([
        [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        for ((x1, x2), (y1, y2)) in rects
    ])
    # project onto main diagonal direction (1, 1) normalized
    proj = centroids.sum(axis=1)
    order = np.argsort(proj)

    n = len(rects)
    if k_hint is not None:
        k = k_hint
    else:
        # heuristic: if n is close to a multiple of 4*k^2 for some k, use that k
        k = max(1, int(np.sqrt(n / 4)))

    k = max(1, min(k, n_max, n))
    # split sorted projection into k equal-size consecutive chunks
    groups = np.zeros(n, dtype=int)
    chunk = n // k
    remainder = n - chunk * k
    idx = 0
    for g in range(k):
        size = chunk + (1 if g < remainder else 0)
        groups[order[idx:idx + size]] = g
        idx += size
    return groups


# =========================================================================== #
# Plotting                                                                     #
# =========================================================================== #


def _rect_patches(rects, groups, selected_set: set,
                  expand_thickness: float = 0.0, edge_only: bool = True):
    """
    Build a list of matplotlib Rectangle patches with styling baked in.
    expand_thickness: extra padding in axis units added to rectangles for
        visibility. Purely display-only.
    """
    patches_out = []
    facecolors = []
    edgecolors = []
    linewidths = []
    zorders = []
    cmap = plt.get_cmap("tab20")

    # Compute global axis span to scale expansion proportionally if expand<0.
    all_x1 = np.array([r[0][0] for r in rects], dtype=float)
    all_x2 = np.array([r[0][1] for r in rects], dtype=float)
    all_y1 = np.array([r[1][0] for r in rects], dtype=float)
    all_y2 = np.array([r[1][1] for r in rects], dtype=float)
    span_x = (all_x2.max() - all_x1.min())
    span_y = (all_y2.max() - all_y1.min())
    pad = expand_thickness if expand_thickness > 0 else 0.0

    n_groups = int(groups.max()) + 1 if len(groups) else 1
    for i, ((x1, x2), (y1, y2)) in enumerate(rects):
        w = (x2 - x1) + pad
        h = (y2 - y1) + pad
        x = x1 - pad / 2
        y = y1 - pad / 2
        is_sel = i in selected_set
        color = cmap(groups[i] % 20) if n_groups > 1 else "#3a7ca5"
        rect = mpatches.Rectangle((x, y), w, h)
        patches_out.append(rect)
        if is_sel:
            facecolors.append((1.0, 0.0, 0.0, 0.10))  # translucent red fill
            edgecolors.append((0.85, 0.05, 0.05, 1.0))
            linewidths.append(1.8)
            zorders.append(2)
        else:
            # translucent fill tinted by group
            r_, g_, b_, _ = color if isinstance(color, tuple) else (0.2, 0.4, 0.6, 1)
            facecolors.append((r_, g_, b_, 0.06) if edge_only else
                              (r_, g_, b_, 0.3))
            edgecolors.append((r_, g_, b_, 0.85))
            linewidths.append(0.6)
            zorders.append(1)
    return patches_out, facecolors, edgecolors, linewidths, zorders


def plot_rectangles(H, V, selected: Optional[List[int]],
                    title: str, k_hint: Optional[int] = None,
                    expand: float = 0.0,
                    show_ids: bool = False,
                    ax=None):
    rects = build_rects(H, V)
    selected_set = set(selected) if selected else set()
    groups = guess_box_groups(rects, k_hint=k_hint)
    patches_out, fcs, ecs, lws, zs = _rect_patches(
        rects, groups, selected_set, expand_thickness=expand, edge_only=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 11))
    for p, fc, ec, lw, z in zip(patches_out, fcs, ecs, lws, zs):
        p.set_facecolor(fc); p.set_edgecolor(ec)
        p.set_linewidth(lw); p.set_zorder(z)
        ax.add_patch(p)
    # set axis limits
    all_x = [r[0][0] for r in rects] + [r[0][1] for r in rects]
    all_y = [r[1][0] for r in rects] + [r[1][1] for r in rects]
    pad_x = 0.02 * (max(all_x) - min(all_x) + 1)
    pad_y = 0.02 * (max(all_y) - min(all_y) + 1)
    ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
    ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # match the misr_122.png convention (origin at top-left)
    ax.set_title(title)
    ax.set_xlabel("H positions (indices)")
    ax.set_ylabel("V positions (indices)")
    ax.grid(True, linestyle=":", alpha=0.3)

    if show_ids and len(rects) <= 80:
        for i, ((x1, x2), (y1, y2)) in enumerate(rects):
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(i + 1),
                    ha="center", va="center", fontsize=7,
                    alpha=0.85 if i in selected_set else 0.55)

    # legend
    leg = [
        Line2D([0], [0], color="red", lw=1.8, label=f"Selected  ({len(selected_set)})"),
        Line2D([0], [0], color="gray", lw=0.6,
               label=f"Not selected  ({len(rects) - len(selected_set)})"),
    ]
    ax.legend(handles=leg, loc="upper right", fontsize=9)
    return ax


def plot_graph(H, V, selected: Optional[List[int]], title: str, ax=None,
               max_edges_drawn: int = 50_000):
    """
    Node-link plot of the intersection graph. Nodes = rectangle centroids,
    edges between intersecting pairs. Selected nodes in red.

    For large n this can be slow; max_edges_drawn caps rendering.
    """
    rects = build_rects(H, V)
    n = len(rects)
    centroids = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]
                          for ((x1, x2), (y1, y2)) in rects])
    selected_set = set(selected) if selected else set()

    # edges via numpy vector ops (same trick as kbox_parallel.vec_actual_edges)
    x1 = np.array([r[0][0] for r in rects])
    x2 = np.array([r[0][1] for r in rects])
    y1 = np.array([r[1][0] for r in rects])
    y2 = np.array([r[1][1] for r in rects])
    xo = (np.maximum(x1[:, None], x1[None, :]) <=
          np.minimum(x2[:, None], x2[None, :]))
    yo = (np.maximum(y1[:, None], y1[None, :]) <=
          np.minimum(y2[:, None], y2[None, :]))
    adj = xo & yo
    np.fill_diagonal(adj, False)
    iu, ju = np.triu_indices(n, k=1)
    mask = adj[iu, ju]
    edges = list(zip(iu[mask].tolist(), ju[mask].tolist()))
    total_edges = len(edges)
    if total_edges > max_edges_drawn:
        import random
        random.Random(0).shuffle(edges)
        edges = edges[:max_edges_drawn]

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 11))
    # draw edges (thin, gray)
    for i, j in edges:
        is_is_edge = (i in selected_set) and (j in selected_set)  # always False if IS is valid
        ax.plot([centroids[i, 0], centroids[j, 0]],
                [centroids[i, 1], centroids[j, 1]],
                color=("#ff3333" if is_is_edge else "#bbbbbb"),
                lw=0.3, alpha=0.5, zorder=1)
    # nodes
    sel_mask = np.array([i in selected_set for i in range(n)])
    ax.scatter(centroids[~sel_mask, 0], centroids[~sel_mask, 1],
               s=14, c="#555", edgecolor="white", lw=0.3, zorder=2,
               label=f"Not selected ({(~sel_mask).sum()})")
    if sel_mask.any():
        ax.scatter(centroids[sel_mask, 0], centroids[sel_mask, 1],
                   s=28, c="red", edgecolor="white", lw=0.5, zorder=3,
                   label=f"Selected ({sel_mask.sum()})")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title + f"  (|E|={total_edges})")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    return ax


# =========================================================================== #
# Driver                                                                       #
# =========================================================================== #


def make_figure(data: Dict, out: str,
                views: Sequence[str], dpi: int = 200,
                expand: float = 0.0, show_ids: bool = False,
                do_ilp: bool = True, ilp_time_limit: float = 120.0):
    H, V = data["H"], data["V"]
    n = max(max(H), max(V))
    gap = data.get("gap", None)
    tf = data.get("triangle_free", None)
    mc = data.get("max_clique", None)
    k = data.get("k", None)
    round_ = data.get("round", None)

    subtitle_bits = [f"n={n}"]
    if k is not None:
        subtitle_bits.append(f"k={k}")
    if round_ is not None:
        subtitle_bits.append(f"round={round_}")
    if gap is not None:
        subtitle_bits.append(f"gap={gap:.4f}")
    if tf is not None:
        subtitle_bits.append(f"triangle_free={tf}")
    if mc is not None:
        subtitle_bits.append(f"max_clique={mc}")
    subtitle = "  ".join(subtitle_bits)

    selected: Optional[List[int]] = None
    ilp_val: Optional[float] = None
    if do_ilp:
        print("Solving ILP to identify selected rectangles...")
        ilp_val, selected = solve_ilp_selection(H, V, time_limit=ilp_time_limit)
        if ilp_val is not None:
            print(f"  ILP = {ilp_val:.0f},  |S| = {len(selected) if selected else 0}")

    n_panels = len(views)
    fig, axes = plt.subplots(1, n_panels, figsize=(11 * n_panels, 11),
                             squeeze=False)
    axes = axes[0]

    for ax, view in zip(axes, views):
        if view == "rects":
            plot_rectangles(H, V, selected,
                            title=f"All rectangles — {subtitle}",
                            k_hint=k, expand=expand, show_ids=show_ids, ax=ax)
        elif view == "selected":
            if selected is None:
                ax.text(0.5, 0.5, "(ILP not solved; pass --ilp or increase time)",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            else:
                # only-selected view: filter to selected rects
                plot_rectangles(H, V, selected,
                                title=f"Selected IS only — {subtitle}",
                                k_hint=k, expand=expand, ax=ax)
                # additionally grey out non-selected by overlaying white alpha
                pass
        elif view == "graph":
            plot_graph(H, V, selected,
                       title=f"Intersection graph — {subtitle}", ax=ax)
        else:
            ax.text(0.5, 0.5, f"unknown view: {view}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle", help="path to pickle file")
    ap.add_argument("--out", default=None,
                    help="output path (default: <pickle>.pdf)")
    ap.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf",
                    help="output format if --out not set")
    ap.add_argument("--views", nargs="+",
                    default=["rects", "graph"],
                    choices=["rects", "selected", "graph"],
                    help="which panels to render")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--expand", type=float, default=0.0,
                    help="visual padding per rectangle (display-only)")
    ap.add_argument("--show-ids", action="store_true",
                    help="draw rectangle indices (only for n <= 80)")
    ap.add_argument("--no-ilp", action="store_true",
                    help="skip ILP (no selected highlighting)")
    ap.add_argument("--ilp-time", type=float, default=120.0,
                    help="Gurobi ILP time limit in seconds")
    args = ap.parse_args()

    data = load_pickle(args.pickle)

    if args.out is None:
        base = os.path.splitext(args.pickle)[0]
        args.out = f"{base}.{args.format}"

    make_figure(
        data, args.out,
        views=args.views,
        dpi=args.dpi,
        expand=args.expand,
        show_ids=args.show_ids,
        do_ilp=not args.no_ilp,
        ilp_time_limit=args.ilp_time,
    )


if __name__ == "__main__":
    main()
