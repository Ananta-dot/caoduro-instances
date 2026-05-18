# MISR Integrality Gap Search via k-box Seeding + Iterative Extension

Computational search for rectangle intersection graphs with large LP/ILP
integrality gap on the Maximum Independent Set of Rectangles (MISR) problem.

Builds on two pieces of prior work:

- **Chalermsook & Chuzhoy (2008)** — proved an asymptotic `3/2` integrality
  gap lower bound for MISR (`rectanglesfull.pdf`).
- **Caoduro, Cslovjecsek, Pilipczuk, Węgrzycki (2022)** — proved the integrality
  gap approaches `2` in the limit using axis-parallel *segments* (not general
  rectangles). The construction is `M_k`, with `4k²` segments and finite-`n`
  gap `2k²/(k²+3k−2)` (`2205.15189v1.pdf`).

This repo's contribution: realize Caoduro et al.'s construction as thin
rectangles, then computationally search for *strictly stronger* finite-`n`
rectangle instances. We have verified improvements at four consecutive
values of `k`.

The metric reported throughout is **clique LP / ILP**, where the clique LP
is the relaxation with one constraint per intersection grid-point (= one
constraint per maximal clique by Helly's theorem on rectangles). This is
strictly stronger than the edge LP relaxation. Every value below is a
Gurobi computation with `Method=3` LP and `MIPFocus=2` ILP, ILP proved
optimal (mip_gap = 0).

---

## Verified results

### Triangle-free improvements

These are instances where the intersection graph has no triangle (max
clique ≤ 2). For these instances the clique LP equals the edge LP, so
they are also Caoduro-comparable in the paper's metric.

| k | n | LP | ILP | clique LP / ILP | Caoduro pristine | Δ |
|---|---|---|---|---|---|---|
| 9  | 324 | 162   | 104 | **1.5577** | 1.5283 | +0.0294 |
| 10 | 400 | 200.5 | 128 | **1.5664** | 1.5625 | +0.0039 |
| 11 | 484 | 242.5 | 152 | **1.5954** | 1.5921 | +0.0033 |
| 12 | 576 |  —    |  —  | (no triangle-free improvement found in 100 trials) | 1.6180 | 0 |

The k=9 result reduces α (106 → 104). The k=10 and k=11 results leave α
unchanged but increase the LP slightly (LP = n/2 + 0.5), which means the
edge LP is no longer tight at n/2. That's a structurally different kind
of improvement than the k=9 one.

### Triangle-tolerant improvements (max clique = 3)

These instances allow triangles. Clique LP penalizes each triangle by 0.5,
so these are stricter results than the edge-LP analogues.

| k | n | LP | ILP | clique LP / ILP | Caoduro pristine | Δ |
|---|---|---|---|---|---|---|
| 10 | 400 | 199.5 | 127 | **1.5709** | 1.5625 | +0.0084 |
| 11 | 484 | 241   | 151 | **1.5960** | 1.5921 | +0.0039 |
| 11 | 484 | 240   | 150 | **1.6000** | 1.5921 | +0.0079 |
| 11 | 484 | 239   | 149 | **1.6040** | 1.5921 | +0.0119 |
| 11 | 484 | 238   | 148 | **1.6081** | 1.5921 | +0.0160 |
| 12 | 576 | 286.5 | 177 | **1.6186** | 1.6180 | +0.0006 |

The k=11 chain (+1, +2, +3, +4) was built iteratively: pristine M_11 → +1
via two random "long swaps", then each subsequent +δ from the previous
elite via a single mutation.

The k=12 +1 result is real but barely above pristine in clique-LP terms
(+0.0006). Pushing k=12 further in this metric requires longer in-loop
ILP solves to escape the phantom regime described below.

All saved elites are in `elites_above_threshold/`. Pristine and modified
4-panel plots are in `plots/`.

---

## What's the move that produces the improvement?

**Paired endpoint-swap on M_k.** Pristine `M_k` contains "thin segment"
rectangles arranged in a diagonal of `k` boxes. Many pairs `(H_seg, V_seg)`
share an endpoint (an H-segment ends where a V-segment begins). The
modification:

1. Pick such a pair.
2. Swap the shared endpoint between them — H gains the V's other endpoint,
   V loses its old endpoint.
3. Result: H grows from height 1 to a substantial range; V correspondingly
   shrinks. The intersection graph gains new edges, breaking one rectangle
   from the maximum independent set.

At k=11 we apply this move four times to disjoint pairs to get +4 in α
(α: 152 → 148). The resulting instance is no longer triangle-free
(max clique = 3) but the edge LP remains tight at n/2.

Run `python3 geom_outliers.py <pickle>` to see the actual modified
rectangles for any saved elite.

---

## Pipeline at a glance

```
   ┌────────────────┐   ┌──────────────────┐
   │ kbox_misr.py   │   │ extend_experiment │  ← directed mutation driver
   │ (pristine M_k) │   │      .py          │     (this repo's tool)
   └────────┬───────┘   └────────┬──────────┘
            │                    │
            ▼                    ▼
   ┌────────────────┐   ┌──────────────────┐
   │ kbox_search.py │   │  extend_hits/    │  ← raw experiment outputs
   │ (PatternBoost  │   └────────┬──────────┘
   │  search)       │            │
   └────────┬───────┘            ▼ verify
            │           ┌──────────────────┐
            ▼           │ verify_instance  │  ← clique LP, edge LP,
   ┌──────────────────┐ │      .py         │     proved-optimal ILP
   │ elites_above_    │ └────────┬──────────┘
   │   threshold/     │          │
   └──────────────────┘          ▼
                        ┌──────────────────┐
                        │  geom_outliers   │  ← structural diff vs pristine
                        │      .py         │
                        └──────────────────┘
```

---

## File map

### Core pipeline

| File | Role |
|---|---|
| `mistr_runner.py` | Library, not run directly. Original PatternBoost code. Imported by `kbox_search.py`. |
| `kbox_misr.py` | Generates pristine M_k as thin rectangles. Self-verifies against paper's combinatorics. |
| `kbox_search.py` | PatternBoost-style transformer-guided local search. Builds k-box seeds, reloads elite pickles, drives local search. |
| `kbox_fast.py` | Parallel LP+short-ILP scored local search with proved-optimal verification. |
| `kbox_parallel.py` | Vectorized triangle-free check (~60-80x speedup at n ≥ 100) and multiprocessing wrappers. |

### Directed mutation experiment

| File | Role |
|---|---|
| `extend_experiment.py` | The driver behind the verified results. Applies long-distance segment extensions or random label swaps to pristine M_k or to a saved seed. Supports `--workers N` for parallelism, `--triangle-free-only` to filter mutations that introduce triangles, `--verify-ilp-time T` to re-solve any in-loop hit with a longer ILP time limit (essential at large n to avoid phantom hits). |
| `geom_diff.py` | Geometry-only multiset diff between any pickle and pristine M_k. Bypasses the canonicalize-relabel artifact in `diff_vs_pristine.py`. |
| `geom_outliers.py` | Filters the geom_diff output to just the substantively-modified rectangles (L1 ≥ threshold). Surfaces the 1–4 actual structural changes per result. |
| `merge_split_moves.py` | Larger structural moves for the search (merge/split/swap). Used by `kbox_search.py`. |

### Verification + diagnostics

| File | Role |
|---|---|
| `verify_instance.py` | Three-way verification on any pickle: clique LP, edge LP, full ILP with proof of optimality. The clique LP is the metric we report; edge LP is included as a diagnostic (tells us whether the instance is still α\* = n/2). Always run with ≥ 600s time limit at n ≥ 400; 1800s for n ≥ 576. |
| `diff_vs_pristine.py` | Earlier diff tool. Compares by label after canonicalization, which produces misleading "291/324 changed" output. Superseded by `geom_diff.py` / `geom_outliers.py`. |
| `lift_elite.py` | Cross-k seed lifting. Re-encode an elite at one k as a seed at the next k. |
| `inspect_instance.py` | Per-rectangle LP/ILP solution dump. |
| `enumerate_perturbations.py` | Brute-force single-perturbation enumerator. |

### Plotting

| File | Role |
|---|---|
| `plot_large.py` | 4-panel visualization for large-n pickles: anatomy, IS-only, orientation, box structure. |
| `plot_instance.py` | 2-panel quick preview for small-n. |

### Reference

| File | Role |
|---|---|
| `rectanglesfull.pdf` | Chalermsook & Chuzhoy 2008. |
| `2205.15189v1.pdf` | Caoduro et al. 2022. |
| `kbox_instances.json` | Concrete k=3, 5, 10 instances as JSON. |

---

## How to reproduce the verified results

### Sanity check the pristine baseline

```bash
python3 kbox_misr.py --sweep             # structural verification
python3 kbox_misr.py --k 11 --gurobi     # confirm pristine M_11 lp=242, ilp=152
```

### Reproduce the k=11 +4 chain (triangle-tolerant)

```bash
# Step 1: pristine → +1 (clique gap 1.5960)
python3 extend_experiment.py --k 11 --mode directed --trials 25 \
    --seed 1 --multi 2 --ilp-time 15

# Step 2: +1 → +2 (1.6000)
python3 extend_experiment.py --k 11 --mode directed --trials 25 \
    --seed 3 --multi 1 --ilp-time 20 \
    --seed-pickle elites_above_threshold/k11_extend_gap1.5960_tf0.pkl

# Step 3: +2 → +3 (1.6040)
python3 extend_experiment.py --k 11 --mode directed --trials 25 \
    --seed 4 --multi 1 --ilp-time 20 \
    --seed-pickle elites_above_threshold/k11_extend_p2_gap1.6000_tf0.pkl

# Step 4: +3 → +4 (1.6081)
python3 extend_experiment.py --k 11 --mode directed --trials 25 \
    --seed 5 --multi 1 --ilp-time 20 \
    --seed-pickle elites_above_threshold/k11_extend_p3_gap1.6040_tf0.pkl
```

Each step takes ~16 minutes on Apple M4 Max. Hit rate ~4-12% per batch.

### Reproduce the triangle-free k=10 and k=11 results

```bash
# k=10 triangle-free (LP-only mechanism, gap 1.5664)
python3 extend_experiment.py --k 10 --mode random --trials 500 \
    --seed 1 --multi 1 --workers 8 --triangle-free-only --ilp-time 15

# k=11 triangle-free (LP-only mechanism, gap 1.5954)
python3 extend_experiment.py --k 11 --mode random --trials 100 \
    --seed 7 --multi 1 --workers 8 --triangle-free-only \
    --ilp-time 15 --verify-ilp-time 250
```

### Verify any saved result

```bash
python3 verify_instance.py elites_above_threshold/k11_extend_p4_gap1.6081_tf0.pkl 600
```

Expected for the k=11 +4: clique_LP = 238, ILP = 148 proved optimal,
clique_LP/ILP = 1.6081.

### Inspect what structurally changed

```bash
python3 geom_outliers.py elites_above_threshold/k11_extend_p4_gap1.6081_tf0.pkl
```

Shows the 4-8 rectangles that differ substantively from pristine M_11
(filtering out the ~280 1-unit boundary-wiggle passengers).

---

## Open questions

1. **How far does δ_α go at fixed k under the random-extension mutation?**
   At k=11 we reached +4 in α before the directed-mutation hit rate
   collapsed (50 trials at +4 → +5 with 0 hits, both multi=1 and multi=2).
   Whether a smarter mutation primitive could push further is open.
2. **Does the k=12 plateau lift with longer in-loop ILP time?**
   Verified k=12 search at 30s in-loop is unreliable; at 1800s verify
   we found α_min = 177 (pristine = 178). To find a real +2 at k=12 we
   need either (a) ≥ 600s in-loop ILP, or (b) verify-on-hit logic
   (now implemented but not yet run at scale at k=12).
3. **Is there a triangle-free improvement at k ≥ 12?** None found in
   100 trials. Whether this is a genuine structural barrier or just a
   low-hit-rate regime is open.
4. **Can δ_α scale with k²?** For asymptotic gap > 2 we'd need
   `δ_α(k) / k² > 0`. Four data points can't distinguish this from
   `δ_α` bounded. Distinguishing would require pushing several `k`
   values to convergence with the verify-on-hit pipeline.

### Definitively closed

- **Can we reach gap = 2.0 at any finite n?** No.
  `gap = LP / α ≤ (n/2 + ε) / α`, and for α > n/4 we have gap < 2 strictly.
  We have α/n > 1/4 in every verified instance.
- **Can gap > 2 be achieved for rectangles?** Open. Would require either
  finding `δ_α ≥ 3k − 2` (no evidence) or a completely different
  construction (not attempted here).

---

## Output directories

- `elites_above_threshold/` — saved pickles per round, naming
  `k{N}_..._gap{X.XXXX}_tf{0|1}.pkl`. Verified results from this
  session use the `_extend_` prefix. Gitignored (large, reproducible).
- `extend_hits/` — raw outputs of `extend_experiment.py`. **Do not
  trust gap values here without re-verifying** — in-loop ILP time
  limit may produce phantom-optimal results at large n. Gitignored.
- `run_outputs/` — final summaries from full `kbox_search.py` runs.
  Gitignored.
- `plots/` — 4-panel visualizations of pristine and modified instances.
  Tracked in the repo so GitHub renders them.

---

## Dependencies

- Python ≥ 3.10
- `gurobipy` with valid license (academic works)
- `torch` for the transformer (MPS on Apple Silicon, only used by `kbox_search.py`)
- `numpy`, `matplotlib`

---

## Lessons learned

1. **In-loop ILP time matters.** At n ≤ 484, 20s is enough to prove
   optimality for most instances. At n = 576, even 30s is not — the solver
   returns feasible solutions of value below the true optimum, producing
   phantom hits. At n ≥ 576, in-loop ILP time should be ≥ 60s, or
   `--verify-ilp-time T` should be set so each candidate is re-solved
   before being saved as an elite.
2. **Random multi-step mutations interfere from pristine, but iterating
   from a previous elite is productive.** At k=11, `multi=2` from
   pristine had a 12% hit rate; `multi=3` from pristine had 0% in the
   same trial budget. Three random mutations from pristine are more
   likely to undo each other than to compose. But once you have a
   verified +1 elite, single mutations from that elite hit +2, then +3,
   then +4 with roughly constant per-step hit rate.
3. **Geometry-based diff is mandatory.** Comparing by label after
   canonicalization (`diff_vs_pristine.py`) produces 291/324 false
   positives at n=324. Comparing rectangle multisets (`geom_diff.py`)
   shows the 1-4 actual modifications cleanly.
4. **The directed-extension mutation always introduces triangles.**
   Verified: 200/200 candidates rejected by triangle-free filter at k=10
   with `--mode directed --multi 2`. To find triangle-free improvements
   you must use `--mode random`, which has a ~10-15% chance of preserving
   triangle-freeness per single step at k ≥ 9.
5. **Parallelism is essential.** With `--workers 8` on M4 Max, 500 trials
   at k=10 finishes in ~3 minutes (vs ~25 min serially). Without it the
   tf-rejection sweep scale here would not have been feasible.
