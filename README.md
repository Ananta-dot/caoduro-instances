# MISR Integrality Gap Search via k-box Seeding + PatternBoost

Computational search for rectangle intersection graphs with large LP/ILP
integrality gap on the Maximum Independent Set of Rectangles (MISR) problem.

Builds on two pieces of prior work:

- **Chalermsook & Chuzhoy (2008)** — proved an asymptotic `3/2` integrality
  gap lower bound for MISR (the `rectanglesfull.pdf` in this project).
- **Caoduro, Cslovjecsek, Pilipczuk, Węgrzycki (2022)** — proved the integrality
  gap approaches `2` in the limit, but using axis-parallel *segments*, not
  general rectangles (the `2205.15189v1.pdf` in this project).

This repo's goal: realize Caoduro et al.'s construction as thin rectangles,
seed a PatternBoost-style search from it, and computationally search for
rectangle instances whose finite-*n* integrality gap exceeds the Caoduro
family's value at the same *n*.

As of the current state: **one verified triangle-free rectangle instance at
n = 324 with α\*/α = 162/105 = 1.5429**, exceeding Caoduro's 1.5283 at the
same *n*. One rectangle fewer in the maximum independent set than Caoduro's
bound allows, ILP-proved optimal.

---

## The pipeline at a glance

```
   ┌──────────────────┐
   │  mistr_runner.py │  ←  library: local_search, Gurobi LP/ILP,
   │  (unchanged)     │     utility functions, DEVICE detection
   └────────┬─────────┘
            │  imports
            ▼
   ┌──────────────────┐
   │   kbox_search.py │  ←  driver you run
   │                  │     - builds k-box seeds
   │                  │     - reloads prior elite pickles
   │                  │     - trains a transformer on the elite pool
   │                  │     - samples new candidates
   │                  │     - orchestrates local_search
   │                  │     - saves every round's best + a final summary
   └────────┬─────────┘
            │  reads/writes
            ▼
   ┌──────────────────┐
   │  elites_above_   │  ←  pickles per round (k{N}_r{M}_gap{X}_tf{0|1}.pkl)
   │   threshold/     │
   └──────────────────┘
   ┌──────────────────┐
   │   run_outputs/   │  ←  final summary + all-elites dump at end of run
   └──────────────────┘
```

---

## File map

### Primary files (the pipeline)

| File | Role |
|---|---|
| `mistr_runner.py` | **Library, not run directly.** Your original PatternBoost code. `kbox_search.py` imports `local_search`, `solve_lp_ilp`, `DEVICE`, and utilities from it. |
| `kbox_search.py` | **The driver you run.** Builds pristine M_k seeds, loads prior elites from pickles, runs a rebuilt transformer sized for larger n, drives local search via `mistr_runner.local_search`, saves every improvement. |
| `kbox_misr.py` | Generates the pristine M_k instance (the Caoduro segment construction as thin rectangles) in the `(H, V)` twin-sequence format. Self-verifies against the paper's combinatorics. |
| `kbox_parallel.py` | Vectorized triangle-free check (60-80x faster at n ≥ 100) and multiprocessing Gurobi wrappers. |

### Verification + diagnostics

| File | Role |
|---|---|
| `verify_instance.py` | Re-solves any saved pickle three ways: clique LP (your search's metric), edge LP (the paper's α\*), and full ILP with proof-of-optimality. Tells you whether a reported gap is real. |
| `plot_large.py` | 4-panel visualization for large-n pickles: anatomy, IS-only, orientation, box structure. Shows whether the verified instance differs from pristine M_k. |
| `plot_instance.py` | Smaller 2-panel version; useful for quick previews of small-n instances. |

### Data / reference

| File | Role |
|---|---|
| `rectanglesfull.pdf` | Chalermsook & Chuzhoy 2008 paper. |
| `2205.15189v1.pdf` | Caoduro et al. 2022 paper (the k-box construction). |
| `misr_122.png`, `misr_graph_122.png` | Previous n=12 gap-1.5 results. |
| `kbox_instances.json` | Concrete k=3, k=5, k=10 instances as JSON (useful for sanity-checking). |

---

## What each file is doing at each stage

### 1. Setup and sanity — `kbox_misr.py`

Builds M_k as 4k² thin rectangles in the (H, V) encoding. Verifies:

- The intersection graph matches the paper's structural ground truth (e.g., 99
  edges at k=3, 6723 at k=9).
- The graph is triangle-free (every grid point covered by ≤ 2 rectangles).
- The paper's explicit independent set of size k²+3k−2 is actually independent.

Ground-truth values:

| k | n | α\* | α | k-box gap |
|---|---|---|---|---|
| 2 | 16 | 8 | 8 | 1.0000 |
| 3 | 36 | 18 | 16 | 1.1250 |
| 5 | 100 | 50 | 38 | 1.3158 |
| 7 | 196 | 98 | 68 | 1.4412 |
| 9 | 324 | 162 | 106 | 1.5283 |
| 10 | 400 | 200 | 128 | 1.5625 |
| 12 | 576 | 288 | 178 | 1.6180 |

Run command: `python3 kbox_misr.py --sweep` (for the full table above) or
`python3 kbox_misr.py --k 3 --gurobi` (verifies Gurobi returns the expected
LP=18, ILP=16 at k=3).

### 2. The search — `kbox_search.py`

Three channels feed the search pool at each k:

1. **Pristine k-box + perturbations** via `kbox_seeded_pool`. Shuffles boxes
   along the diagonal, swaps segments within a box, perturbs segment extents
   — stays near the k-box's combinatorial neighborhood.
2. **Elite reuse** via `load_pickle_elites`. Scans `elites_above_threshold/`
   for saved (H, V) pairs matching the current k, loads the top-N by gap,
   uses them as seeds (plus a few perturbations each).
3. **Transformer proposer** when `--use-transformer` is on. A `TinyGPT`
   rebuilt inside `kbox_search.py` (sized for the target n, so MAX_N=416 at
   k=10 rather than the stock 128) trains on the accumulated elite pool
   between rounds and samples new candidates.

Each seed goes into `mistr_runner.local_search`, which runs tabu+SA local
search against the exact Gurobi LP/ILP. Elites from every seed get pushed
back into the transformer's training pool. Every round's best is saved to
`elites_above_threshold/` if its gap ≥ `--save-threshold` (default 1.40).

At the end of the run (or on Ctrl+C / exception), a final summary pickle is
written to `run_outputs/` unconditionally — thanks to a `try/finally`
wrapper, you never lose data from a long run crashing at the end.

### 3. Verification — `verify_instance.py`

Takes one pickle, recomputes everything three ways:

- **Clique LP**: one constraint per intersection region. This is what your
  search scores against. Tight for triangle-free instances.
- **Edge LP**: one constraint per pairwise intersection. Equal to n/2 if
  and only if the instance is triangle-free. This is the paper's α\*.
- **ILP** with `MIPFocus=2` (prove-optimal mode): finds α and *proves* it's
  optimal. If the time limit runs out without proof, the reported ILP is
  only an upper bound on α, so the true gap might be lower.

This is the script that converts "looks like a good result" into
"certifiably a good result." Run it on every headline pickle before
claiming anything. Typical run on n=324: a few seconds. On n=400 with a
triangle-free instance: a few minutes.

### 4. Visualization — `plot_large.py`

Four panels per pickle, all at once:

- **A. Anatomy**: all rectangles, IS in red.
- **B. Selected only**: just the independent set, non-selected invisible.
- **C. Orientation**: horizontals vs verticals vs square-ish.
- **D. Box structure**: rectangles colored by inferred k-box membership,
  with IS count per box in the legend.

Panel D is the interesting one for research — if your verified instance has
a very different IS-per-box distribution than pristine M_k, that's where the
structural difference lives. That's the first clue toward a theorem.

---

## When do I run `mistr_runner.py` directly?

**Almost never.** It's a library, imported by `kbox_search.py`. Treat it as
read-only.

The only reason to run it directly would be for an ablation: to measure how
stock PatternBoost (without k-box seeds) performs at large n. That would
give you a baseline to contrast against the k-box-guided runs.

```bash
# Pure PatternBoost baseline (no k-box seeding)
python3 mistr_runner.py --n_start 8 --n_target 400 --rounds_per_n 5
```

Expect this to plateau well below 1.5 at large n — without k-box seeds,
the search has no reason to find diagonally-block-structured instances.

---

## Typical workflow

### Initial sanity (5 minutes)

```bash
python3 kbox_misr.py --sweep             # structural verification
python3 kbox_misr.py --k 3 --gurobi      # Gurobi sanity
python3 kbox_search.py --save-smoke      # save path works
python3 kbox_parallel.py --bench-vec     # vectorization speedup check
```

### Verify any existing result

```bash
python3 verify_instance.py elites_above_threshold/k9_r2_gap1.5429_tf1.pkl 1800
```

Expected: `triangle-free: True`, `edge_LP == n/2: True`,
`ILP proved optimal: True`, both metrics agree at 1.542857.

### Main experiment

```bash
python3 kbox_search.py --run \
    --k-start 10 --k-end 10 --rounds 20 \
    --use-transformer --reuse-tf-only --reuse-min-gap 1.5 \
    --xf-samples 12 --xf-steps 60 --xf-pool 256 \
    --local-time 4.0 --seeds-per-round 24
```

Live output every round shows best-so-far and transformer loss. Pickles of
every improvement land in `elites_above_threshold/`. A final summary lands
in `run_outputs/` when the run ends.

### Visualize a result

```bash
python3 plot_large.py elites_above_threshold/k9_r2_gap1.5429_tf1.pkl --format png --dpi 200
```

### Extend the sweep

```bash
# After k=10 verifies, extend to k=11, 12
python3 kbox_search.py --run \
    --k-start 11 --k-end 12 --rounds 20 \
    --use-transformer --reuse-tf-only --reuse-min-gap 1.5 \
    --local-time 6.0 --seeds-per-round 24
```

---

## Output directories

- `elites_above_threshold/` — One pickle per round where gap ≥ save-threshold
  (default 1.40). Filename `k{N}_r{M}_gap{X.XXXX}_tf{0|1}.pkl` encodes k,
  round, gap, and triangle-free status so you can tell at a glance which
  pickles matter without opening them.
- `run_outputs/` — One pickle per run, written unconditionally at end
  (including on Ctrl+C). Contains the best instance and all per-round
  elites that exceeded the threshold.

Neither directory is under version control. Delete to reset.

---

## Key results and the open questions

### Verified

- **α\*/α = 1.5429 on a triangle-free rectangle intersection graph at n=324.**
  Instance saved as `elites_above_threshold/k9_r2_gap1.5429_tf1.pkl`.
  One rectangle fewer in the IS than Caoduro's k-box construction at the same
  n (105 vs 106). ILP proved optimal in 7.7 seconds.

### Open

1. Does the +1 improvement over Caoduro replicate at k=10, k=11, k=12? If
   yes, this is a family of finite-n improvements. If no, the k=9 result is
   a standalone data point.
2. Is there a combinatorial description of the modification that produced
   the 1.5429 instance? If yes, it might generalize to a stated theorem. If
   no, we have computational evidence without a theoretical path.
3. Can the approach produce a +2 improvement (α = 104 at k=9, gap = 1.5577)?
   That would be a qualitatively stronger result.

### Not open (definitively)

- "Can we reach gap = 2.0?" No. At any finite n, gap = 2k²/(k²+3k−2) < 2
  strictly. The Caoduro family approaches 2 only in the limit. Reaching
  1.99 would require n ≈ 1.4 million, which is infeasible with current
  methods.
- "Can gap > 2 be achieved for rectangles?" Unknown, open research
  question. Would require a construction no one has found.

---

## Dependencies

- Python ≥ 3.10
- `gurobipy` with valid license (academic works; instance solves are all
  under a minute at n ≤ 400 with MIPFocus=2)
- `torch` for the transformer (MPS on Apple Silicon works well)
- `numpy`, `matplotlib`
- No `scipy` — everything that needs solving goes through Gurobi.

---

## Hardware notes

Current setup assumes Apple Silicon (M4 Max). Gurobi runs on CPU; the
transformer uses MPS. Multiprocessing Gurobi scales well up to the number
of performance cores. `kbox_parallel.parallel_evaluate` supports it, but
`kbox_search.py`'s driver currently runs local search serially — parallel
seeds per round would be the next engineering win.
