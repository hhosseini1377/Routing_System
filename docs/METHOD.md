# Method: SLO-Aware Prompt Routing and Deployment Search

This document describes the full method implemented in this repository: how routing fractions are optimized, how latency and quality are modeled, and how deployment setups are filtered by GPU-resource feasibility.

## 1) Problem Setup

We have a set of routed backends (currently three in code) and a global incoming request rate:

- Models indexed by `i = 1..K`.
- Global load: `lambda_global` (RPS).
- Routing fractions: `w in Delta_K`, where `w_i >= 0` and `sum_i w_i = 1`.
- Per-model load: `lambda_i = lambda_global * w_i`.

For each prompt `j` and model `i`, we have a predicted quality score `s_{j,i}` from RouterBench-style data.

For each model and deployment setup `(tp, threads)`, we build a latency curve `ell_i(load)` from measured profiling JSON files.

The goal is to maximize quality while meeting a latency SLO `tau`:

- Quality term: `S(w)` (computed via assignment under target fractions).
- Latency term: `L(w) = sum_i w_i * ell_i(lambda_i)`.
- Constraint: `L(w) <= tau`.

## 2) Lagrangian Formulation

The optimizer uses a relaxed objective:

`max_w  S(w) - beta * (L(w)/tau - 1),   w in Delta_K`

where `beta >= 0` is the latency-penalty multiplier.

The normalized slack is:

`slack = L/tau - 1`

- `slack > 0`: SLO violation.
- `slack < 0`: under SLO.

## 3) Computing the Quality Term `S(w)` (Dual Prices)

`S(w)` is not a direct closed form because counts assigned to each model must match `w` (after integer rounding).

Implemented in `resource_allocation/dual_prices.py`:

1. Convert fractions `w` to integer counts `c`.
2. Maintain per-model dual prices `alpha_i`.
3. Assign each prompt to `argmax_i (s_{j,i} - alpha_i)`.
4. Update `alpha` to reduce count mismatch.
5. Optionally repair assignments to exactly match target counts.

Outputs include:

- `S_hat` (estimated quality under the assignment),
- `alpha` (used as the score-gradient signal in `w` updates).

## 4) Latency Model

Implemented with `PiecewiseLinearLatency` in `resource_allocation/optimize_fractions.py`.

Latency curves are built from `performance_data_*_final.json` by filtering rows at a specific `(tp, thread)` setup and aggregating metric vs load.

Supported metrics include:

- `tpot`,
- `avg_latency_ms`,
- `p95_ttft` (and `p95_tpot` in brute-force path).

For each model:

- `ell_i(load)` is piecewise-linear interpolation,
- `d ell_i / d load` is the segment slope.

Then:

`dL/dw_i = ell_i(lambda_i) + lambda_global * w_i * d ell_i / d lambda_i`.

## 5) Inner Optimization over Routing Fractions `w`

Implemented in `optimize_fractions(...)` (`resource_allocation/optimize_fractions.py`).

At each step:

1. Compute `S_hat` and `alpha` from dual-prices routine.
2. Compute `L`, `dL/dw` from latency curves.
3. Form gradient:
   `grad_w = alpha - beta * (dL/dw) / tau`.
4. Apply optional momentum:
   `velocity = momentum * velocity + grad_w`.
5. Update and project to simplex:
   `w <- ProjectDelta(w + eta * delta)`.

Tracking and outputs:

- Histories for `L`, `S`, objective, and `w`.
- `best_w`, `best_L`, `best_S`, `best_obj`, `best_alpha`.
- Optional EMA on `w` for returned `result.w` (`w_ema_decay`).

## 6) Outer Search for `beta`

Two methods are available.

### A) Subgradient-style update (`optimize_beta`)

`beta <- clip(beta + eta_beta * slack, beta_min, beta_max)`

with:

- adaptive damping on sign flips,
- optional decay of `eta_beta`,
- warm-start of inner optimization using previous `best_w`.

### B) Bracket + bisection (`optimize_beta_bisection`)

1. Evaluate slack at `beta_init`.
2. Search toward bounds to find a sign-change bracket in slack.
3. Bisect the bracket until slack tolerance or interval tolerance is reached.
4. Return midpoint beta and corresponding optimization result.

If no sign change is found inside `[beta_min, beta_max]`, the endpoint with smaller `|slack|` is returned.

## 7) Data and Model Ordering

Canonical backend order is defined in `resource_allocation/models_config.py` (`ROUTING_MODELS`).

`routerbench_data.load_scores(...)` reorders score columns to this canonical order when metadata is present.

Important detail:

- `resource_allocation/main.py` accepts CLI order `(Mistral, Vicuna, Yi)` for `--tp/--threads` and internally reorders to canonical order for optimization.

## 8) Deployment Feasibility (Packing)

Implemented in `resource_allocation/resource_packing.py`.

Given candidate arrays `(tp_levels, thread_percentages, memory_percentages)`:

- Build all shards implied by TP.
- Enforce per-GPU capacities in two dimensions:
  - memory <= 1.0,
  - thread <= 1.0.
- Enforce optional no-same-model-per-GPU.
- Use first-fit decreasing 2D packing heuristic.

The method returns `feasible` with assignment (or reason for infeasibility).

## 9) Brute-Force Setup Search

Implemented in `resource_allocation/brute_force_setup.py`.

For each candidate configuration over:

- TP options,
- thread options,
- memory scale options,
- (optionally) subsets of models,

the pipeline does:

1. Optional underutilization filter via `min_thread_sum_ratio`.
2. Compute per-model memory demand from profiling minima and memory scale.
3. Check packing feasibility (`resource_packing.check_feasibility`).
4. If pack-feasible, run outer-beta search + inner `w` optimization.
5. Mark SLO feasibility if `best_L <= tau * (1 + slack_tol)`.
6. Keep best feasible setup by score (tie-break by lower latency).

Per-candidate records include:

- setup (`subset`, `tp_levels`, `thread_levels`, `memory_levels`),
- feasibility flags,
- best `beta`, score, latency, slack,
- dual prices `alpha`.

## 10) Utility Scans

`resource_allocation/count_packing_feasible.py` counts how many `(tp, thread)` combinations are packing-feasible across GPU counts and model subsets, with the same thread-demand filter and memory-minima assumptions.

## 11) Main Assumptions and Scope

- Latency is modeled as a function of load under measured deployment settings.
- Memory is used directly for feasibility/packing in brute-force search.
- The pipeline currently expects `K=3` models in brute-force code paths.
- Packing feasibility uses a heuristic (FFD), not an exact solver.

## 12) Entry Points

- `python -m resource_allocation.main`:
  optimize `w` for a fixed setup and optional `beta` search.
- `python -m resource_allocation.brute_force_setup`:
  enumerate deployment candidates and optimize each feasible one.
- `python -m resource_allocation.count_packing_feasible`:
  count feasible packing combinations without optimization.

