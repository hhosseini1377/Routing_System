# Project Description: LLM Routing Optimization

This document describes the optimization problem at the heart of the routing system and how we solve it.

---

## Overview

The routing system distributes incoming LLM inference requests across multiple model backends (e.g. Mistral 7B, Vicuna 13B, Yi 34B). The central question is: **how should we route requests to maximize output quality while satisfying a latency service-level objective (SLO)?**

Different models excel on different prompts; blindly splitting traffic uniformly is suboptimal. At the same time, load-dependent latency means that overloading a single model increases queueing delays. We formulate this as a constrained optimization problem and solve it with gradient-based methods.

---

## The Optimization Problem

### Setup

- **K model backends**, each with its own capacity, latency characteristics, and per-prompt quality.
- **Global request rate** λ (requests per second) arriving at the router.
- **Routing fractions** w = (w₁, …, wₖ) on the simplex: wᵢ ≥ 0, ∑ᵢ wᵢ = 1. Fraction wᵢ of requests is sent to model i.
- **Per-model load** λᵢ = λ · wᵢ (RPS at model i).
- **Latency** ℓᵢ(λᵢ) for model i depends on its load (via queueing effects). Latency increases as load approaches capacity.
- **Per-prompt quality scores** sⱼᵢ: predicted score if prompt j is served by model i. These come from a 0-shot RouterBench-style predictor (e.g. `routerbench_0shot_scores.pkl`).

### Objective and Constraint

We want to:

1. **Maximize quality** S(w): the expected score over all prompts when routed according to w.
2. **Satisfy a latency SLO** τ: the weighted average latency L(Θ, w) must not exceed τ (ms).

Formally:

```
max    S(w)
w∈Δₖ   subject to  L(Θ, w) ≤ τ
```

where Δₖ is the probability simplex and

- **S(w)** = expected score when each prompt is assigned to a model such that the resulting per-model counts match the fractions w (up to integer rounding).
- **L(Θ, w)** = ∑ᵢ wᵢ · ℓᵢ(λᵢ) with λᵢ = λ · wᵢ. Here Θ encodes model setup (e.g. tensor parallelism, thread percentage). The latency ℓᵢ is a function of load λᵢ and setup Θ.

---

## How We Solve It

### 1. Lagrangian Relaxation

We relax the constraint into the objective via a Lagrange multiplier β:

```
max   S(w) − β · (L(Θ, w) / τ − 1)
w∈Δₖ
```

The term (L/τ − 1) is the **normalized slack**: it is zero when L = τ, positive when the SLO is violated, and negative when we are under the SLO. β controls how strongly we penalize violation. For large β, the solution tends to satisfy L ≈ τ.

### 2. Computing S(w): Dual Prices

The score S(w) is not given by a simple closed form. For given fractions w, we need to assign each of N prompts to one of K models such that the per-model counts match w · N (up to rounding), and the total score is maximized.

We use a **dual prices** method:

- Introduce prices αᵢ per model.
- At each iteration, assign each prompt j to model i that maximizes (sⱼᵢ − αᵢ).
- Update α to push counts toward the target c = w · N (subgradient ascent on the dual).
- Optionally apply a greedy repair step to enforce exact integer counts with minimal score loss.

The gradient of S with respect to w is related to the optimal dual variables α. We use the learned α from `score_under_fractions_dual` as the gradient signal for the score term.

### 3. Latency Model: Piecewise-Linear Curves

For each model i, we have a **piecewise-linear** curve: load (RPS) → latency (ms). These are built from profiled performance data (`performance_data_*_final.json`), collected by sweeping load levels and measuring latency metrics (e.g. TPOT, p95 TTFT).

- Between grid points: linear interpolation.
- Outside the grid: clamp to the nearest endpoint.
- The derivative ∂ℓᵢ/∂λᵢ is available in closed form (slope of the active segment).

### 4. Gradient of L with Respect to w

Since λᵢ = λ · wᵢ, we have

```
∂L/∂wᵢ = ℓᵢ(λᵢ) + λ · wᵢ · ∂ℓᵢ/∂λᵢ
```

The piecewise-linear model provides both ℓᵢ and ∂ℓᵢ/∂λᵢ, so the full gradient ∇ₘL is exact.

### 5. Projected Gradient Ascent

The Lagrangian gradient is:

```
∇ₘ [S(w) − β(L/τ − 1)] = α − β · (1/τ) · ∇ₘL
```

We perform **projected gradient ascent** on the simplex:

1. Compute S(w) and α via the dual prices routine.
2. Compute L and ∇ₘL from the piecewise-linear latency curves.
3. Update w ← project(w + η · gradient) onto Δₖ (using Duchi et al. projection).
4. Optionally use momentum and/or exponential moving average of w for stability.

### 6. Choosing β

For a fixed β, the optimizer finds w that balances score and slack. To hit the SLO exactly (L ≈ τ), we can **search for β**:

- If L/τ − 1 > 0 (over SLO), increase β.
- If L/τ − 1 < 0 (under SLO), decrease β.

The `optimize_beta` routine implements this outer loop, adjusting β until |L/τ − 1| is within a tolerance.

---

## Data Pipeline

1. **Per-prompt scores**  
   Load `routerbench_0shot_scores.pkl` (shape N×K). Entry sⱼᵢ is the predicted score for prompt j on model i.

2. **Performance profiling**  
   Run `profiler/collect_performance_data.py` for each model under different (tensor_parallel_size, thread_percentage, load_rps) configurations. Output: `performance_data_*_final.json`.

3. **Latency curves**  
   For each model, extract (load_rps, metric) pairs from the JSON, aggregate per load, and build a `PiecewiseLinearLatency` curve.

4. **Optimization**  
   Run `resource_allocation/main.py` with:
   - `--lambda-global`: global RPS λ
   - `--tau`: latency SLO (ms)
   - `--beta`: Lagrange multiplier (or use `--optimize-beta` to search)
   - `--tp` and `--threads`: setup for each backend

---

## Summary

| Component | Role |
|-----------|------|
| **S(w)** | Quality score; computed via dual prices + greedy repair |
| **L(Θ, w)** | Weighted average latency; from piecewise-linear load→latency curves |
| **Constraint** | L ≤ τ (latency SLO) |
| **Method** | Lagrangian relaxation + projected gradient ascent on simplex |
| **β** | Lagrange multiplier; can be tuned to achieve L ≈ τ |

The resulting routing fractions w can be used to configure the router (e.g. as weights for random or weighted routing) so that requests are distributed across models in a way that maximizes quality while meeting the latency SLO.
