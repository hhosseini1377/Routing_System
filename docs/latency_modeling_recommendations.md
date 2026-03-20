# Latency Modeling Recommendations for Routing Optimization

Your optimization uses **ℓ_i(θ_i, λ_i)** with **λ_i = λ w_i** and requires **∇L(Θ, w)** w.r.t. **w**. The following keeps the model accurate and optimizer-friendly.

---

## 1. What to Predict (Target Metric)

- **Match the SLO:** If the constraint is on p95 latency, train on **p95_latency_ms** (or **p95_ttft** if you use TTFT as proxy). Your script already supports **avg_latency_ms**, **p99_latency_ms**, **p50_latency_ms**, **p95_ttft**, **p99_ttft**, **tpot**, etc.
- **Recommendation:** Use **p95_ttft** or **p95_latency_ms** (or p99) so the learned ℓ_i matches the percentile you constrain in the optimization.

---

## 2. Inputs (Setup + Load)

Your profiled dimensions:


| Input                      | Role                      | Suggestion                                                                                           |
| -------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------- |
| **tensor_parallel_size**   | GPUs (1, 2, 4)            | Keep. Capacity and latency scale with TP.                                                            |
| **thread_percentage**      | CPU threads for the model | Keep. Strong effect on throughput and latency.                                                       |
| **load_rps**               | λ_i in the formulation    | Keep. Use **log1p(load_rps)** for scale; consider **load_rps** in capacity-style models (see below). |
| memory_util, max_model_len | Fixed in your sweep?      | Add as features only if they vary; otherwise omit.                                                   |


So: **θ = (tensor_parallel_size, thread_percentage)** (and optionally memory_util, max_model_len), **λ = load_rps**. Your current 3-feature setup (tp, log1p(load_rps), thread_pct) is a good base.

---

## 3. Model Family: Capacity-Inspired vs Black-Box

### Option A: Capacity-inspired (recommended for optimization)

Latency often grows as load approaches capacity (queueing). Use a parametric form:

**ℓ(θ, λ) = a(θ) + b(θ) / (c(θ) − λ)**

- **a(θ)** = baseline latency at low load  
- **b(θ), c(θ)** = scale and **capacity (max sustainable RPS)** from setup  
- **c(θ) > λ** in the feasible region; as λ → c, latency increases.

**Regimes:** The model is intended for the **stable regime** (throughput ≈ load, λ < c(θ)). Training with `min_throughput_load_ratio` (e.g. 0.99) keeps only such points. In **overload** (λ ≥ c(θ)) the formula is clamped and gradients are zero—use the model only for λ < c(θ) in the optimizer, or add a constraint/penalty for overload.

**Advantages:**

- Matches queueing intuition and your stable vs overload regimes.
- **Closed-form ∂ℓ/∂λ** (and hence **∂L/∂w** via λ_i = λ w_i), so the routing optimizer gets exact gradients.
- a, b, c can be small NNs in (tp, thread_pct) so the model stays smooth in θ.

You already have a **QueueInspiredLogTTFTModel** (TTFT = a + b/(c − load)); the same idea can be used for end-to-end or p95 latency (see `resource_allocation/capacity_inspired_latency_model.py`).

### Option B: Black-box MLP (current)

- **LatencyModel**: (tp, log1p(load_rps), thread_pct) → log(latency).
- **Pros:** Simple, flexible.  
- **Cons:** **∇_w L** requires autograd or finite differences through the model; less interpretable; can extrapolate poorly when load is near or past capacity.

**Recommendation:** Prefer a capacity-inspired model for the optimizer; keep the MLP as a baseline and for comparison (e.g. k-fold MAE / MAPE).

---

## 4. Training and Data

- **Stable regime only:** Keep **min_throughput_load_ratio** (e.g. 0.99) so you only fit points where throughput ≈ load. The optimizer should only use the model in this regime; overload points can be handled separately (e.g. penalty or two-stage gate).
- **Log target:** Keep training on **log(latency)** (or log1p) so a single model handles a wide range (e.g. 18 ms–3000 ms).
- **Loss:** **Huber** is robust to outliers (e.g. rare high latencies); MSE is fine if data are clean.
- **Validation:** Keep k-fold CV. Add **relative error** (e.g. MAPE) in addition to MAE, since latency scale differs across models. Optionally, one fold with **load_rps** above the rest to test extrapolation to higher load.

---

## 5. Per-Model vs Unified

- **Per-model (current):** One latency model per architecture (e.g. Mistral 7B, Vicuna 13B, Yi 34B). Best accuracy when each model has enough data (e.g. your 300–570 points per file).
- **Unified:** One model with a “model id” or “size” embedding. Useful when adding new models with few samples; usually slightly worse fit per model.

**Recommendation:** Stay **per-model** for now; use the same feature set and training pipeline for each.

---

## 6. Summary


| Choice         | Suggestion                                                                                            |
| -------------- | ----------------------------------------------------------------------------------------------------- |
| **Target**     | p95 or p99 latency (or TTFT) matching your SLO                                                        |
| **Features**   | (tensor_parallel_size, thread_percentage, load_rps) with log1p(load_rps) for MLP                      |
| **Model**      | Capacity-inspired **ℓ = a(θ) + b(θ)/(c(θ)−λ)** for interpretability and exact **∇L**; MLP as baseline |
| **Training**   | Stable regime (throughput/load > 0.99), log target, Huber, k-fold + MAPE                              |
| **Deployment** | One trained model per LLM; same interface (θ, λ) → latency for the router                             |


Implementations:

- **Current (MLP):** `resource_allocation/latency_model.py` + `resource_allocation/train_latency_model.py`
- **Capacity-inspired:** `resource_allocation/capacity_inspired_latency_model.py` + `resource_allocation/train_capacity_latency_model.py` — use raw load_rps; exposes **∂ℓ/∂λ** via `d_latency_d_load(x)` (scale by `scaler_scale[2]` if inputs are standardized)

