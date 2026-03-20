"""
End-to-end script to optimize routing fractions w for a given setup using:

- Per-prompt dual-score model (routerbench_0shot_scores.pkl) for S(w).
- Piecewise-linear latency-vs-load curves from performance_data_*.json.
- Projected gradient ascent on w on the simplex (optimize_fractions).

Usage example:

    python -m resource_allocation.main \\
        --lambda-global 30.0 \\
        --beta 0.01 \\
        --tp 4 4 4 \\
        --threads 50 50 50 \\
        --metric tpot
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .dual_prices import score_under_fractions_dual  # noqa: F401  (re-exported use)
from .optimize_fractions import (
    PiecewiseLinearLatency,
    OptimizationResult,
    BetaOptimizationResult,
    optimize_fractions,
    optimize_beta,
)


def load_scores(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "scores" not in data:
        raise KeyError("routerbench_0shot_scores.pkl must contain 'scores'")
    S = np.asarray(data["scores"], dtype=float)
    if S.ndim != 2:
        raise ValueError("scores must be 2D (N,K)")
    return S


def extract_metric_vs_load_single_model(
    path: str,
    tensor_parallel_size: int,
    thread_percentage: int,
    metric: str = "tpot",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (load_rps, metric_values) for a single model from a performance_data JSON.

    Filters rows to a given setup (tensor_parallel_size, thread_percentage),
    then aggregates metric values per unique load_rps and returns loads and
    averaged metric, sorted by load.
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Collect all values per load so we can aggregate and ensure strictly
    # increasing grid (one point per unique load).
    by_load: dict[float, list[float]] = {}
    for item in data:
        setup = item.get("setup", {}) or {}
        result = item.get("result", {})
        perf = result.get("performance", {}) or {}

        # Filter by setup
        tp = int(setup.get("tensor_parallel_size", -1))
        threads = int(setup.get("thread_percentage", -1))
        if tp != tensor_parallel_size or threads != thread_percentage:
            continue

        load_rps = float(setup["load_rps"])

        if metric == "tpot":
            tpot_list = result.get("tpot_ms_list")
            if tpot_list:
                val = float(np.mean(tpot_list))
            else:
                val = perf.get("avg_tpot_ms")
        elif metric == "avg_latency_ms":
            val = perf.get("avg_latency_ms")
        elif metric == "p95_ttft":
            ttfts = result.get("ttfts_ms")
            if ttfts:
                val = float(np.percentile(ttfts, 95))
            else:
                val = (result.get("metrics_after", {}) or {}).get("p95_ttft_ms")
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if val is not None:
            by_load.setdefault(load_rps, []).append(float(val))

    if not by_load:
        raise ValueError(
            f"No valid (load, {metric}) pairs found in {path} "
            f"for tp={tensor_parallel_size}, threads={thread_percentage}"
        )

    unique_loads = sorted(by_load.keys())
    loads_arr = np.asarray(unique_loads, dtype=float)
    vals_arr = np.asarray([np.mean(by_load[l]) for l in unique_loads], dtype=float)
    return loads_arr, vals_arr


def build_latency_curves_for_three_models(
    tps: list[int],
    threads: list[int],
    metric: str,
) -> list[PiecewiseLinearLatency]:
    """
    Build PiecewiseLinearLatency curves for the three backends using the
    *_final.json performance files.

    Each backend can have its own (tp, threads) setup; the order of models is
    assumed to match the columns of S in routerbench_0shot_scores.pkl:
      0: Mistral, 1: Vicuna, 2: Yi.
    """
    root = Path(".")
    files = [
        root / "performance_data_mistral_7b_final.json",
        root / "performance_data_vicuna_13b_final.json",
        root / "performance_data_yi34b_final.json",
    ]

    if len(tps) != 3 or len(threads) != 3:
        raise ValueError("tps and threads must each have length 3 (one per backend)")

    curves: list[PiecewiseLinearLatency] = []
    for path, tp, th in zip(files, tps, threads):
        loads, vals = extract_metric_vs_load_single_model(
            str(path),
            tensor_parallel_size=tp,
            thread_percentage=th,
            metric=metric,
        )
        curves.append(PiecewiseLinearLatency(load_grid=loads, latency_ms=vals))
    return curves


def run_optimize(
    lambda_global: float,
    beta: float,
    tau: float,
    tps: list[int],
    threads: list[int],
    metric: str,
    n_steps: int,
    eta: float,
    seed: int,
    sample_frac: float | None = None,
    momentum: float = 0.0,
    w_ema_decay: float | None = None,
    patience: int | None = None,
    obj_tol: float = 1e-4,
) -> OptimizationResult:
    """
    Load data and run optimize_fractions for the three-model setup.
    tau is the latency requirement (SLO) in ms.
    """
    S = load_scores("datasets/routerbench_0shot_scores.pkl")
    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        rng = np.random.default_rng(seed)
        N = S.shape[0]
        m = max(1, int(sample_frac * N))
        idx = rng.permutation(N)[:m]
        S = S[idx]
        print(f"Subsampled scores: using {m}/{N} prompts (fraction={sample_frac:.3f})")
    latency_curves = build_latency_curves_for_three_models(tps=tps, threads=threads, metric=metric)

    K = S.shape[1]
    w0 = np.ones(K, dtype=float) / K

    result = optimize_fractions(
        S=S,
        latency_curves=latency_curves,
        lambda_global=lambda_global,
        beta=beta,
        tau=tau,
        w_init=w0,
        n_steps=n_steps,
        eta=eta,
        seed=seed,
        momentum=momentum,
        w_ema_decay=w_ema_decay,
        patience=patience,
        obj_tol=obj_tol,
    )
    return result


def run_optimize_beta(
    lambda_global: float,
    tau: float,
    tps: list[int],
    threads: list[int],
    metric: str,
    n_steps: int,
    eta: float,
    seed: int,
    sample_frac: float | None = None,
    momentum: float = 0.0,
    w_ema_decay: float | None = None,
    patience: int | None = None,
    obj_tol: float = 1e-4,
    beta_init: float = 0.01,
    max_outer_steps: int = 50,
    eta_beta: float = 0.01,
    eta_beta_min: float = 1e-4,
    eta_beta_decay: float = 0.98,
    slack_tol: float = 0.01,
) -> BetaOptimizationResult:
    """Find β such that normalized slack (L/tau - 1) ≈ 0."""
    S = load_scores("datasets/routerbench_0shot_scores.pkl")
    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        rng = np.random.default_rng(seed)
        N = S.shape[0]
        m = max(1, int(sample_frac * N))
        idx = rng.permutation(N)[:m]
        S = S[idx]
        print(f"Subsampled scores: using {m}/{N} prompts (fraction={sample_frac:.3f})")
    latency_curves = build_latency_curves_for_three_models(tps=tps, threads=threads, metric=metric)

    return optimize_beta(
        S=S,
        latency_curves=latency_curves,
        lambda_global=lambda_global,
        tau=tau,
        beta_init=beta_init,
        max_outer_steps=max_outer_steps,
        eta_beta=eta_beta,
        eta_beta_min=eta_beta_min,
        eta_beta_decay=eta_beta_decay,
        slack_tol=slack_tol,
        n_steps=n_steps,
        eta=eta,
        seed=seed,
        momentum=momentum,
        w_ema_decay=w_ema_decay,
        patience=patience,
        obj_tol=obj_tol,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize routing fractions w using dual prices + piecewise-linear latency."
    )
    parser.add_argument(
        "--lambda-global",
        type=float,
        required=True,
        help="Global request rate λ (RPS).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        required=True,
        help="Lagrange multiplier β for latency penalty.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        required=True,
        help="Latency requirement (SLO) in ms. Constraint: L(Θ,w) ≤ tau.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        nargs=3,
        required=True,
        metavar=("TP_MISTRAL", "TP_VICUNA", "TP_YI"),
        help="tensor_parallel_size for each backend (Mistral, Vicuna, Yi).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs=3,
        required=True,
        metavar=("TH_MISTRAL", "TH_VICUNA", "TH_YI"),
        help="thread_percentage for each backend (Mistral, Vicuna, Yi).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="tpot",
        choices=("tpot", "avg_latency_ms", "p95_ttft"),
        help="Latency metric to build latency curves from (default: tpot).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of projected gradient ascent steps.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="Step size for w updates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed passed to dual prices routine.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        metavar="F",
        help="If set in (0,1), randomly subsample a fraction F of prompts from scores "
             "before optimization (e.g., 0.25 to use 25% of 36120).",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        metavar="M",
        help="Momentum for gradient updates (0=off). Use e.g. 0.9 for smoother w updates.",
    )
    parser.add_argument(
        "--w-ema-decay",
        type=float,
        default=None,
        metavar="D",
        help="Exponential moving average decay for w (e.g. 0.99). Returned w is the EMA, not last step.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        metavar="P",
        help="Early stopping: stop if objective has not improved by obj_tol for P steps. Default: no early stopping.",
    )
    parser.add_argument(
        "--obj-tol",
        type=float,
        default=1e-4,
        metavar="T",
        help="Minimum objective improvement to count as progress (used with --patience).",
    )
    parser.add_argument(
        "--optimize-beta",
        action="store_true",
        help="Search for optimal β such that normalized slack ≈ 0. --beta is used as initial β.",
    )
    parser.add_argument(
        "--max-outer-steps",
        type=int,
        default=50,
        help="Max outer iterations when --optimize-beta (default: 50).",
    )
    parser.add_argument(
        "--eta-beta",
        type=float,
        default=0.01,
        help="Initial step size for β updates when --optimize-beta (default: 0.01).",
    )
    parser.add_argument(
        "--eta-beta-min",
        type=float,
        default=1e-4,
        help="Minimum eta_beta after oscillation damping (default: 1e-4).",
    )
    parser.add_argument(
        "--eta-beta-decay",
        type=float,
        default=0.98,
        help="Decay factor for eta_beta each outer iteration; 1.0 disables decay (default: 0.98).",
    )
    parser.add_argument(
        "--slack-tol",
        type=float,
        default=0.01,
        help="Stop when |L/tau - 1| < this when --optimize-beta (default: 0.01).",
    )
    args = parser.parse_args()

    if args.optimize_beta:
        beta_result = run_optimize_beta(
            lambda_global=args.lambda_global,
            tau=args.tau,
            tps=list(args.tp),
            threads=list(args.threads),
            metric=args.metric,
            n_steps=args.steps,
            eta=args.eta,
            seed=args.seed,
            sample_frac=args.sample_frac,
            momentum=args.momentum,
            w_ema_decay=args.w_ema_decay,
            patience=args.patience,
            obj_tol=args.obj_tol,
            beta_init=args.beta,
            max_outer_steps=args.max_outer_steps,
            eta_beta=args.eta_beta,
            eta_beta_min=args.eta_beta_min,
            eta_beta_decay=args.eta_beta_decay,
            slack_tol=args.slack_tol,
        )
        result = beta_result.result
        print("\n=== β search finished ===")
        print("Optimal β:", beta_result.best_beta)
        print("Outer iterations:", len(beta_result.history_beta))
        print("Slack history:", [f"{s:.4f}" for s in beta_result.history_slack])

    else:
        result = run_optimize(
            lambda_global=args.lambda_global,
            beta=args.beta,
            tau=args.tau,
            tps=list(args.tp),
            threads=list(args.threads),
            metric=args.metric,
            n_steps=args.steps,
            eta=args.eta,
            seed=args.seed,
            sample_frac=args.sample_frac,
            momentum=args.momentum,
            w_ema_decay=args.w_ema_decay,
            patience=args.patience,
            obj_tol=args.obj_tol,
        )

    print("\n=== Optimization finished ===")
    print("Returned w (final/EMA):", result.w)
    print("Best objective S - β(L/τ - 1):", result.best_obj)
    print("  at w:", result.best_w)
    print("  score S_hat:", result.best_S)
    print("  latency L:", result.best_L)
    print("Latency requirement tau:", result.tau)
    print("Normalized slack L/tau - 1 (at best):", (result.best_L / result.tau) - 1.0)

if __name__ == "__main__":
    main()
