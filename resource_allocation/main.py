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
        --metric tpot \\
        --num-gpus 4
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from .dual_prices import score_under_fractions_dual  # noqa: F401  (re-exported use)
from .optimize_fractions import (
    PiecewiseLinearLatency,
    OptimizationResult,
    BetaOptimizationResult,
    optimize_fractions,
    optimize_beta,
    optimize_beta_bisection,
)
from .models_config import get_model_memory_paths
from .resource_packing import FeasibilityResult, check_feasibility
from .routerbench_data import load_model_memory_config, load_scores as load_routerbench_scores


def load_scores(path: str) -> np.ndarray:
    """
    Load and reorder RouterBench scores so columns match the canonical
    backend ordering used by routing (`models_config.ROUTING_MODELS`).
    """

    return load_routerbench_scores(path)


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

    CLI argument order for `tps/threads` is:
      (0) Mistral, (1) Vicuna, (2) Yi.

    Internally, we reorder the curves so their output order matches the
    canonical score order produced by `routerbench_data.load_scores`:
      (0) Mistral, (1) Yi, (2) Vicuna.
    """
    root = Path(".")
    files = [
        root / "performance_data_mistral_7b_final.json",
        root / "performance_data_vicuna_13b_final.json",
        root / "performance_data_yi34b_final.json",
    ]

    if len(tps) != 3 or len(threads) != 3:
        raise ValueError("tps and threads must each have length 3 (one per backend)")

    curves_cli_order: list[PiecewiseLinearLatency] = []
    for path, tp, th in zip(files, tps, threads):
        loads, vals = extract_metric_vs_load_single_model(
            str(path),
            tensor_parallel_size=tp,
            thread_percentage=th,
            metric=metric,
        )
        curves_cli_order.append(PiecewiseLinearLatency(load_grid=loads, latency_ms=vals))

    # Reorder to canonical score order: [Mistral, Yi, Vicuna]
    # CLI order is [Mistral, Vicuna, Yi] so mapping is [0, 2, 1].
    reorder = [0, 2, 1]
    return [curves_cli_order[i] for i in reorder]


def _thread_int_to_frac(thread_percentage: int) -> float:
    return float(thread_percentage) / 100.0


def _get_memory_frac(min_gpu_util_per_tp: dict[str, float], tp: int, memory_scale: float) -> float:
    key = str(int(tp))
    if key not in min_gpu_util_per_tp:
        raise KeyError(f"No memory util entry for tp={tp} (key='{key}').")
    return float(min_gpu_util_per_tp[key]) * float(memory_scale)


def packing_feasibility_for_cli_setup(
    tps_cli: list[int],
    threads_cli: list[int],
    *,
    num_gpus: int,
    root: Path | str = ".",
    memory_scale: float = 1.0,
) -> FeasibilityResult:
    """
    Map CLI order (Mistral, Vicuna, Yi) to canonical score order (Mistral, Yi, Vicuna)
    and run the same 2D packing check as brute_force_setup (min memory at each TP).
    """
    if len(tps_cli) != 3 or len(threads_cli) != 3:
        raise ValueError("tps and threads must each have length 3")

    reorder = [0, 2, 1]
    tp_canon = [tps_cli[i] for i in reorder]
    th_canon = [threads_cli[i] for i in reorder]

    root = Path(root)
    model_min_gpu_utils = [load_model_memory_config(p) for p in get_model_memory_paths(root)]
    thread_fracs = [_thread_int_to_frac(th) for th in th_canon]
    memory_levels = [
        _get_memory_frac(model_min_gpu_utils[k], int(tp), memory_scale)
        for k, tp in enumerate(tp_canon)
    ]

    return check_feasibility(
        tp_levels=np.asarray(tp_canon, dtype=int),
        thread_percentages=thread_fracs,
        memory_percentages=np.asarray(memory_levels, dtype=float),
        num_gpus=num_gpus,
    )


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
    # dual-prices hyperparameters (passed to score_under_fractions_dual)
    dual_max_iter: int = 300,
    dual_eta0: float = 1e-3,
    dual_tol: int = 1,
    dual_tie_noise: float = 1e-9,
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
        dual_max_iter=dual_max_iter,
        dual_eta0=dual_eta0,
        dual_tol=dual_tol,
        dual_tie_noise=dual_tie_noise,
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
    # dual-prices hyperparameters (passed to score_under_fractions_dual)
    dual_max_iter: int = 300,
    dual_eta0: float = 1e-3,
    dual_tol: int = 1,
    dual_tie_noise: float = 1e-9,
    beta_init: float = 0.01,
    max_outer_steps: int = 50,
    eta_beta: float = 0.01,
    eta_beta_min: float = 1e-4,
    eta_beta_decay: float = 0.98,
    slack_tol: float = 0.01,
    beta_method: str = "subgradient",
    beta_min: float = 0.0,
    beta_max: float = 5.0,
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

    inner_kw = dict(
        n_steps=n_steps,
        eta=eta,
        seed=seed,
        momentum=momentum,
        w_ema_decay=w_ema_decay,
        patience=patience,
        obj_tol=obj_tol,
        dual_max_iter=dual_max_iter,
        dual_eta0=dual_eta0,
        dual_tol=dual_tol,
        dual_tie_noise=dual_tie_noise,
    )

    if beta_method == "subgradient":
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
            beta_min=beta_min,
            beta_max=beta_max,
            **inner_kw,
        )
    if beta_method == "bisection":
        return optimize_beta_bisection(
            S=S,
            latency_curves=latency_curves,
            lambda_global=lambda_global,
            tau=tau,
            beta_init=beta_init,
            max_outer_steps=max_outer_steps,
            slack_tol=slack_tol,
            beta_min=beta_min,
            beta_max=beta_max,
            **inner_kw,
        )
    raise ValueError(f"Unknown beta_method: {beta_method!r} (use 'subgradient' or 'bisection')")


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
            "before optimization (e.g., 0.25 to use 25%% of 36120).",
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
        "--optimize-beta-method",
        type=str,
        default="subgradient",
        choices=("subgradient", "bisection"),
        help="Outer β search: subgradient (default) or bracket+bisection on slack (bisection).",
    )
    parser.add_argument(
        "--beta-min",
        type=float,
        default=0.0,
        help="Lower clip for β in both outer search methods (default: 0).",
    )
    parser.add_argument(
        "--beta-max",
        type=float,
        default=5.0,
        help="Upper clip for β in both outer search methods (default: 5).",
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
    parser.add_argument(
        "--dual-max-iter",
        type=int,
        default=300,
        help="Dual-prices iterations inside the score solver (default: 300).",
    )
    parser.add_argument(
        "--dual-eta0",
        type=float,
        default=1e-3,
        help="Dual-prices base step size (default: 1e-3).",
    )
    parser.add_argument(
        "--dual-tol",
        type=int,
        default=1,
        help="Dual-prices tolerance for count mismatch (default: 1).",
    )
    parser.add_argument(
        "--dual-tie-noise",
        type=float,
        default=1e-9,
        help="Dual-prices tie-breaking noise (default: 1e-9).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        metavar="G",
        help="If set, report 2D packing feasibility (resource_packing) for this TP/thread setup.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root for memory JSON files (used with --num-gpus).",
    )
    parser.add_argument(
        "--memory-scale",
        type=float,
        default=1.0,
        metavar="S",
        help="Scale min GPU memory fractions when checking packing (default: 1.0).",
    )
    args = parser.parse_args()

    pack_res: FeasibilityResult | None = None
    if args.num_gpus is not None:
        pack_res = packing_feasibility_for_cli_setup(
            list(args.tp),
            list(args.threads),
            num_gpus=args.num_gpus,
            root=args.root,
            memory_scale=args.memory_scale,
        )
        print("\n=== GPU packing feasibility (canonical order: Mistral, Yi, Vicuna) ===")
        print("Packing feasible:", pack_res.feasible)
        if pack_res.reason:
            print("Reason:", pack_res.reason)

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
            dual_max_iter=args.dual_max_iter,
            dual_eta0=args.dual_eta0,
            dual_tol=args.dual_tol,
            dual_tie_noise=args.dual_tie_noise,
            beta_init=args.beta,
            max_outer_steps=args.max_outer_steps,
            eta_beta=args.eta_beta,
            eta_beta_min=args.eta_beta_min,
            eta_beta_decay=args.eta_beta_decay,
            slack_tol=args.slack_tol,
            beta_method=args.optimize_beta_method,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
        )
        result = beta_result.result
        print("\n=== β search finished ===")
        print("β search method:", args.optimize_beta_method)
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
            dual_max_iter=args.dual_max_iter,
            dual_eta0=args.dual_eta0,
            dual_tol=args.dual_tol,
            dual_tie_noise=args.dual_tie_noise,
        )

    print("\n=== Optimization finished ===")
    if pack_res is not None:
        print("GPU packing feasible:", pack_res.feasible)
    print("Returned w (final/EMA):", result.w)
    print("Best objective S - β(L/τ - 1):", result.best_obj)
    print("  at w:", result.best_w)
    print("  score S_hat:", result.best_S)
    print("  latency L:", result.best_L)
    print("Latency requirement tau:", result.tau)
    print("Normalized slack L/tau - 1 (at best):", (result.best_L / result.tau) - 1.0)

if __name__ == "__main__":
    main()
