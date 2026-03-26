"""
Brute-force search over deployment setups.

Given the router's objective/constraint pieces:
- score matrix S (from RouterBench scores pickle)
- latency-vs-load curves (from per-model performance JSON)
- feasibility of (tp, threads, memory) using 2D packing constraints

We enumerate candidate deployments and run `optimize_beta` (dual price solver
inside `optimize_fractions`) to find routing fractions for each candidate.

This file also contains pickle-safe helpers (`save_result` / `load_result`)
so old pickled artifacts can be loaded in notebooks.
"""

from __future__ import annotations

import argparse
import itertools
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from tqdm import tqdm

from .models_config import ROUTING_MODELS, get_scores_path, get_model_memory_paths, get_performance_data_paths
from .routerbench_data import (
    build_latency_curves,
    load_model_memory_config,
    load_scores,
)
from .resource_packing import check_feasibility
from .optimize_fractions import optimize_beta


@dataclass
class SetupCandidate:
    """
    One deployment candidate.

    For subset scenarios, `subset` stores the indices (into ROUTING_MODELS)
    of the deployed backends. The arrays `tp_levels`, `thread_levels`,
    `memory_levels` correspond to that subset order.
    """

    subset: tuple[int, ...]
    tp_levels: list[int]
    thread_levels: list[int]  # integer percentages (e.g. 90 means 0.90)
    memory_levels: list[float]  # fractions in [0,1]

    # Optimization outputs (filled when feasible).
    best_beta: float | None = None
    best_score: float | None = None
    best_latency: float | None = None
    best_slack: float | None = None

    # Feasibility flags
    packing_feasible: bool = False
    slo_feasible: bool = False


@dataclass
class BruteForceResult:
    """
    Summary of brute-force search.

    Pickle compatibility:
    - Old pickles can contain additional keys. Since dataclasses here are
      regular (non-slotted) Python objects, unpickling will simply set them
      as attributes on this class even if we don't explicitly declare them.
    """

    root: str
    metric: str
    total_candidates: int

    all_candidates: list[SetupCandidate] = field(default_factory=list)
    all_feasible: list[SetupCandidate] = field(default_factory=list)

    best_setup: SetupCandidate | None = None
    best_score: float | None = None
    best_latency: float | None = None


def save_result(result: BruteForceResult, path: str | Path) -> None:
    """Pickle-save `BruteForceResult`."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_result(path: str | Path) -> BruteForceResult:
    """
    Load a pickled BruteForceResult and support legacy pickles.

    If the artifact was saved after running `python -m resource_allocation.brute_force_setup`,
    pickle may record classes under `__main__` during save. In notebook environments,
    `__main__` doesn't define those classes. We register them before loading.
    """

    path = Path(path)

    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        # Ensure pickle can resolve these names if it stored them under __main__.
        setattr(main_mod, "BruteForceResult", BruteForceResult)
        setattr(main_mod, "SetupCandidate", SetupCandidate)

    with open(path, "rb") as f:
        obj = pickle.load(f)

    return obj


def _thread_int_to_frac(thread_percentage: int) -> float:
    """Convert integer percentage (0..100) to fraction (0..1)."""

    return float(thread_percentage) / 100.0


def _get_memory_frac(min_gpu_util_per_tp: dict[str, float], tp: int, memory_scale: float) -> float:
    key = str(int(tp))
    if key not in min_gpu_util_per_tp:
        raise KeyError(f"No memory util entry for tp={tp} (key='{key}').")
    return float(min_gpu_util_per_tp[key]) * float(memory_scale)


def brute_force_setup(
    *,
    root: Path | str = ".",
    num_gpus: int,
    lambda_global: float,
    tau: float,
    metric: str,
    beta_init: float,
    slack_tol: float,
    tp_options: Sequence[int] = (1, 2, 4),
    thread_options: Sequence[int] = tuple(range(10, 101, 10)),
    memory_scale_options: Sequence[float] = (1.0,),
    min_thread_sum_ratio: float = 0.0,
    include_subset_scenarios: bool = False,
    subset_sizes: Sequence[int] = (2, 3),
    sample_frac: float | None = None,
    seed: int = 0,
    # optimize_fractions params
    n_steps: int = 200,
    eta: float = 0.1,
    # dual-prices (score_under_fractions_dual) params passed through to optimize_fractions
    dual_max_iter: int = 300,
    dual_eta0: float = 1e-3,
    dual_tol: int = 1,
    dual_tie_noise: float = 1e-9,
    momentum: float = 0.0,
    w_ema_decay: float | None = None,
    patience: int | None = None,
    obj_tol: float = 1e-4,
    # optimize_beta params
    max_outer_steps: int = 20,
    eta_beta: float = 0.01,
    eta_beta_min: float = 1e-4,
    eta_beta_decay: float = 0.98,
) -> BruteForceResult:
    """
    Brute-force enumerate (tp, threads, memory) setups and pick the best SLO-feasible.
    """

    root = Path(root)

    K = len(ROUTING_MODELS)
    if K != 3:
        raise ValueError(f"Current pipeline expects exactly 3 routing models, got K={K}")

    model_memory_paths = get_model_memory_paths(root)
    performance_data_paths = get_performance_data_paths(root)

    # Load per-model memory configs once.
    model_min_gpu_utils: list[dict[str, float]] = [
        load_model_memory_config(p) for p in model_memory_paths
    ]

    # Load + (optionally) subsample score matrix once.
    scores_path = get_scores_path(root)
    S_full = load_scores(scores_path)
    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        rng = np.random.default_rng(seed)
        N = S_full.shape[0]
        m = max(1, int(sample_frac * N))
        idx = rng.permutation(N)[:m]
        S_full = S_full[idx]

    # Subset list
    if include_subset_scenarios:
        subset_indices: list[tuple[int, ...]] = []
        for sz in subset_sizes:
            for comb in itertools.combinations(range(K), sz):
                subset_indices.append(tuple(comb))
        # Always include full deployment
        if tuple(range(K)) not in subset_indices:
            subset_indices.append(tuple(range(K)))
    else:
        subset_indices = [tuple(range(K))]

    # Latency cache keyed by (model_index_in_full, tp, thread_percentage, metric).
    latency_cache: dict[tuple[int, int, int, str], Any] = {}

    def get_curve_for_model(model_idx: int, tp: int, thread_percentage: int) -> Any:
        key = (model_idx, int(tp), int(thread_percentage), metric)
        if key in latency_cache:
            return latency_cache[key]
        curves = build_latency_curves(
            tps=[tp],
            threads=[thread_percentage],
            performance_data_paths=[performance_data_paths[model_idx]],
            metric=metric,
        )
        latency_cache[key] = curves[0]
        return curves[0]

    all_candidates: list[SetupCandidate] = []
    all_feasible: list[SetupCandidate] = []

    best_score: float | None = None
    best_setup: SetupCandidate | None = None
    best_latency: float | None = None

    # Pre-compute how many candidate combinations we will consider (after the
    # underutilization filter). This is what we want shown in tqdm's `total=...`.
    total_candidates = 0
    for subset in subset_indices:
        subset = tuple(subset)
        K_sub = len(subset)
        num_mem_scales = len(memory_scale_options) ** K_sub

        for tp_levels in itertools.product(tp_options, repeat=K_sub):
            tp_arr = np.asarray(tp_levels, dtype=float)
            for thread_levels in itertools.product(thread_options, repeat=K_sub):
                thread_fracs = [_thread_int_to_frac(th) for th in thread_levels]
                total_thread_demand = float(np.dot(tp_arr, np.asarray(thread_fracs, dtype=float)))
                if min_thread_sum_ratio > 0.0:
                    # Interpret min_thread_sum_ratio as "fraction of total thread capacity".
                    if total_thread_demand < float(min_thread_sum_ratio) * float(num_gpus):
                        continue
                total_candidates += num_mem_scales

    # Track brute-force candidate evaluation progress.
    # (We keep tqdm output inside this file and disable tqdm inside optimize_beta.)
    pbar = tqdm(total=total_candidates, desc="Evaluating candidates", unit="cand", dynamic_ncols=True)
    try:
        for subset in subset_indices:
            subset = tuple(subset)
            subset_order = list(subset)
            K_sub = len(subset_order)

            # Materialize choice spaces.
            # NOTE: `itertools.product(...)` returns an iterator that gets exhausted.
            # Since these spaces are iterated across nested loops, we must
            # materialize to lists so each outer loop can re-iterate fully.
            tp_choice_space = list(itertools.product(tp_options, repeat=K_sub))
            thread_choice_space = list(itertools.product(thread_options, repeat=K_sub))
            mem_choice_space = list(itertools.product(memory_scale_options, repeat=K_sub))

            for tp_levels in tp_choice_space:
                tp_arr_int = np.asarray(tp_levels, dtype=int)
                tp_arr_float = np.asarray(tp_levels, dtype=float)

                for thread_levels in thread_choice_space:
                    # Thread demand filter (optional).
                    thread_fracs_sub = [_thread_int_to_frac(th) for th in thread_levels]
                    total_thread_demand = float(np.dot(tp_arr_float, np.asarray(thread_fracs_sub, dtype=float)))
                    if min_thread_sum_ratio > 0.0:
                        if total_thread_demand < float(min_thread_sum_ratio) * float(num_gpus):
                            continue

                    # Evaluate each memory_scale combination for this (tp, thread).
                    for mem_scales in mem_choice_space:
                        pbar.update(1)

                        memory_levels = [
                            _get_memory_frac(model_min_gpu_utils[full_k], tp, mem_scale)
                            for full_k, tp, mem_scale in zip(subset_order, tp_levels, mem_scales)
                        ]
                        mem_arr_float = np.asarray(memory_levels, dtype=float)

                        # Packing feasibility
                        feas = check_feasibility(
                            tp_levels=tp_arr_int,
                            thread_percentages=thread_fracs_sub,
                            memory_percentages=mem_arr_float,
                            num_gpus=num_gpus,
                        )

                        cand = SetupCandidate(
                            subset=tuple(subset_order),
                            tp_levels=list(map(int, tp_levels)),
                            thread_levels=list(map(int, thread_levels)),
                            memory_levels=list(map(float, memory_levels)),
                            packing_feasible=bool(feas.feasible),
                        )

                        all_candidates.append(cand)

                        if not feas.feasible:
                            continue

                        # Build latency curves for the subset.
                        latency_curves = [
                            get_curve_for_model(model_idx, tp, th)
                            for model_idx, tp, th in zip(subset_order, tp_levels, thread_levels)
                        ]

                        # Extract score columns for the subset.
                        S = S_full[:, list(subset_order)]

                        beta_res = optimize_beta(
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
                            dual_max_iter=dual_max_iter,
                            dual_eta0=dual_eta0,
                            dual_tol=dual_tol,
                            dual_tie_noise=dual_tie_noise,
                            momentum=momentum,
                            w_ema_decay=w_ema_decay,
                            patience=patience,
                            obj_tol=obj_tol,
                            show_progress=False,
                        )

                        res = beta_res.result
                        cand.best_beta = float(beta_res.best_beta)
                        cand.best_score = float(res.best_S)
                        cand.best_latency = float(res.best_L)
                        cand.best_slack = float((res.best_L / tau) - 1.0)

                        cand.slo_feasible = res.best_L <= tau * (1.0 + slack_tol)

                        if not cand.slo_feasible:
                            continue

                        all_feasible.append(cand)

                        # Rank by best score; tie-break by lower latency.
                        if (best_score is None) or (
                            cand.best_score is not None and cand.best_score > best_score
                        ):
                            best_score = cand.best_score
                            best_setup = cand
                            best_latency = cand.best_latency
                        elif (
                            cand.best_score is not None
                            and best_score is not None
                            and np.isclose(cand.best_score, best_score)
                            and cand.best_latency is not None
                            and best_latency is not None
                            and cand.best_latency < best_latency
                        ):
                            best_setup = cand
                            best_latency = cand.best_latency

        result = BruteForceResult(
            root=str(root),
            metric=metric,
            total_candidates=total_candidates,
            all_candidates=all_candidates,
            all_feasible=all_feasible,
            best_setup=best_setup,
            best_score=best_score,
            best_latency=best_latency,
        )
        return result
    finally:
        pbar.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Brute-force deployment setup search.")
    p.add_argument("--root", type=str, default=".", help="Project root for data files.")
    p.add_argument("--num-gpus", type=int, required=True)
    p.add_argument("--lambda-global", type=float, required=True)
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--metric", type=str, default="tpot", choices=("tpot", "avg_latency_ms", "p95_ttft", "p95_topt"))
    p.add_argument("--beta-init", type=float, default=0.01)
    p.add_argument("--slack-tol", type=float, default=0.02, help="SLO slack tolerance.")

    p.add_argument("--tp-options", type=int, nargs="*", default=[1, 2, 4])
    p.add_argument("--thread-options", type=int, nargs="*", default=list(range(10, 101, 10)))
    p.add_argument("--memory-scale-options", type=float, nargs="*", default=[1.0])

    p.add_argument("--min-thread-sum-ratio", type=float, default=0.0)
    p.add_argument("--include-subset-scenarios", action="store_true")
    p.add_argument("--subset-sizes", type=int, nargs="*", default=[2, 3])

    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)

    # optimize_fractions
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--dual-max-iter", type=int, default=300)
    p.add_argument("--dual-eta0", type=float, default=1e-3)
    p.add_argument("--dual-tol", type=int, default=1)
    p.add_argument("--dual-tie-noise", type=float, default=1e-9)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--w-ema-decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--obj-tol", type=float, default=1e-4)

    # optimize_beta
    p.add_argument("--max-outer-steps", type=int, default=20)
    p.add_argument("--eta-beta", type=float, default=0.01)
    p.add_argument("--eta-beta-min", type=float, default=1e-4)
    p.add_argument("--eta-beta-decay", type=float, default=0.98)

    p.add_argument("-o", "--output", type=str, required=True, help="Where to save the result pickle.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    result = brute_force_setup(
        root=args.root,
        num_gpus=args.num_gpus,
        lambda_global=args.lambda_global,
        tau=args.tau,
        metric=args.metric,
        beta_init=args.beta_init,
        slack_tol=args.slack_tol,
        tp_options=args.tp_options,
        thread_options=args.thread_options,
        memory_scale_options=args.memory_scale_options,
        min_thread_sum_ratio=args.min_thread_sum_ratio,
        include_subset_scenarios=bool(args.include_subset_scenarios),
        subset_sizes=tuple(args.subset_sizes),
        sample_frac=args.sample_frac,
        seed=args.seed,
        n_steps=args.steps,
        eta=args.eta,
        dual_max_iter=args.dual_max_iter,
        dual_eta0=args.dual_eta0,
        dual_tol=args.dual_tol,
        dual_tie_noise=args.dual_tie_noise,
        momentum=args.momentum,
        w_ema_decay=args.w_ema_decay,
        patience=args.patience,
        obj_tol=args.obj_tol,
        max_outer_steps=args.max_outer_steps,
        eta_beta=args.eta_beta,
        eta_beta_min=args.eta_beta_min,
        eta_beta_decay=args.eta_beta_decay,
    )

    save_result(result, args.output)

    if result.best_setup is not None:
        print("Best setup:")
        print("  subset:", result.best_setup.subset)
        print("  tp:", result.best_setup.tp_levels)
        print("  threads:", result.best_setup.thread_levels)
        print("  memory:", result.best_setup.memory_levels)
        print("Best score:", result.best_score)
        print("Best latency:", result.best_latency)


if __name__ == "__main__":
    main()

