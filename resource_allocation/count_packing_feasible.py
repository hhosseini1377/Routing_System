"""
Enumerate (tp, thread) combinations and count how many pass:

1. Thread demand filter (same as brute_force_setup): total thread demand
   sum_i tp_i * (thread_pct_i/100) >= min_thread_sum_ratio * num_gpus.
2. Packing feasibility via resource_packing.check_feasibility, using minimum
   GPU memory fractions at each model's TP (from profiler JSON), scaled by
   a fixed memory_scale (default 1.0).

Use this to see how feasible candidate counts change with num_gpus and with
which models are deployed (subset size / identity).
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .models_config import ROUTING_MODELS, get_model_memory_paths
from .routerbench_data import load_model_memory_config
from .resource_packing import check_feasibility


def _thread_int_to_frac(thread_percentage: int) -> float:
    return float(thread_percentage) / 100.0


def _get_memory_frac(min_gpu_util_per_tp: dict[str, float], tp: int, memory_scale: float) -> float:
    key = str(int(tp))
    if key not in min_gpu_util_per_tp:
        raise KeyError(f"No memory util entry for tp={tp} (key='{key}').")
    return float(min_gpu_util_per_tp[key]) * float(memory_scale)


@dataclass(frozen=True)
class SubsetCount:
    num_gpus: int
    subset: tuple[int, ...]
    n_pass_thread_filter: int
    n_packing_feasible: int


def count_packing_feasible_for_subset(
    *,
    subset: tuple[int, ...],
    num_gpus: int,
    min_thread_sum_ratio: float,
    require_exact_thread_sum: bool,
    tp_options: Sequence[int],
    thread_options: Sequence[int],
    model_min_gpu_utils: list[dict[str, float]],
    memory_scale: float = 1.0,
) -> tuple[int, int]:
    """
    Returns (n_pass_thread_filter, n_packing_feasible) for one model subset.

    Only combinations that pass the thread-demand filter are eligible for
    packing; n_packing_feasible is a subset of those.
    """
    subset_order = list(subset)
    k_sub = len(subset_order)

    n_pass = 0
    n_feas = 0

    for tp_levels in itertools.product(tp_options, repeat=k_sub):
        tp_arr_int = np.asarray(tp_levels, dtype=int)
        tp_arr_float = np.asarray(tp_levels, dtype=float)

        for thread_levels in itertools.product(thread_options, repeat=k_sub):
            if require_exact_thread_sum:
                # Exact equality filter using integer math:
                # total_thread_demand == num_gpus
                # <=> sum_i tp_i * (thread_pct_i/100) == num_gpus
                # <=> sum_i tp_i * thread_pct_i == 100 * num_gpus
                demand_units = int(
                    sum(int(tp) * int(th) for tp, th in zip(tp_levels, thread_levels))
                )
                if demand_units != 100 * int(num_gpus):
                    continue

            thread_fracs_sub = [_thread_int_to_frac(th) for th in thread_levels]
            total_thread_demand = float(
                np.dot(tp_arr_float, np.asarray(thread_fracs_sub, dtype=float))
            )

            if (not require_exact_thread_sum) and (min_thread_sum_ratio > 0.0):
                if total_thread_demand < float(min_thread_sum_ratio) * float(num_gpus):
                    continue

            n_pass += 1

            memory_levels = [
                _get_memory_frac(model_min_gpu_utils[full_k], int(tp), memory_scale)
                for full_k, tp in zip(subset_order, tp_levels)
            ]
            mem_arr_float = np.asarray(memory_levels, dtype=float)

            feas = check_feasibility(
                tp_levels=tp_arr_int,
                thread_percentages=thread_fracs_sub,
                memory_percentages=mem_arr_float,
                num_gpus=num_gpus,
            )
            if feas.feasible:
                n_feas += 1

    return n_pass, n_feas


def run_scan(
    *,
    root: Path,
    num_gpus_list: Sequence[int],
    min_thread_sum_ratio: float,
    require_exact_thread_sum: bool,
    tp_options: Sequence[int],
    thread_options: Sequence[int],
    subset_sizes: Sequence[int],
    memory_scale: float,
) -> list[SubsetCount]:
    """Run counts for every requested GPU count and model subset."""
    k_full = len(ROUTING_MODELS)
    if k_full != 3:
        raise ValueError(f"This tool expects 3 ROUTING_MODELS, got K={k_full}")

    model_memory_paths = get_model_memory_paths(root)
    model_min_gpu_utils = [load_model_memory_config(p) for p in model_memory_paths]

    results: list[SubsetCount] = []

    for num_gpus in num_gpus_list:
        for sz in subset_sizes:
            if sz < 1 or sz > k_full:
                raise ValueError(f"subset size must be in [1, {k_full}], got {sz}")
            for subset in itertools.combinations(range(k_full), sz):
                subset_t = tuple(subset)
                n_pass, n_feas = count_packing_feasible_for_subset(
                    subset=subset_t,
                    num_gpus=num_gpus,
                    min_thread_sum_ratio=min_thread_sum_ratio,
                    require_exact_thread_sum=require_exact_thread_sum,
                    tp_options=tp_options,
                    thread_options=thread_options,
                    model_min_gpu_utils=model_min_gpu_utils,
                    memory_scale=memory_scale,
                )
                results.append(
                    SubsetCount(
                        num_gpus=num_gpus,
                        subset=subset_t,
                        n_pass_thread_filter=n_pass,
                        n_packing_feasible=n_feas,
                    )
                )

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Count packing-feasible (tp,thread) setups vs GPUs and model subsets."
    )
    p.add_argument("--root", type=str, default=".", help="Project root (for memory JSON paths).")
    p.add_argument(
        "--num-gpus",
        type=int,
        nargs="+",
        required=True,
        help="One or more GPU counts to evaluate (e.g. 2 4 8).",
    )
    p.add_argument(
        "--min-thread-sum-ratio",
        type=float,
        default=0.0,
        help="Same as brute_force_setup: require total thread demand >= this * num_gpus.",
    )
    p.add_argument(
        "--require-exact-thread-sum",
        action="store_true",
        help=(
            "If set, require exact equality sum_i tp_i*(thread_pct_i/100) == num_gpus "
            "(checked via integer units: sum_i tp_i*thread_pct_i == 100*num_gpus). "
            "When set, --min-thread-sum-ratio is ignored."
        ),
    )
    p.add_argument(
        "--tp-options",
        type=int,
        nargs="*",
        default=[1, 2, 4],
        help="TP levels per model (same default as brute_force_setup).",
    )
    p.add_argument(
        "--thread-options",
        type=int,
        nargs="*",
        default=list(range(10, 101, 10)),
        help="Thread %% per model (same default as brute_force_setup).",
    )
    p.add_argument(
        "--subset-sizes",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="For each size s, enumerate all C(K,s) model subsets (canonical indices).",
    )
    p.add_argument(
        "--memory-scale",
        type=float,
        default=1.0,
        help="Scale min GPU memory fractions from profiler JSON (default 1.0).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.root)

    results = run_scan(
        root=root,
        num_gpus_list=args.num_gpus,
        min_thread_sum_ratio=args.min_thread_sum_ratio,
        require_exact_thread_sum=bool(args.require_exact_thread_sum),
        tp_options=args.tp_options,
        thread_options=args.thread_options,
        subset_sizes=args.subset_sizes,
        memory_scale=args.memory_scale,
    )

    # Pretty print: match ROUTING_MODELS display for subset indices
    names = tuple(m.display_name for m in ROUTING_MODELS)

    print(
        f"tp_options={list(args.tp_options)} thread_options={list(args.thread_options)} "
        f"min_thread_sum_ratio={args.min_thread_sum_ratio} "
        f"require_exact_thread_sum={bool(args.require_exact_thread_sum)} "
        f"memory_scale={args.memory_scale}"
    )
    print(
        "num_gpus  subset(idx)  subset(names)  pass_thread_filter  packing_feasible"
    )

    by_gpu: dict[int, int] = {}
    for r in results:
        idx_str = str(list(r.subset))
        name_str = ", ".join(names[i] for i in r.subset)
        print(
            f"{r.num_gpus:8d}  {idx_str:11s}  {name_str:13s}  {r.n_pass_thread_filter:19d}  {r.n_packing_feasible:16d}"
        )
        by_gpu[r.num_gpus] = by_gpu.get(r.num_gpus, 0) + r.n_packing_feasible

    print("\nTotal packing_feasible (sum over all subsets listed above), per num_gpus:")
    for g in sorted(by_gpu.keys()):
        print(f"  num_gpus={g}: {by_gpu[g]}")


if __name__ == "__main__":
    main()
