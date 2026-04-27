"""
Count (tp, thread_percentage) configurations across N models.

Problem statement
-----------------
Inputs:
- number of models (N)
- tp options each model can take
- thread options each model can take (integer percentages, e.g. 10..100)
- number of GPUs (G)

Enumerate all configurations for all models (each model picks one (tp, thread) pair)
and count how many satisfy:

    sum_i tp_i * thread_percentage_i == 100 * G

Notes
-----
- This script intentionally does *not* use ROUTING_MODELS and does not assume K=3.
- The equality is computed in integer "percentage points" (no floats).
- Use --unweighted to instead enforce sum_i thread_percentage_i == 100 * G.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Sequence


def count_thread_sum_combinations(
    *,
    num_models: int,
    thread_options: Sequence[int],
    target_thread_sum: int,
) -> int:
    """
    Count length-N sequences from thread_options whose sum equals target_thread_sum.

    Uses dynamic programming with multiplicities to avoid brute force enumeration.
    """
    if num_models < 0:
        raise ValueError(f"num_models must be >= 0, got {num_models}")
    thread_options = [int(x) for x in thread_options]
    if any(x < 0 for x in thread_options):
        raise ValueError("thread_options must be non-negative integers")

    # dp[sum] = number of ways to achieve 'sum' after processing i models.
    dp: Counter[int] = Counter()
    dp[0] = 1
    for _ in range(num_models):
        nxt: Counter[int] = Counter()
        for s, cnt in dp.items():
            for th in thread_options:
                ns = s + th
                if ns <= target_thread_sum:
                    nxt[ns] += cnt
        dp = nxt
        if not dp:
            return 0
    return int(dp.get(target_thread_sum, 0))

def count_weighted_thread_sum_combinations(
    *,
    num_models: int,
    tp_options: Sequence[int],
    thread_options: Sequence[int],
    target_weighted_sum: int,
) -> int:
    """
    Count length-N sequences of (tp, thread) pairs whose weighted sum matches target.

      sum_i tp_i * thread_i == target_weighted_sum
    """
    if num_models < 0:
        raise ValueError(f"num_models must be >= 0, got {num_models}")
    tp_options = [int(x) for x in tp_options]
    thread_options = [int(x) for x in thread_options]
    if any(tp <= 0 for tp in tp_options):
        raise ValueError("tp_options must be positive integers")
    if any(th < 0 for th in thread_options):
        raise ValueError("thread_options must be non-negative integers")

    dp: Counter[int] = Counter()
    dp[0] = 1
    for _ in range(num_models):
        nxt: Counter[int] = Counter()
        for s, cnt in dp.items():
            for tp in tp_options:
                for th in thread_options:
                    ns = s + (tp * th)
                    if ns <= target_weighted_sum:
                        nxt[ns] += cnt
        dp = nxt
        if not dp:
            return 0
    return int(dp.get(target_weighted_sum, 0))


def count_configurations(
    *,
    num_models: int,
    tp_options: Sequence[int],
    thread_options: Sequence[int],
    num_gpus: int,
    unweighted: bool = False,
) -> int:
    """
    Count (tp,thread) configurations for N models satisfying the thread-sum constraint.

    - Default (weighted): sum_i tp_i * thread_pct_i == 100 * num_gpus
    - Unweighted (--unweighted): sum_i thread_pct_i == 100 * num_gpus
    """
    num_models = int(num_models)
    num_gpus = int(num_gpus)
    if num_models < 0:
        raise ValueError(f"num_models must be >= 0, got {num_models}")
    if num_gpus < 0:
        raise ValueError(f"num_gpus must be >= 0, got {num_gpus}")

    tp_options = [int(x) for x in tp_options]
    thread_options = [int(x) for x in thread_options]
    if not tp_options:
        raise ValueError("tp_options must be non-empty")
    if not thread_options:
        raise ValueError("thread_options must be non-empty")

    target_sum = 100 * num_gpus
    if unweighted:
        # Count thread sequences and multiply by TP choices per model.
        thread_ways = count_thread_sum_combinations(
            num_models=num_models,
            thread_options=thread_options,
            target_thread_sum=target_sum,
        )
        return int((len(tp_options) ** num_models) * thread_ways)

    # Weighted: DP directly over (tp,thread) pair choices.
    return int(
        count_weighted_thread_sum_combinations(
            num_models=num_models,
            tp_options=tp_options,
            thread_options=thread_options,
            target_weighted_sum=target_sum,
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Count (tp,thread_percentage) configurations across N models where "
            "sum(tp_i*thread_percentage_i) == 100*num_gpus (default)."
        )
    )
    p.add_argument("--num-models", type=int, required=True, help="Number of models (N).")
    p.add_argument(
        "--tp-options",
        type=int,
        nargs="+",
        required=True,
        help="Allowed TP values for each model (e.g. 1 2 4).",
    )
    p.add_argument(
        "--thread-options",
        type=int,
        nargs="+",
        required=True,
        help="Allowed thread_percentage values for each model (e.g. 10 20 ... 100).",
    )
    p.add_argument("--num-gpus", type=int, required=True, help="Number of GPUs (G).")
    p.add_argument(
        "--unweighted",
        action="store_true",
        help="Use unweighted constraint sum(thread_percentage)==100*num_gpus.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    n = int(args.num_models)
    g = int(args.num_gpus)
    tp_opts = list(map(int, args.tp_options))
    th_opts = list(map(int, args.thread_options))

    total_configs = (len(tp_opts) * len(th_opts)) ** n
    feasible = count_configurations(
        num_models=n,
        tp_options=tp_opts,
        thread_options=th_opts,
        num_gpus=g,
        unweighted=bool(args.unweighted),
    )

    print(f"num_models={n} num_gpus={g}")
    print(f"tp_options={tp_opts}")
    print(f"thread_options={th_opts}")
    print(f"Total configurations: {total_configs}")
    if args.unweighted:
        print(f"Feasible configurations (sum(thread_pct)=100*num_gpus): {feasible}")
    else:
        print(f"Feasible configurations (sum(tp*thread_pct)=100*num_gpus): {feasible}")


if __name__ == "__main__":
    main()

