"""
2D bin packing feasibility check for GPU resource allocation.

Given K models with (tp_levels, thread_percentages, memory_percentages), we have
sum(tp_levels) shards. Each shard of model k requires:
  - memory_percentages[k] of a GPU's memory (fraction in [0, 1])
  - thread_percentages[k] of a GPU's CPU threads (fraction in [0, 1])

We pack shards into num_gpus bins (GPUs). Each GPU has capacity 1.0 in both dimensions.
A packing is feasible iff for every GPU:
  - sum(memory of shards on GPU) <= 1
  - sum(thread of shards on GPU) <= 1
  - at most one shard per model on each GPU (no co-location of same-model shards)

This is a 2D bin packing problem with model-separation constraint. We use First-Fit
Decreasing (FFD) to check feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Shard:
    """A single model shard with memory and thread requirements."""

    model_id: int
    memory: float  # fraction of GPU memory in [0, 1]
    thread: float  # fraction of GPU CPU threads in [0, 1]

    def size_sort_key(self) -> float:
        """Key for FFD: prefer placing larger items first (by max dimension)."""
        return max(self.memory, self.thread)


@dataclass
class FeasibilityResult:
    """Result of the feasibility check."""

    feasible: bool
    assignment: list[list[int]] = field(default_factory=list)
    """assignment[gpu_id] = list of shard indices placed on that GPU"""
    shards: list[Shard] = field(default_factory=list)
    """Ordered list of shards (indices match assignment references)"""
    reason: str = ""
    """If infeasible, a short explanation"""


def build_shards(
    tp_levels: np.ndarray | list[int],
    thread_percentages: np.ndarray | list[float],
    memory_percentages: np.ndarray | list[float],
) -> list[Shard]:
    """
    Build the list of shards from per-model arrays.

    Parameters
    ----------
    tp_levels : array of length K
        Tensor parallelism level for each model (number of shards per model).
    thread_percentages : array of length K
        CPU thread fraction per model, in [0, 1].
    memory_percentages : array of length K
        GPU memory fraction per model, in [0, 1].

    Returns
    -------
    List of Shard objects, one per model shard.
    """
    tp_levels = np.asarray(tp_levels, dtype=int)
    thread_percentages = np.asarray(thread_percentages, dtype=float)
    memory_percentages = np.asarray(memory_percentages, dtype=float)

    K = len(tp_levels)
    if len(thread_percentages) != K or len(memory_percentages) != K:
        raise ValueError(
            f"tp_levels, thread_percentages, memory_percentages must have same length K={K}"
        )

    shards: list[Shard] = []
    for k in range(K):
        mem = float(memory_percentages[k])
        thr = float(thread_percentages[k])
        for _ in range(tp_levels[k]):
            shards.append(Shard(model_id=k, memory=mem, thread=thr))
    return shards


def _pack_first_fit_decreasing(
    shards: list[Shard], num_gpus: int, no_same_model_per_gpu: bool = True
) -> tuple[bool, list[list[int]]]:
    """
    Pack shards into num_gpus bins using First-Fit Decreasing (2D).

    Sort shards by decreasing size (max of memory, thread), then place each
    in the first bin where constraints are satisfied.

    Parameters
    ----------
    no_same_model_per_gpu : bool, default True
        If True, at most one shard per model on each GPU (no co-location).

    Returns
    -------
    (feasible, assignment) where assignment[gpu_id] = list of shard indices.
    """
    if num_gpus <= 0:
        return False, []

    # Sort by decreasing size (larger items first)
    indexed = [(i, s) for i, s in enumerate(shards)]
    indexed.sort(key=lambda x: -x[1].size_sort_key())

    # Bins: (used_memory, used_thread, model_ids on GPU, list of shard indices)
    bins: list[tuple[float, float, set[int], list[int]]] = [
        (0.0, 0.0, set(), []) for _ in range(num_gpus)
    ]

    for shard_idx, shard in indexed:
        placed = False
        for b in range(num_gpus):
            used_mem, used_thr, model_ids, ids = bins[b]
            if no_same_model_per_gpu and shard.model_id in model_ids:
                continue
            new_mem = used_mem + shard.memory
            new_thr = used_thr + shard.thread
            if new_mem <= 1.0 and new_thr <= 1.0:
                bins[b] = (
                    new_mem,
                    new_thr,
                    model_ids | {shard.model_id},
                    ids + [shard_idx],
                )
                placed = True
                break
        if not placed:
            return False, []

    assignment: list[list[int]] = [list(ids) for _, _, _, ids in bins]
    return True, assignment


def check_feasibility(
    tp_levels: np.ndarray | list[int],
    thread_percentages: np.ndarray | list[float],
    memory_percentages: np.ndarray | list[float],
    num_gpus: int,
    no_same_model_per_gpu: bool = True,
) -> FeasibilityResult:
    """
    Check if there exists a feasible GPU allocation for the given setup.

    Parameters
    ----------
    tp_levels : array of length K
        Tensor parallelism level for each model.
    thread_percentages : array of length K
        CPU thread fraction per model [0, 1].
    memory_percentages : array of length K
        GPU memory fraction per model [0, 1].
    num_gpus : int
        Number of GPUs available.
    no_same_model_per_gpu : bool, default True
        If True, at most one shard per model on each GPU (tensor parallelism
        requires different GPUs for different shards of the same model).

    Returns
    -------
    FeasibilityResult with feasible flag, assignment (if feasible), and reason (if not).
    """
    # Validate inputs
    if num_gpus < 1:
        return FeasibilityResult(
            feasible=False,
            reason=f"num_gpus must be >= 1, got {num_gpus}",
        )

    try:
        shards = build_shards(tp_levels, thread_percentages, memory_percentages)
    except (ValueError, TypeError) as e:
        return FeasibilityResult(feasible=False, reason=str(e))

    if not shards:
        return FeasibilityResult(
            feasible=True,
            assignment=[[] for _ in range(num_gpus)],
            shards=[],
            reason="No shards to pack",
        )

    # Necessary conditions: total resource demand vs total capacity
    total_memory = sum(s.memory for s in shards)
    total_thread = sum(s.thread for s in shards)
    if total_memory > num_gpus:
        return FeasibilityResult(
            feasible=False,
            shards=shards,
            reason=f"Total memory demand {total_memory:.3f} > num_gpus {num_gpus}",
        )
    if total_thread > num_gpus:
        return FeasibilityResult(
            feasible=False,
            shards=shards,
            reason=f"Total thread demand {total_thread:.3f} > num_gpus {num_gpus}",
        )

    # Check no single shard exceeds bin capacity
    for i, s in enumerate(shards):
        if s.memory > 1.0 or s.thread > 1.0:
            return FeasibilityResult(
                feasible=False,
                shards=shards,
                reason=f"Shard {i} (model {s.model_id}) exceeds GPU capacity: "
                f"memory={s.memory:.3f}, thread={s.thread:.3f}",
            )

    # With no_same_model_per_gpu: each model k with tp=t needs t different GPUs
    if no_same_model_per_gpu:
        max_tp = int(np.max(tp_levels))
        if num_gpus < max_tp:
            return FeasibilityResult(
                feasible=False,
                shards=shards,
                reason=f"With no co-location, need at least max(tp_levels)={max_tp} GPUs, got {num_gpus}",
            )

    # Attempt 2D bin packing
    feasible, assignment = _pack_first_fit_decreasing(
        shards, num_gpus, no_same_model_per_gpu=no_same_model_per_gpu
    )

    if feasible:
        return FeasibilityResult(
            feasible=True,
            assignment=assignment,
            shards=shards,
        )
    else:
        return FeasibilityResult(
            feasible=False,
            shards=shards,
            reason="2D bin packing failed: no valid assignment found (FFD heuristic)",
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check GPU resource packing feasibility")
    parser.add_argument("--tp", type=int, nargs="+", required=True, help="TP level per model")
    parser.add_argument(
        "--threads",
        type=float,
        nargs="+",
        required=True,
        help="Thread fraction per model (0-1)",
    )
    parser.add_argument(
        "--memory",
        type=float,
        nargs="+",
        required=True,
        help="Memory fraction per model (0-1)",
    )
    parser.add_argument("--gpus", type=int, required=True, help="Number of GPUs")
    parser.add_argument(
        "--allow-same-model-per-gpu",
        action="store_true",
        help="Allow multiple shards of the same model on one GPU (default: no co-location)",
    )
    args = parser.parse_args()

    result = check_feasibility(
        tp_levels=args.tp,
        thread_percentages=args.threads,
        memory_percentages=args.memory,
        num_gpus=args.gpus,
        no_same_model_per_gpu=not args.allow_same_model_per_gpu,
    )

    print(f"Feasible: {result.feasible}")
    if result.reason:
        print(f"Reason: {result.reason}")
    if result.feasible and result.assignment:
        print("Assignment (shard indices per GPU):")
        for gpu_id, shard_ids in enumerate(result.assignment):
            if shard_ids:
                details = [
                    f"m{result.shards[i].model_id}(mem={result.shards[i].memory:.2f},thr={result.shards[i].thread:.2f})"
                    for i in shard_ids
                ]
                print(f"  GPU {gpu_id}: {shard_ids} -> {details}")
