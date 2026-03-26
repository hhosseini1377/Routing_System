"""
RouterBench data loading utilities.

This module centralizes:
- loading + reordering RouterBench prompt-model score matrix `S`
- extracting latency-vs-load curves from `performance_data_*.json`
- loading per-model memory configuration from `profiler/model_memory/*.json`

The goal is to avoid duplicating parsing / ordering logic across scripts.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .models_config import ROUTING_MODELS, get_performance_data_paths, get_scores_path
from .models_config import get_model_memory_paths
from .optimize_fractions import PiecewiseLinearLatency


def load_scores(path: str | Path) -> np.ndarray:
    """
    Load `routerbench_0shot_scores.pkl` and return an (N,K) score matrix S.

    If the pickle contains `score_columns`, we reorder columns to match
    `ROUTING_MODELS` canonical order.
    """

    path = Path(path)
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if "scores" in data:
            raw_scores = data["scores"]
        elif "S" in data:
            raw_scores = data["S"]
        else:
            raise KeyError("routerbench_0shot_scores.pkl must contain 'scores' (or 'S').")
    else:
        # Some older artifacts stored just the matrix.
        raw_scores = data

    S = np.asarray(raw_scores, dtype=float)
    if S.ndim != 2:
        raise ValueError(f"scores must be 2D (N,K). Got shape={S.shape}")

    K_expected = len(ROUTING_MODELS)
    if S.shape[1] != K_expected:
        raise ValueError(
            f"Expected {K_expected} score columns (K={K_expected}), got S.shape={S.shape}"
        )

    if isinstance(data, dict) and "score_columns" in data:
        score_columns = list(data["score_columns"])
        col_to_idx = {c: i for i, c in enumerate(score_columns)}

        perm: list[int] = []
        for m in ROUTING_MODELS:
            if m.score_column not in col_to_idx:
                raise KeyError(
                    f"Missing expected score column '{m.score_column}' in pickle score_columns."
                )
            perm.append(col_to_idx[m.score_column])

        S = S[:, perm]

    return S


def _get_metric_value_from_item(item: dict[str, Any], metric: str) -> float | None:
    """
    Extract a scalar metric value from one performance_data JSON record.

    The JSON format differs slightly across artifacts (some use `result`,
    others store fields at the top-level). We support both.
    """

    base = item.get("result") if isinstance(item.get("result"), dict) else item
    perf = base.get("performance") if isinstance(base.get("performance"), dict) else {}
    metrics_after = base.get("metrics_after") if isinstance(base.get("metrics_after"), dict) else {}

    if metric == "tpot":
        tpot_list = base.get("tpot_ms_list") or item.get("tpot_ms_list")
        if tpot_list:
            return float(np.mean(tpot_list))
        v = perf.get("avg_tpot_ms")
        return float(v) if v is not None else None

    if metric == "avg_latency_ms":
        v = perf.get("avg_latency_ms")
        return float(v) if v is not None else None

    if metric == "p95_ttft":
        ttfts = base.get("ttfts_ms") or item.get("ttfts_ms")
        if ttfts:
            return float(np.percentile(ttfts, 95))
        v = perf.get("p95_ttft_ms")
        if v is not None:
            return float(v)
        v2 = metrics_after.get("p95_ttft_ms")
        return float(v2) if v2 is not None else None

    if metric == "p95_tpot":
        tpot_list = base.get("tpot_ms_list") or item.get("tpot_ms_list")
        if tpot_list:
            return float(np.percentile(tpot_list, 95))
        v = perf.get("p95_tpot_ms")
        if v is not None:
            return float(v)
        v2 = metrics_after.get("p95_tpot_ms")
        return float(v2) if v2 is not None else None

    raise ValueError(f"Unsupported metric: {metric}")


def extract_metric_vs_load_single_model(
    path: str | Path,
    tensor_parallel_size: int,
    thread_percentage: int,
    metric: str = "tpot",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (load_rps, metric_values) for a fixed backend setup.

    Returns
    -------
    loads_arr : (M,) increasing float array
    vals_arr : (M,) float array
    """

    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)

    by_load: dict[float, list[float]] = {}

    for item in data:
        if not isinstance(item, dict):
            continue

        setup = item.get("setup") or item.get("config") or {}
        if not isinstance(setup, dict):
            continue

        tp = int(setup.get("tensor_parallel_size", -1))
        threads = int(setup.get("thread_percentage", -1))
        if tp != tensor_parallel_size or threads != thread_percentage:
            continue

        # Most artifacts use `load_rps` in the setup/config dict.
        if "load_rps" not in setup:
            continue
        load_rps = float(setup["load_rps"])

        val = _get_metric_value_from_item(item, metric=metric)
        if val is None or not np.isfinite(val):
            continue

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


def build_latency_curves(
    tps: list[int] | tuple[int, ...],
    threads: list[int] | tuple[int, ...],
    performance_data_paths: list[str | Path],
    metric: str,
) -> list[PiecewiseLinearLatency]:
    """Build per-model latency curves for the provided (tp,thread) setup."""

    if len(tps) != len(threads) or len(tps) != len(performance_data_paths):
        raise ValueError("tps, threads, performance_data_paths must have same length")

    curves: list[PiecewiseLinearLatency] = []
    for path, tp, th in zip(performance_data_paths, tps, threads):
        loads, vals = extract_metric_vs_load_single_model(
            str(path),
            tensor_parallel_size=int(tp),
            thread_percentage=int(th),
            metric=metric,
        )
        curves.append(PiecewiseLinearLatency(load_grid=loads, latency_ms=vals))
    return curves


def build_latency_curves_for_three_models(
    tps: list[int] | tuple[int, ...],
    threads: list[int] | tuple[int, ...],
    metric: str,
    root: Path | str = ".",
) -> list[PiecewiseLinearLatency]:
    """Convenience wrapper for the canonical 3-backend routing setup."""

    if len(tps) != 3 or len(threads) != 3:
        raise ValueError("tps and threads must each have length 3 (one per backend)")

    paths = get_performance_data_paths(root)
    return build_latency_curves(tps=list(tps), threads=list(threads), performance_data_paths=paths, metric=metric)


def load_model_memory_config(path: str | Path) -> dict[str, float]:
    """
    Load per-model min GPU memory utilization as a function of TP.

    Expected JSON format:
      { "model": "...", "min_gpu_util_per_tp": { "1": 0.2, "2": 0.2, "4": 0.2 } }
    """

    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)

    mp = data.get("min_gpu_util_per_tp")
    if not isinstance(mp, dict):
        raise KeyError(f"Expected key 'min_gpu_util_per_tp' in {path}")
    return {str(k): float(v) for k, v in mp.items()}


def get_default_config(root: Path | str = ".") -> tuple[list[Path], list[Path]]:
    """
    Default resource files for the canonical 3-backend routing setup.

    Returns
    -------
    (model_memory_paths, performance_data_paths)
    """

    model_memory_paths = get_model_memory_paths(root)
    performance_data_paths = get_performance_data_paths(root)
    return model_memory_paths, performance_data_paths


__all__ = [
    "load_scores",
    "extract_metric_vs_load_single_model",
    "build_latency_curves",
    "build_latency_curves_for_three_models",
    "load_model_memory_config",
    "get_default_config",
]

