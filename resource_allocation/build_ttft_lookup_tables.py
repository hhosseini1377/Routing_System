#!/usr/bin/env python3
"""
Build `{load_rps: TTFT_ms}` lookup tables for each (tp_level, thread_percentage_level)
from the per-model `performance_data_*.json` artifacts.

Example:
  python -m resource_allocation.build_ttft_lookup_tables \
    --root . \
    --tp-options 1 2 4 \
    --thread-options 10 20 30 40 50 \
    --metric p95_ttft \
    --output outputs/ttft_lookup.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from .models_config import ROUTING_MODELS, get_performance_data_paths
from .routerbench_data import build_metric_vs_load_lookup_tables_for_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create load->TTFT lookup tables for each (tp, thread) setup."
    )
    p.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root containing performance_data_*.json.",
    )
    p.add_argument(
        "--tp-options",
        type=int,
        nargs="*",
        default=[1, 2, 4],
        help="Candidate tensor_parallel_size values.",
    )
    p.add_argument(
        "--thread-options",
        type=int,
        nargs="*",
        default=list(range(10, 101, 10)),
        help="Candidate thread_percentage values (0..100).",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="p95_ttft",
        choices=("tpot", "avg_latency_ms", "avg_ttft", "p95_ttft", "p95_tpot"),
        help="Metric to extract: avg_ttft = mean(ttfts_ms); default p95_ttft.",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Where to write the lookup tables (.pkl or .json).",
    )
    p.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=("auto", "pickle", "json"),
        help="Output serialization format (default: auto from --output extension).",
    )
    return p.parse_args()


def _to_json_serializable(
    lookup: dict[str, dict[tuple[int, int], dict[float, float]]]
) -> dict[str, Any]:
    """
    Convert to JSON-serializable structure.

    - (tp, thread) keys -> "tp=<tp>,thread=<thread>"
    - load_rps float keys -> strings
    """
    out: dict[str, Any] = {}
    for model_id, by_pair in lookup.items():
        out_pairs: dict[str, Any] = {}
        for (tp, th), load_to_val in by_pair.items():
            pair_key = f"tp={tp},thread={th}"
            out_pairs[pair_key] = {str(load): val for load, val in load_to_val.items()}
        out[model_id] = out_pairs
    return out


def main() -> None:
    args = _parse_args()
    root = Path(args.root)

    performance_paths = get_performance_data_paths(root)
    if len(performance_paths) != len(ROUTING_MODELS):
        raise RuntimeError("Unexpected models/performance_paths mismatch.")

    lookup_by_model: dict[str, dict[tuple[int, int], dict[float, float]]] = {}
    for model_spec, perf_path in zip(ROUTING_MODELS, performance_paths):
        lookup_by_model[model_spec.model_id] = build_metric_vs_load_lookup_tables_for_model(
            performance_data_path=perf_path,
            tp_options=args.tp_options,
            thread_options=args.thread_options,
            metric=args.metric,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fmt = args.format
    if fmt == "auto":
        fmt = "json" if output.suffix.lower() == ".json" else "pickle"

    if fmt == "pickle":
        with open(output, "wb") as f:
            pickle.dump(lookup_by_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote pickle lookup tables to {output}")
        return

    # json
    json_payload = _to_json_serializable(lookup_by_model)
    with open(output, "w") as f:
        json.dump(json_payload, f, indent=2, sort_keys=True)
    print(f"Wrote JSON lookup tables to {output}")


if __name__ == "__main__":
    main()

