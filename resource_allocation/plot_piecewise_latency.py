#!/usr/bin/env python3
"""
Utility to visualize piecewise-linear interpolation of a latency metric vs load
for a fixed setup (tensor_parallel_size, thread_percentage).

Example:
    python -m resource_allocation.plot_piecewise_latency \\
        --input performance_data_mistral_7b_final.json \\
        --tp 4 \\
        --threads 50 \\
        --metric tpot \\
        --output plots/tpot_tp4_threads50.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def extract_metric_vs_load(
    path: str,
    tensor_parallel_size: int,
    thread_percentage: int,
    metric: str = "tpot",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (loads, values) for a fixed (tp, thread_pct) across loads.
    """
    with open(path, "r") as f:
        data = json.load(f)

    loads: list[float] = []
    values: list[float] = []
    for item in data:
        setup = item.get("setup", {})
        result = item.get("result", {})
        perf = result.get("performance", {}) or {}

        if (
            int(setup.get("tensor_parallel_size", -1)) == tensor_parallel_size
            and int(setup.get("thread_percentage", -1)) == thread_percentage
        ):
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
                loads.append(load_rps)
                values.append(float(val))

    if not loads:
        raise ValueError(
            f"No matching setups found for tp={tensor_parallel_size}, threads={thread_percentage}"
        )

    loads_arr = np.asarray(loads, dtype=float)
    vals_arr = np.asarray(values, dtype=float)
    idx = np.argsort(loads_arr)
    return loads_arr[idx], vals_arr[idx]


def plot_piecewise_linear(
    json_path: str,
    tensor_parallel_size: int,
    thread_percentage: int,
    metric: str = "tpot",
    out_path: str | None = None,
) -> None:
    loads, vals = extract_metric_vs_load(
        json_path,
        tensor_parallel_size=tensor_parallel_size,
        thread_percentage=thread_percentage,
        metric=metric,
    )

    fig, ax = plt.subplots()
    ax.scatter(loads, vals, color="C0", alpha=0.7, label="observed")
    ax.plot(loads, vals, color="C1", lw=2, label="piecewise-linear interp")

    ax.set_xlabel("load_rps")
    ax.set_ylabel(metric)
    ax.set_title(
        f"{metric} vs load  (tp={tensor_parallel_size}, threads={thread_percentage}%)"
    )
    ax.legend()
    ax.grid(True, alpha=0.2)

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {out}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot piecewise-linear interpolation of a metric vs load for a fixed setup."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Performance JSON file (e.g., performance_data_mistral_7b_final.json)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        required=True,
        help="tensor_parallel_size for the setup to plot",
    )
    parser.add_argument(
        "--threads",
        type=int,
        required=True,
        help="thread_percentage for the setup to plot",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="tpot",
        choices=("tpot", "avg_latency_ms", "p95_ttft"),
        help="Metric to plot (default: tpot)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="If set, save plot to this path instead of showing it.",
    )
    args = parser.parse_args()

    plot_piecewise_linear(
        json_path=args.input,
        tensor_parallel_size=args.tp,
        thread_percentage=args.threads,
        metric=args.metric,
        out_path=args.output,
    )


if __name__ == "__main__":
    main()

