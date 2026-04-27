#!/usr/bin/env python3
"""
Plot TTFT-vs-load curves from a lookup pickle produced by
`python -m resource_allocation.build_ttft_lookup_tables`.

Supports:
  - flat pickle:   data[model_id][(tp, thread)][load_rps] = value
  - nested pickle: data[model_id][metric][(tp, thread)][load_rps] = value
    (from `build_ttft_lookup_tables --metrics ttft p95_ttft`)

Examples:
  python notebooks/plot_latencies.py --metric ttft --output plots/ttft.pdf
  python notebooks/plot_latencies.py --input outputs/ttft_lookup.pkl --metric p95_ttft
  python notebooks/plot_latencies.py --max-load 25 --metric p95_ttft
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRIC_YLABEL: dict[str, str] = {
    "ttft": "Mean TTFT (ms)",
    "avg_ttft": "Mean TTFT (ms)",
    "p95_ttft": "P95 TTFT (ms)",
    "tpot": "TPOT (ms)",
    "avg_latency_ms": "Avg latency (ms)",
    "p95_tpot": "P95 TPOT (ms)",
}


def get_model_table(
    data: dict,
    model_id: str,
    metric: str,
) -> dict[tuple[int, int], dict[float, float]]:
    """
    Return {(tp, thread_pct): {load_rps: value}} for one model.

    If the pickle is flat (one metric only), *metric* is ignored.
    If nested (multiple metrics), select *metric*.
    """
    m = data[model_id]
    if not m:
        raise KeyError(f"Empty table for model {model_id!r}")
    k0 = next(iter(m.keys()))
    if isinstance(k0, tuple):
        return m
    if isinstance(k0, str):
        if metric not in m:
            raise KeyError(
                f"Model {model_id!r} has metrics {sorted(m.keys())!r}; "
                f"requested {metric!r}"
            )
        return m[metric]
    raise ValueError(f"Unexpected lookup structure under {model_id!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot TTFT (or other metric) vs load from lookup pickle.")
    p.add_argument(
        "--input",
        "-i",
        type=str,
        default="outputs/ttft_lookup.pkl",
        help="Pickle from build_ttft_lookup_tables.",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="ttft",
        choices=tuple(METRIC_YLABEL.keys()),
        help="Which series to plot if the pickle is nested; also sets y-axis label. Default: ttft (mean TTFT).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output PDF path (default: plots/ttft_overlay_<slug>.pdf).",
    )
    p.add_argument(
        "--log-y",
        action="store_true",
        help="Use logarithmic y-axis (values must be strictly positive).",
    )
    p.add_argument(
        "--max-load",
        type=float,
        default=None,
        metavar="RPS",
        help="Only use measured loads strictly below this RPS (omit flag for all loads).",
    )
    return p.parse_args()


# Default overlay: edit or pass via env / extend script
DEFAULT_PANELS: list[tuple[str, int, int]] = [
    ("yi-34b", 4, 30),
    ("yi-34b", 4, 50),
    ("yi-34b", 4, 70),
    ("yi-34b", 2, 30),
    ("yi-34b", 2, 50),
    ("yi-34b", 2, 70),
]


def main() -> None:
    args = parse_args()
    metric = args.metric

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    panels = DEFAULT_PANELS

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "lines.linewidth": 1.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    n_series = len(panels)
    fig, ax = plt.subplots(figsize=(3.6, 2.65))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_series, 3)))

    for i, (model_id, tp_level, thread_percentage_level) in enumerate(panels):
        key = (tp_level, thread_percentage_level)
        model_data = get_model_table(data, model_id, metric)
        if key not in model_data:
            raise KeyError(
                f"Missing lookup for {model_id} at {key}. "
                f"Available keys: {sorted(model_data.keys())}"
            )

        load_to_vals = model_data[key]
        loads = np.asarray(sorted(load_to_vals.keys()), dtype=float)
        y_ms = np.asarray([load_to_vals[float(l)] for l in loads], dtype=float)

        if args.max_load is not None:
            mask = loads < float(args.max_load)
            loads = loads[mask]
            y_ms = y_ms[mask]
        if loads.size == 0:
            hint = (
                f" (no loads < {args.max_load} RPS)"
                if args.max_load is not None
                else ""
            )
            raise ValueError(f"No load points for {model_id} {key}{hint}.")

        if loads.size >= 2:
            x_dense = np.linspace(loads.min(), loads.max(), 400)
            y_interp = np.interp(x_dense, loads, y_ms)
        else:
            x_dense = loads
            y_interp = y_ms

        c = colors[i % len(colors)]
        label = f"{model_id}, tp={tp_level}, threads={thread_percentage_level}%"

        ax.plot(
            x_dense,
            y_interp,
            color=c,
            linestyle="-",
            linewidth=1.8,
            label=label,
        )
        ax.plot(
            loads,
            y_ms,
            linestyle="none",
            marker="o",
            markersize=3.8,
            markerfacecolor="white",
            markeredgecolor=c,
            markeredgewidth=1.0,
            zorder=3,
        )

    ax.set_xlabel("Input load (RPS)")
    ax.set_ylabel(METRIC_YLABEL.get(metric, metric))
    if args.log_y:
        ax.set_yscale("log")
    ax.grid(True, axis="both", linestyle=":", linewidth=0.6, alpha=0.3)
    ax.set_axisbelow(True)

    ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        facecolor="white",
        framealpha=1.0,
        fontsize=8,
        borderpad=0.35,
        handlelength=2.0,
        handletextpad=0.45,
        labelspacing=0.35,
    )

    plt.tight_layout(pad=0.2)

    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        slug = "_".join(f"{m}_tp{t}_th{th}" for m, t, th in panels)
        out_path = out_dir / f"ttft_overlay_{metric}_{slug}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
