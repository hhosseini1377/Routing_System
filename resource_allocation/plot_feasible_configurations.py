"""
Plot feasible configuration counts versus num_gpus.

For a fixed num_models and option sets, this script sweeps GPU counts and plots
the number of feasible (tp, thread) configurations under either:

- weighted (default):   sum_i tp_i * thread_pct_i == 100 * num_gpus
- unweighted (optional): sum_i thread_pct_i == 100 * num_gpus
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .count_thread_sum_configs import count_configurations


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot feasible configuration count vs number of GPUs."
    )
    p.add_argument("--num-models", type=int, required=True, help="Number of models (N).")
    p.add_argument(
        "--tp-options",
        type=int,
        nargs="+",
        required=True,
        help="Allowed TP values per model (e.g. 1 2 4).",
    )
    p.add_argument(
        "--thread-options",
        type=int,
        nargs="+",
        required=True,
        help="Allowed thread percentages per model (e.g. 10 20 ... 100).",
    )
    p.add_argument(
        "--num-gpus-list",
        type=int,
        nargs="+",
        default=None,
        help="Explicit GPU values to evaluate (e.g. 1 2 3 4 5 6).",
    )
    p.add_argument(
        "--min-gpus",
        type=int,
        default=None,
        help="Minimum GPU count for range sweep (used when --num-gpus-list is not set).",
    )
    p.add_argument(
        "--max-gpus",
        type=int,
        default=None,
        help="Maximum GPU count for range sweep (used when --num-gpus-list is not set).",
    )
    p.add_argument(
        "--step-gpus",
        type=int,
        default=1,
        help="GPU step for range sweep (default: 1).",
    )
    p.add_argument(
        "--unweighted",
        action="store_true",
        help="Use unweighted constraint sum(thread_pct)=100*num_gpus.",
    )
    p.add_argument(
        "--log-y",
        action="store_true",
        help="Use logarithmic y-axis.",
    )
    p.add_argument(
        "--paper-ready",
        action="store_true",
        help="Apply IEEE-friendly styling defaults for compact subplot figures.",
    )
    p.add_argument(
        "--fig-width",
        type=float,
        default=None,
        help="Figure width in inches (overrides defaults).",
    )
    p.add_argument(
        "--fig-height",
        type=float,
        default=None,
        help="Figure height in inches (overrides defaults).",
    )
    p.add_argument(
        "--font-size",
        type=float,
        default=None,
        help="Base font size in points (overrides defaults).",
    )
    p.add_argument(
        "--line-width",
        type=float,
        default=None,
        help="Line width for the curve.",
    )
    p.add_argument(
        "--marker-size",
        type=float,
        default=None,
        help="Marker size for points.",
    )
    p.add_argument(
        "--no-title",
        action="store_true",
        help="Omit plot title (useful for paper subfigures).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="plots/feasible_configurations_vs_gpus.png",
        help="Output plot path. Pass empty string to skip writing a plot.",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV output path with (num_gpus, feasible_count).",
    )
    return p


def _resolve_gpu_list(args: argparse.Namespace) -> list[int]:
    if args.num_gpus_list is not None:
        vals = [int(x) for x in args.num_gpus_list]
        if not vals:
            raise ValueError("--num-gpus-list cannot be empty")
        if any(v < 0 for v in vals):
            raise ValueError("All --num-gpus-list values must be >= 0")
        return vals

    if args.min_gpus is None or args.max_gpus is None:
        raise ValueError(
            "Provide either --num-gpus-list OR both --min-gpus and --max-gpus."
        )
    if args.step_gpus <= 0:
        raise ValueError("--step-gpus must be > 0")
    if args.min_gpus < 0 or args.max_gpus < 0:
        raise ValueError("--min-gpus and --max-gpus must be >= 0")
    if args.min_gpus > args.max_gpus:
        raise ValueError("--min-gpus must be <= --max-gpus")

    return list(range(args.min_gpus, args.max_gpus + 1, args.step_gpus))


def main() -> None:
    args = build_arg_parser().parse_args()

    num_models = int(args.num_models)
    tp_options = [int(x) for x in args.tp_options]
    thread_options = [int(x) for x in args.thread_options]
    unweighted = bool(args.unweighted)
    gpu_values = _resolve_gpu_list(args)

    feasible_counts: list[int] = []
    for g in gpu_values:
        cnt = count_configurations(
            num_models=num_models,
            tp_options=tp_options,
            thread_options=thread_options,
            num_gpus=g,
            unweighted=unweighted,
        )
        feasible_counts.append(int(cnt))

    if args.output:
        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required to save a plot. Install it (e.g. pip install matplotlib), "
                "or pass --output '' to skip plotting and only print/save counts."
            ) from exc

        if args.paper_ready:
            # IEEE-friendly defaults for side-by-side figures in one column.
            width = args.fig_width if args.fig_width is not None else 1.75
            height = args.fig_height if args.fig_height is not None else 1.35
            font_size = args.font_size if args.font_size is not None else 7.0
            line_width = args.line_width if args.line_width is not None else 1.0
            marker_size = args.marker_size if args.marker_size is not None else 2.8
        else:
            width = args.fig_width if args.fig_width is not None else 6.0
            height = args.fig_height if args.fig_height is not None else 3.6
            font_size = args.font_size if args.font_size is not None else 10.0
            line_width = args.line_width if args.line_width is not None else 1.5
            marker_size = args.marker_size if args.marker_size is not None else 4.0

        mpl.rcParams.update(
            {
                "font.size": font_size,
                "axes.labelsize": font_size,
                "axes.titlesize": font_size,
                "xtick.labelsize": max(5.0, font_size - 1.0),
                "ytick.labelsize": max(5.0, font_size - 1.0),
                "legend.fontsize": max(5.0, font_size - 1.0),
                "axes.linewidth": 0.7 if args.paper_ready else 0.9,
                "xtick.major.width": 0.6 if args.paper_ready else 0.8,
                "ytick.major.width": 0.6 if args.paper_ready else 0.8,
                "xtick.major.size": 2.5 if args.paper_ready else 3.5,
                "ytick.major.size": 2.5 if args.paper_ready else 3.5,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.01 if args.paper_ready else 0.03,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
            }
        )

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(width, height))
        ax.plot(
            gpu_values,
            feasible_counts,
            marker="o",
            linewidth=line_width,
            markersize=marker_size,
            color="#1f77b4",
        )
        ax.set_xlim(min(gpu_values), max(gpu_values))
        ax.set_xticks(gpu_values)
        if args.paper_ready and len(gpu_values) > 8:
            # Keep tick labels readable for compact paper figures.
            for idx, label in enumerate(ax.get_xticklabels()):
                if idx % 2 == 1:
                    label.set_visible(False)
        ax.set_xlabel("GPUs")
        ax.set_ylabel("Feasible configs")
        mode = "unweighted" if unweighted else "TP-weighted"
        if not args.no_title:
            ax.set_title(f"{mode}")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        if args.log_y:
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(out_path, dpi=300 if args.paper_ready else 200)
        plt.close(fig)
        print(f"Saved plot to: {out_path}")

    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["num_gpus", "feasible_count"])
            for g, c in zip(gpu_values, feasible_counts):
                writer.writerow([g, c])

    if args.output_csv:
        print(f"Saved CSV to: {args.output_csv}")
    print("num_gpus -> feasible_count:")
    for g, c in zip(gpu_values, feasible_counts):
        print(f"  {g} -> {c}")


if __name__ == "__main__":
    main()

