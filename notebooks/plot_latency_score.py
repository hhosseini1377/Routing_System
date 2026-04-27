"""
Score vs latency scatter for brute-force pickles (same logic as plot_latency_score.ipynb).

Overlays (when computable from feasible full deployments):
  - Equal-Split: ``find_most_equal_thread_split`` (min variance of TP×thread demand).
  - Size-Proportional: ``find_most_size_proportional_thread_split`` (min MSE to size shares).
  - Isolated: ``find_max_score_no_shared_gpu`` (highest score among no shared-GPU packings).

Run on all results:
  python notebooks/plot_latency_score.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Allow `python notebooks/plot_latency_score.py` (repo root is parent of notebooks/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from resource_allocation.brute_force_setup import BruteForceResult, SetupCandidate, load_result
from resource_allocation.models_config import ROUTING_MODELS
from resource_allocation.resource_packing import check_feasibility


def repo_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "resource_allocation").is_dir():
        return cwd
    if (cwd.parent / "resource_allocation").is_dir():
        return cwd.parent
    return cwd


def parse_brute_force_path(path: str | Path) -> tuple[int, float, int]:
    """Parse G, λ, τ from `brute_force_{G}g_lam{λ}_tau{τ}.pkl`."""
    name = Path(path).name
    m = re.match(r"brute_force_(\d+)g_lam([\d.]+)_tau(\d+)\.pkl$", name)
    if not m:
        raise ValueError(
            "Expected filename brute_force_<G>g_lam<λ>_tau<τ>.pkl, got: " + repr(name)
        )
    g, lam_s, tau_s = m.groups()
    return int(g), float(lam_s), int(tau_s)


def infer_run_params(
    path: str | Path,
    data: BruteForceResult,
    *,
    default_num_gpus: int,
) -> tuple[int, float, int]:
    """
    (num_gpus, λ RPS load, τ ms) from filename, else optional attrs on ``data``, else error.

    Supports ``brute_force_{G}g_lam{λ}_tau{τ}.pkl`` and ``brute_force_lam{λ}_tau{τ}.pkl``
    (uses ``default_num_gpus`` for the latter).
    """
    name = Path(path).name
    m = re.match(r"brute_force_(\d+)g_lam([\d.]+)_tau(\d+)\.pkl$", name)
    if m:
        return int(m.group(1)), float(m.group(2)), int(m.group(3))
    m = re.match(r"brute_force_lam([\d.]+)_tau(\d+)\.pkl$", name)
    if m:
        return default_num_gpus, float(m.group(1)), int(m.group(2))

    g = getattr(data, "num_gpus", None)
    lam = getattr(data, "lambda_global", None)
    tau = getattr(data, "tau", None)
    if g is not None and lam is not None and tau is not None:
        return int(g), float(lam), int(tau)

    raise ValueError(
        f"Cannot infer num_gpus, λ, τ from filename {name!r} or pickle attributes; "
        "use brute_force_<G>g_lam<λ>_tau<τ>.pkl or brute_force_lam<λ>_tau<τ>.pkl"
    )


def _thread_int_to_frac(thread_percentage: int) -> float:
    return float(thread_percentage) / 100.0


def find_most_equal_thread_split(
    candidates: list[SetupCandidate],
) -> tuple[SetupCandidate, float, float]:
    if not candidates:
        raise ValueError("candidates is empty")

    best: SetupCandidate | None = None
    best_key: tuple | None = None

    for c in candidates:
        if len(c.subset) != len(ROUTING_MODELS):
            continue
        th_frac = np.asarray(
            [_thread_int_to_frac(int(t)) for t in c.thread_levels], dtype=float
        )
        tp = np.asarray(c.tp_levels, dtype=float)
        if th_frac.size == 0:
            continue
        total_demand = tp * th_frac
        var_tot = float(np.var(total_demand))
        spread_tot = float(np.max(total_demand) - np.min(total_demand))
        var_tp = float(np.var(tp)) if tp.size else 0.0
        key = (var_tot, spread_tot, var_tp)
        if best is None or key < best_key:  # type: ignore[operator]
            best = c
            best_key = key

    if best is None:
        raise ValueError("no full-deployment candidate with thread_levels")

    assert best.best_score is not None and best.best_latency is not None
    return best, float(best.best_score), float(best.best_latency)


_MODEL_PARAM_BILLIONS_BY_ID: dict[str, float] = {
    "mistral-7b": 7.0,
    "yi-34b": 34.0,
    "vicuna-13b": 13.0,
}


def find_most_size_proportional_thread_split(
    candidates: list[SetupCandidate],
) -> tuple[SetupCandidate, float, float]:
    if not candidates:
        raise ValueError("candidates is empty")

    best: SetupCandidate | None = None
    best_key: tuple | None = None

    for c in candidates:
        if len(c.subset) != len(ROUTING_MODELS):
            continue
        th_frac = np.asarray(
            [_thread_int_to_frac(int(t)) for t in c.thread_levels], dtype=float
        )
        tp = np.asarray(c.tp_levels, dtype=float)
        if th_frac.size == 0:
            continue
        try:
            sizes = np.array(
                [_MODEL_PARAM_BILLIONS_BY_ID[ROUTING_MODELS[i].model_id] for i in c.subset],
                dtype=float,
            )
        except KeyError:
            continue
        if sizes.size != th_frac.size:
            continue
        demand = tp * th_frac
        ssum = float(sizes.sum())
        dsum = float(demand.sum())
        if ssum <= 0.0 or dsum <= 0.0:
            continue
        ideal = sizes / ssum
        obs = demand / dsum
        mse = float(np.sum((obs - ideal) ** 2))
        l1 = float(np.sum(np.abs(obs - ideal)))
        key = (mse, l1)
        if best is None or key < best_key:  # type: ignore[operator]
            best = c
            best_key = key

    if best is None:
        raise ValueError(
            "no full-deployment candidate with valid thread_levels / subset"
        )

    assert best.best_score is not None and best.best_latency is not None
    return best, float(best.best_score), float(best.best_latency)


def packing_has_at_most_one_shard_per_gpu(
    candidate: SetupCandidate, num_gpus: int
) -> bool:
    thread_fracs = [_thread_int_to_frac(th) for th in candidate.thread_levels]
    feas = check_feasibility(
        tp_levels=np.asarray(candidate.tp_levels, dtype=int),
        thread_percentages=thread_fracs,
        memory_percentages=np.asarray(candidate.memory_levels, dtype=float),
        num_gpus=num_gpus,
    )
    if not feas.feasible:
        return False
    return all(len(shard_ids) <= 1 for shard_ids in feas.assignment)


def find_max_score_no_shared_gpu(
    candidates: list[SetupCandidate],
    num_gpus: int,
) -> tuple[SetupCandidate, float, float]:
    filtered = [
        c
        for c in candidates
        if len(c.subset) == len(ROUTING_MODELS)
        and packing_has_at_most_one_shard_per_gpu(c, num_gpus)
    ]
    if not filtered:
        raise ValueError(
            "no full-deployment feasible setup with at most one shard per GPU "
            "(no shared-GPU packing)"
        )
    best = max(
        filtered,
        key=lambda c: float(c.best_score) if c.best_score is not None else -1e300,
    )
    assert best.best_score is not None and best.best_latency is not None
    return best, float(best.best_score), float(best.best_latency)


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_score_latency_figure(
    result_path: str | Path,
    *,
    out_dir: Path | None = None,
    close_fig: bool = True,
    default_num_gpus: int = 8,
) -> Path:
    """
    Build the score–latency scatter (feasible points, τ line, baselines) and save PDF.

    Overlays one point each when available: most equal thread split, most size-proportional
    split, and highest-score isolated (no shared GPU) setup.

    Output: ``<out_dir>/score_latency_scatter_<pickle_stem>.pdf`` (default ``plots/``).
    """
    result_path = Path(result_path).resolve()
    if out_dir is None:
        out_dir = repo_root() / "plots"
    else:
        out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_result(path=result_path)
    gpu_count, input_load, latency_threshold = infer_run_params(
        result_path, data, default_num_gpus=default_num_gpus
    )
    feasible_data = [
        row
        for row in data.all_feasible
        if row.best_latency is not None and float(row.best_latency) >= 0.0
    ]

    scores = np.array([row.best_score for row in feasible_data], dtype=float)
    latencies = np.array([row.best_latency for row in feasible_data], dtype=float)

    if len(feasible_data) == 0:
        raise ValueError(f"No feasible setups (with latency >= 0) in {result_path}")

    best_idx = int(np.argmax(scores))
    bf_score = float(scores[best_idx])
    bf_latency = float(latencies[best_idx])

    eq: tuple[float, float] | None = None
    prop: tuple[float, float] | None = None
    noshare: tuple[float, float] | None = None

    try:
        _, eq_s, eq_l = find_most_equal_thread_split(feasible_data)
        eq = (eq_s, eq_l)
    except ValueError:
        pass
    try:
        _, prop_s, prop_l = find_most_size_proportional_thread_split(feasible_data)
        prop = (prop_s, prop_l)
    except ValueError:
        pass
    try:
        _, ns_s, ns_l = find_max_score_no_shared_gpu(feasible_data, gpu_count)
        noshare = (ns_s, ns_l)
    except ValueError:
        pass

    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.4))

    ax.scatter(
        scores,
        latencies,
        s=16,
        alpha=0.8,
        zorder=2,
        label="Retained setups",
    )

    ax.axhline(
        latency_threshold,
        linestyle="--",
        linewidth=1.0,
        color="black",
        zorder=1,
        label=r"Latency target $\tau$",
    )
    if eq is not None:
        ax.scatter(
            [eq[0]],
            [eq[1]],
            marker="s",
            s=52,
            color="tab:green",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            label="Equal-Split",
        )
    if prop is not None:
        ax.scatter(
            [prop[0]],
            [prop[1]],
            marker="^",
            s=52,
            color="tab:purple",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            label="Size-Proportional",
        )
    if noshare is not None:
        ax.scatter(
            [noshare[0]],
            [noshare[1]],
            marker="D",
            s=48,
            color="tab:orange",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            label="Isolated ",
        )

    ax.scatter(
        [bf_score],
        [bf_latency],
        marker="*",
        s=90,
        color="red",
        edgecolors="black",
        linewidths=0.5,
        zorder=10,
        label="Best setup",
    )


    ax.set_xlabel("Best achievable score")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(
        f"Retained setups",
        pad=4,
    )

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    extra_s: list[float] = [bf_score]
    extra_l: list[float] = [bf_latency]
    if eq is not None:
        extra_s.append(eq[0])
        extra_l.append(eq[1])
    if prop is not None:
        extra_s.append(prop[0])
        extra_l.append(prop[1])
    if noshare is not None:
        extra_s.append(noshare[0])
        extra_l.append(noshare[1])

    extra_s_arr = np.asarray(extra_s, dtype=float)
    extra_l_arr = np.asarray(extra_l, dtype=float)

    all_s = np.concatenate([scores, extra_s_arr])
    all_l = np.concatenate([latencies, extra_l_arr])
    x_margin = 0.02 * (all_s.max() - all_s.min() + 1e-8)
    y_margin = 0.03 * (all_l.max() - all_l.min() + 1e-8)
    ax.set_xlim(all_s.min() - x_margin, all_s.max() + x_margin)
    ax.set_ylim(all_l.min() - y_margin, all_l.max() + y_margin)

    ax.legend(frameon=False, loc="lower left")
    plt.tight_layout(pad=0.3)

    stem = result_path.stem
    out_path = out_dir / f"score_latency_scatter_{stem}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    if close_fig:
        plt.close(fig)
    return out_path


def iter_brute_force_pickles(results_dir: Path | None = None) -> list[Path]:
    root = repo_root()
    d = results_dir if results_dir is not None else root / "resource_allocation" / "brute_force_results"
    if not d.is_dir():
        return []
    return sorted(d.glob("*.pkl"))


def plot_all_brute_force_results(
    results_dir: Path | None = None,
    out_dir: Path | None = None,
    *,
    default_num_gpus: int = 8,
) -> list[Path]:
    """Generate score–latency PDFs for every ``*.pkl`` under ``results_dir``."""
    root = repo_root()
    os.chdir(root)
    if out_dir is None:
        out_dir = root / "plots"
    paths = iter_brute_force_pickles(results_dir)
    saved: list[Path] = []
    for p in paths:
        try:
            out = plot_score_latency_figure(
                p,
                out_dir=out_dir,
                close_fig=True,
                default_num_gpus=default_num_gpus,
            )
            saved.append(out)
            print(f"Saved: {out}")
        except Exception as e:
            print(f"Skip {p.name}: {e}", file=sys.stderr)
    return saved


def main() -> None:
    root = repo_root()
    os.chdir(root)
    if not (root / "resource_allocation").is_dir():
        print("Run this script from the repository root (or notebooks/).", file=sys.stderr)
        sys.exit(1)
    ap = argparse.ArgumentParser(description="Plot score vs latency for brute-force pickles.")
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing *.pkl (default: resource_allocation/brute_force_results)",
    )
    ap.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Output directory for PDFs (default: plots/)",
    )
    ap.add_argument(
        "--default-num-gpus",
        type=int,
        default=8,
        help="GPU count when filename is brute_force_lam*_tau*.pkl without Gg (default: 8)",
    )
    args = ap.parse_args()
    plot_all_brute_force_results(
        results_dir=args.results_dir,
        out_dir=args.plots_dir,
        default_num_gpus=args.default_num_gpus,
    )


if __name__ == "__main__":
    main()
