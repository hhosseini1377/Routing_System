#!/usr/bin/env python3
"""
Train the QueueInspiredLogTTFTModel using performance_data JSON datasets.

Uses k-fold cross validation (default 5-fold: 80% train, 20% val per fold)
with early stopping (patience). Target: log1p(TTFT).
Input: setup [tp, thread_pct], load [load_rps].

Usage:
    python -m resource_allocation.train_queue_inspired_model --input performance_data_mistral.json
    python -m resource_allocation.train_queue_inspired_model -i file1.json --n-folds 5 --patience 50
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from resource_allocation.queue_inspired_model import QueueInspiredLogTTFTModel


def load_performance_data_files(input_files: list[str]) -> list:
    """Load and merge performance data from the given JSON files."""
    all_data = []
    for path in input_files:
        with open(path, "r") as f:
            data = json.load(f)
        all_data.extend(data)
    return all_data


METRIC_CHOICES = ("ttft", "p95_ttft", "p95_tpot")


def extract_xy_ttft(
    data: list,
    metric: str = "ttft",
    log_load_rps: bool = True,
    min_throughput_load_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract X and y_ms.
    metric: "ttft" (mean), "p95_ttft" (95th pctl of ttfts_ms), or "p95_tpot" (95th pctl of tpot_ms_list).
    X: [tp, thread_pct, load_rps] - setup (first 2) + load (last 1) for QueueInspiredLogTTFTModel.
    If min_throughput_load_ratio is set, only keeps samples where throughput_rps/load_rps > r.
    """
    X_list = []
    y_list = []
    for item in data:
        setup = item["setup"]
        result = item["result"]

        if result.get("failed"):
            continue

        if min_throughput_load_ratio is not None:
            perf = result.get("performance", {})
            throughput_rps = perf.get("throughput_rps")
            load_rps = float(setup["load_rps"])
            if throughput_rps is None or load_rps <= 0:
                continue
            if float(throughput_rps) / load_rps <= min_throughput_load_ratio:
                continue

        if metric == "ttft":
            ttfts = result.get("ttfts_ms")
            if not ttfts or len(ttfts) == 0:
                continue
            y_ms = float(np.mean(ttfts))
        elif metric == "p95_ttft":
            ttfts = result.get("ttfts_ms")
            if not ttfts or len(ttfts) == 0:
                continue
            y_ms = float(np.percentile(ttfts, 95))
        elif metric == "p95_tpot":
            tpot_list = result.get("tpot_ms_list")
            if not tpot_list or len(tpot_list) == 0:
                continue
            y_ms = float(np.percentile(tpot_list, 95))
        else:
            raise ValueError(f"Unknown metric: {metric}. Choose from {METRIC_CHOICES}")

        tp = int(setup["tensor_parallel_size"])
        thread_pct = float(setup["thread_percentage"])
        load_rps = float(setup["load_rps"])
        load_feat = np.log1p(load_rps) if log_load_rps else load_rps

        X_list.append([tp, thread_pct, load_feat])
        y_list.append(y_ms)

    if not X_list:
        raise ValueError(
            f"No valid (setup, {metric}) tuples found. "
            "All results may be failed or missing required data."
            + (f" (min_throughput_load_ratio={min_throughput_load_ratio})" if min_throughput_load_ratio is not None else "")
        )

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Train QueueInspiredLogTTFTModel on performance_data (setup, load) -> log1p(TTFT)"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        nargs="+",
        required=True,
        help="Input JSON file(s) with performance data (setup + result)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="resource_allocation/queue_inspired_model.pth",
        help="Output path for trained model checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=16,
        help="Hidden size for QueueInspiredLogTTFTModel (default: 16)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds for k-fold cross validation",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "--min-throughput-load-ratio",
        "-r",
        type=float,
        default=None,
        metavar="R",
        help="Only use setups where throughput_rps/load_rps > R (stable regime). Default: no filter.",
    )
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        choices=METRIC_CHOICES,
        default="ttft",
        help="Target metric: ttft, p95_ttft, or p95_tpot (default: ttft)",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="Delta for Huber loss (default: 1.0)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_performance_data_files(args.input)
    print(f"Loaded {len(data)} experiments from {len(args.input)} file(s)")

    X, y_ttft_ms = extract_xy_ttft(data, metric=args.metric, min_throughput_load_ratio=args.min_throughput_load_ratio)
    y_log_ttft = np.log1p(y_ttft_ms).astype(np.float32)
    ratio_filter = f", throughput/load > {args.min_throughput_load_ratio}" if args.min_throughput_load_ratio is not None else ""
    print(f"Using {len(X)} successful experiments{ratio_filter}, target: {args.metric}, n_folds={args.n_folds}, patience={args.patience}")

    def mae_ttft_ms(pred_log_ttft_tensor, y_ttft_ms_true):
        pred_log = pred_log_ttft_tensor.detach().numpy()
        pred_ms = np.expm1(pred_log)
        mae = float(np.abs(pred_ms - y_ttft_ms_true).mean())
        return mae if np.isfinite(mae) else float("inf")

    model_fn = lambda: QueueInspiredLogTTFTModel(hidden=args.hidden)
    optimizer_fn = lambda m: torch.optim.Adam(m.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=args.huber_delta)

    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_val_maes = []
    best_overall_val_mae = float("inf")
    best_state = None
    best_scaler_mean = None
    best_scaler_scale = None

    for fold, (idx_train, idx_val) in enumerate(kfold.split(X)):
        print(f"\n--- Fold {fold + 1}/{args.n_folds} ---")
        X_train_f, X_val_f = X[idx_train], X[idx_val]
        y_train_log = y_log_ttft[idx_train]
        y_val_ttft_ms = y_ttft_ms[idx_val]

        scaler_x = StandardScaler()
        X_train_scaled = scaler_x.fit_transform(X_train_f)
        X_val_scaled = scaler_x.transform(X_val_f)

        X_train_t = torch.from_numpy(X_train_scaled)
        y_train_t = torch.from_numpy(y_train_log).unsqueeze(1)
        X_val_t = torch.from_numpy(X_val_scaled)

        model = model_fn()
        optimizer = optimizer_fn(model)

        best_fold_val_mae = float("inf")
        best_fold_state = None
        epochs_no_improve = 0

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_t)
            loss = loss_fn(pred, y_train_t)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_val = model(X_val_t)
                val_mae = mae_ttft_ms(pred_val, y_val_ttft_ms)

            if val_mae < best_fold_val_mae or best_fold_state is None:
                best_fold_val_mae = val_mae
                best_fold_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                train_mae = mae_ttft_ms(pred, y_ttft_ms[idx_train])
                print(f"  Epoch {epoch + 1}/{args.epochs}  loss={loss.item():.4f}  train_mae={train_mae:.1f}ms  val_mae={val_mae:.1f}ms")

            if epochs_no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch + 1} (no improvement for {args.patience} epochs)")
                break

        model.load_state_dict(best_fold_state)
        fold_val_maes.append(best_fold_val_mae)

        if best_fold_val_mae < best_overall_val_mae:
            best_overall_val_mae = best_fold_val_mae
            best_state = best_fold_state
            best_scaler_mean = scaler_x.mean_.tolist()
            best_scaler_scale = scaler_x.scale_.tolist()
            print(f"  New best model (val_mae={best_fold_val_mae:.1f}ms)")

    mean_val = np.mean(fold_val_maes)
    std_val = np.std(fold_val_maes)
    print(f"\nK-fold validation results:")
    print(f"  Average val_mae: {mean_val:.1f} ms")
    print(f"  Standard deviation: {std_val:.1f} ms")
    print(f"  Validation target (actual values): mean = {np.mean(y_ttft_ms):.1f} ms, std = {np.std(y_ttft_ms):.1f} ms")

    if best_state is None:
        best_state = best_fold_state
        best_scaler_mean = scaler_x.mean_.tolist()
        best_scaler_scale = scaler_x.scale_.tolist()
        best_overall_val_mae = best_fold_val_mae

    model = model_fn()
    model.load_state_dict(best_state)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_x_mean": best_scaler_mean,
            "scaler_x_scale": best_scaler_scale,
            "log_load_rps": True,
            "target": args.metric,
            "log1p_target": True,
        },
        out_path,
    )
    print(f"Saved best checkpoint (val_mae={best_overall_val_mae:.1f}ms) to {out_path}")


if __name__ == "__main__":
    main()
