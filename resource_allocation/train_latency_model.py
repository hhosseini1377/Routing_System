#!/usr/bin/env python3
"""
Train the LatencyModel (PyTorch NN) using performance_data JSON datasets.

Uses k-fold cross validation (default 5-fold: 80%% train, 20%% val per fold)
with early stopping (patience). Saves the best model across folds.

Usage:
    python -m resource_allocation.train_latency_model --input performance_data_mistral.json
    python -m resource_allocation.train_latency_model -i file1.json -m p99_latency_ms --n-folds 5 --patience 50
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from resource_allocation.latency_model import LatencyModel


def load_performance_data_files(input_files: list[str]) -> list:
    """Load and merge performance data from the given JSON files."""
    all_data = []
    for path in input_files:
        with open(path, "r") as f:
            data = json.load(f)
        all_data.extend(data)
    return all_data


METRIC_CHOICES = ("avg_latency_ms", "p99_latency_ms", "p50_latency_ms", "ttft", "p99_ttft", "p50_ttft", "p95_ttft", "tpot", "p99_tpot", "p95_tpot")


def get_metric_value(result: dict, metric: str) -> float | None:
    """Extract target metric value (ms) from result dict. Returns None if missing."""
    if result.get("failed"):
        return None
    perf = result.get("performance", {})
    if metric == "avg_latency_ms":
        return perf.get("avg_latency_ms")
    if metric == "p99_latency_ms":
        return perf.get("p99_latency_ms")
    if metric == "p50_latency_ms":
        return perf.get("p50_latency_ms")
    if metric == "ttft":
        ttfts = result.get("ttfts_ms")
        if ttfts and len(ttfts) > 0:
            return float(np.mean(ttfts))
        return perf.get("avg_ttft_ms") or (result.get("metrics_after", {}) or {}).get("avg_ttft_ms")
    if metric == "p99_ttft":
        ttfts = result.get("ttfts_ms")
        if ttfts and len(ttfts) > 0:
            return float(np.percentile(ttfts, 99))
        return perf.get("p99_ttft_ms") or (result.get("metrics_after", {}) or {}).get("p99_ttft_ms")
    if metric == "p95_ttft":
        ttfts = result.get("ttfts_ms")
        if ttfts and len(ttfts) > 0:
            return float(np.percentile(ttfts, 95))
        return perf.get("p95_ttft_ms") or (result.get("metrics_after", {}) or {}).get("p95_ttft_ms")      
    if metric == "p99_tpot":
        tpot_list = result.get("tpot_ms_list")
        if tpot_list and len(tpot_list) > 0:
            return float(np.percentile(tpot_list, 99))
        return perf.get("p99_tpot_ms") or (result.get("metrics_after", {}) or {}).get("p99_tpot_ms")
    if metric == "p95_tpot":
        tpot_list = result.get("tpot_ms_list")
        if tpot_list and len(tpot_list) > 0:
            return float(np.percentile(tpot_list, 95))
        return perf.get("p95_tpot_ms") or (result.get("metrics_after", {}) or {}).get("p95_tpot_ms")
    if metric == "tpot":
        tpot_list = result.get("tpot_ms_list")
        if tpot_list and len(tpot_list) > 0:
            return float(np.mean(tpot_list))
        return perf.get("avg_tpot_ms") or (result.get("metrics_after", {}) or {}).get("avg_tpot_ms")
    raise ValueError(f"Unknown metric: {metric}. Choose from {METRIC_CHOICES}")


def extract_xy(
    data: list,
    metric: str = "avg_latency_ms",
    log_load_rps: bool = True,
    min_throughput_load_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and target y (ms).
    Features: tensor_parallel_size, load_rps (or log), thread_percentage.
    Skips failed experiments.
    If min_throughput_load_ratio is set, only keeps samples where throughput_rps/load_rps > r.
    """
    X_list = []
    y_list = []
    for item in data:
        setup = item["setup"]
        result = item["result"]

        value_ms = get_metric_value(result, metric)
        if value_ms is None:
            continue

        if min_throughput_load_ratio is not None:
            perf = result.get("performance", {})
            throughput_rps = perf.get("throughput_rps")
            load_rps = float(setup["load_rps"])
            if throughput_rps is None or load_rps <= 0:
                continue
            if float(throughput_rps) / load_rps <= min_throughput_load_ratio:
                continue

        tp = int(setup["tensor_parallel_size"])
        load_rps = float(setup["load_rps"])
        thread_pct = float(setup["thread_percentage"])

        load_feat = np.log1p(load_rps) if log_load_rps else load_rps
        X_list.append([tp, load_feat, thread_pct])
        y_list.append(float(value_ms))

    if not X_list:
        raise ValueError(
            f"No valid (setup, {metric}) pairs found. "
            "All results may be failed or missing the chosen metric."
            + (f" (min_throughput_load_ratio={min_throughput_load_ratio})" if min_throughput_load_ratio is not None else "")
        )

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Train LatencyModel on performance_data*.json (tensor_parallel_size, load_rps, thread_percentage -> avg_latency_ms)"
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
        "--metric",
        "-m",
        type=str,
        choices=METRIC_CHOICES,
        default="avg_latency_ms",
        help="Target metric: avg_latency_ms, p99/p50_latency_ms, ttft, p99/p50/p95_ttft, tpot, p99/p95_tpot (default: avg_latency_ms)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="resource_allocation/latency_model.pth",
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
        "--log-target",
        action="store_true",
        default=True,
        help="Train on log(latency) for better scaling (default: True)",
    )
    parser.add_argument(
        "--no-log-target",
        action="store_false",
        dest="log_target",
        help="Disable log-target (train on raw latency)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=("mse", "huber"),
        default="mse",
        help="Loss function: mse or huber (default: mse)",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="Delta for Huber loss (default: 1.0). Larger = more MSE-like.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds for k-fold cross validation (80%% train, 20%% val per fold when k=5)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience: stop if val_mae does not improve for this many epochs",
    )
    parser.add_argument(
        "--min-throughput-load-ratio",
        "-r",
        type=float,
        default=None,
        metavar="R",
        help="Only use setups where throughput_rps/load_rps > R (stable regime). Default: no filter.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    data = load_performance_data_files(args.input)
    print(f"Loaded {len(data)} experiments from {len(args.input)} file(s)")

    X, y_ms = extract_xy(data, metric=args.metric, min_throughput_load_ratio=args.min_throughput_load_ratio)
    ratio_filter = f", throughput/load > {args.min_throughput_load_ratio}" if args.min_throughput_load_ratio is not None else ""
    print(f"Using {len(X)} successful experiments{ratio_filter}, target: {args.metric}, loss: {args.loss}")
    print(f"K-fold: {args.n_folds} folds (80%% train / 20%% val), patience: {args.patience}")

    # Train on log(latency) for better scaling across wide latency range
    log_target = args.log_target
    y_target = np.log(y_ms).astype(np.float32) if log_target else y_ms.astype(np.float32)

    def mae_ms(pred_tensor, y_ms_true, log_tgt=log_target):
        """MAE in ms: if log_target, pred is log(ms), else raw."""
        pred_np = pred_tensor.detach().numpy()
        if log_tgt:
            pred_ms = np.exp(pred_np)
        else:
            pred_ms = pred_np
        return float(np.abs(pred_ms - y_ms_true).mean())

    if args.loss == "huber":
        criterion = torch.nn.HuberLoss(delta=args.huber_delta)
    else:
        criterion = torch.nn.MSELoss()

    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_val_maes = []
    best_overall_val_mae = float("inf")
    best_state = None
    best_scaler_mean = None
    best_scaler_scale = None

    for fold, (idx_train, idx_val) in enumerate(kfold.split(X)):
        print(f"\n--- Fold {fold + 1}/{args.n_folds} ---")
        X_train_f, X_val_f = X[idx_train], X[idx_val]
        y_train_f = y_target[idx_train]
        y_val_ms_f = y_ms[idx_val]

        # Fit scaler on train only, transform both
        scaler_x = StandardScaler()
        X_train_scaled = scaler_x.fit_transform(X_train_f)
        X_val_scaled = scaler_x.transform(X_val_f)

        X_train_t = torch.from_numpy(X_train_scaled)
        y_train_t = torch.from_numpy(y_train_f).unsqueeze(1)
        X_val_t = torch.from_numpy(X_val_scaled)

        model = LatencyModel(in_features=3, hidden=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_fold_val_mae = float("inf")
        best_fold_state = None
        epochs_no_improve = 0

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_t)
            loss = criterion(pred, y_train_t)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_val = model(X_val_t)
                val_mae = mae_ms(pred_val, y_val_ms_f)

            if val_mae < best_fold_val_mae:
                best_fold_val_mae = val_mae
                best_fold_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                train_mae = mae_ms(pred, y_ms[idx_train])
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
    print(f"  Validation target (actual values): mean = {np.mean(y_ms):.1f} ms, std = {np.std(y_ms):.1f} ms")

    # Save best model across folds
    model = LatencyModel(in_features=3, hidden=128)
    model.load_state_dict(best_state)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_x_mean": best_scaler_mean,
            "scaler_x_scale": best_scaler_scale,
            "log_target": log_target,
            "log_load_rps": True,
            "metric": args.metric,
            "loss": args.loss,
        },
        out_path,
    )
    print(f"Saved best checkpoint (val_mae={best_overall_val_mae:.1f}ms) to {out_path}")


if __name__ == "__main__":
    main()
