#!/usr/bin/env python3
"""
Train the TwoStageTTFTModel using performance_data JSON datasets.

Uses k-fold cross validation (default 5-fold: 80% train, 20% val per fold)
with early stopping (patience). Target: TTFT (log1p). Requires throughput_rps
for gate supervision (overload = throughput/load < feasibility_threshold).

Usage:
    python -m resource_allocation.train_stage_ttft_model --input performance_data_mistral.json
    python -m resource_allocation.train_stage_ttft_model -i file1.json --n-folds 5 --patience 50
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from resource_allocation.two_stage_ttft_loss import compute_losses
from resource_allocation.two_stage_ttft_model import TwoStageTTFTModel


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract X, y_ms, and throughput_load_ratio.
    metric: "ttft" (mean), "p95_ttft" (95th pctl of ttfts_ms), or "p95_tpot" (95th pctl of tpot_ms_list).
    Features: tensor_parallel_size, load_rps (or log), thread_percentage.
    Skips failed experiments and rows missing required data or throughput_rps.
    If min_throughput_load_ratio is set, only keeps samples where throughput/load > r.
    """
    X_list = []
    y_list = []
    ratio_list = []
    for item in data:
        setup = item["setup"]
        result = item["result"]

        if result.get("failed"):
            continue

        perf = result.get("performance", {})
        throughput_rps = perf.get("throughput_rps")
        load_rps = float(setup["load_rps"])
        if throughput_rps is None or load_rps <= 0:
            continue

        ratio = float(throughput_rps) / load_rps
        if min_throughput_load_ratio is not None and ratio <= min_throughput_load_ratio:
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
        load_feat = np.log1p(load_rps) if log_load_rps else load_rps

        X_list.append([tp, load_feat, thread_pct])
        y_list.append(y_ms)
        ratio_list.append(ratio)

    if not X_list:
        raise ValueError(
            f"No valid (setup, {metric}, throughput_rps) tuples found. "
            "All results may be failed or missing required data / throughput_rps."
            + (f" (min_throughput_load_ratio={min_throughput_load_ratio})" if min_throughput_load_ratio is not None else "")
        )

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(ratio_list, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train TwoStageTTFTModel on performance_data (tp, load_rps, thread_pct) -> TTFT"
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
        default="resource_allocation/two_stage_ttft_model.pth",
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
        "--p-over-threshold",
        type=float,
        default=0.5,
        help="Threshold on p_over: penalty when p_over < this and overloaded (default: 0.5)",
    )
    parser.add_argument(
        "--large-penalty",
        type=float,
        default=100.0,
        help="Large penalty when gate wrongly predicts stable (default: 100.0)",
    )
    parser.add_argument(
        "--feasibility-threshold",
        type=float,
        default=0.98,
        help="Throughput/load ratio threshold: overload if ratio < this (default: 0.98)",
    )
    parser.add_argument(
        "--gate-hidden",
        type=int,
        default=16,
        help="Gate MLP hidden size (default: 16)",
    )
    parser.add_argument(
        "--expert-hidden",
        type=int,
        default=32,
        help="Expert MLP hidden size (default: 32)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate (default: 0.0)",
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
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        choices=METRIC_CHOICES,
        default="ttft",
        help="Target metric: ttft, p95_ttft, or p95_tpot (default: ttft)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    data = load_performance_data_files(args.input)
    print(f"Loaded {len(data)} experiments from {len(args.input)} file(s)")

    X, y_ttft_ms, ratio = extract_xy_ttft(data, metric=args.metric, min_throughput_load_ratio=args.min_throughput_load_ratio)
    y_log_ttft = np.log1p(y_ttft_ms).astype(np.float32)
    ratio_filter = f", throughput/load > {args.min_throughput_load_ratio}" if args.min_throughput_load_ratio is not None else ""
    print(f"Using {len(X)} successful experiments{ratio_filter}, target: {args.metric}, n_folds={args.n_folds}, patience={args.patience}")

    def mae_ttft_ms(pred_log_ttft_tensor, y_ttft_ms_true):
        """MAE in ms: pred_log_ttft is log1p(ttft), convert back."""
        pred_log = pred_log_ttft_tensor.detach().numpy()
        pred_ms = np.expm1(pred_log)
        return float(np.abs(pred_ms - y_ttft_ms_true).mean())

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
        ratio_train = ratio[idx_train]
        ratio_val = ratio[idx_val]

        # Fit scaler on train only
        scaler_x = StandardScaler()
        X_train_scaled = scaler_x.fit_transform(X_train_f)
        X_val_scaled = scaler_x.transform(X_val_f)

        X_train_t = torch.from_numpy(X_train_scaled)
        y_train_log_t = torch.from_numpy(y_train_log).unsqueeze(1)
        ratio_train_t = torch.from_numpy(ratio_train).unsqueeze(1)
        X_val_t = torch.from_numpy(X_val_scaled)
        ratio_val_t = torch.from_numpy(ratio_val).unsqueeze(1)

        model = TwoStageTTFTModel(
            in_features=3,
            gate_hidden=args.gate_hidden,
            expert_hidden=args.expert_hidden,
            dropout=args.dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        best_fold_val_mae = float("inf")
        best_fold_state = None
        epochs_no_improve = 0

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            out = model(X_train_t)
            losses = compute_losses(
                model_out=out,
                y_log_ttft=y_train_log_t,
                throughput_load_ratio=ratio_train_t,
                feasibility_threshold=args.feasibility_threshold,
                p_over_threshold=args.p_over_threshold,
                large_penalty=args.large_penalty,
            )
            losses["total_loss"].backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = model(X_val_t)
                val_mae = mae_ttft_ms(out_val["pred_log_ttft"], y_val_ttft_ms)

            if val_mae < best_fold_val_mae:
                best_fold_val_mae = val_mae
                best_fold_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                train_mae = mae_ttft_ms(out["pred_log_ttft"], y_ttft_ms[idx_train])
                print(
                    f"  Epoch {epoch + 1}/{args.epochs}  "
                    f"total={losses['total_loss'].item():.4f}  "
                    f"reg={losses['reg_loss'].item():.4f}  "
                    f"penalty={losses['penalty_loss'].item():.4f}  "
                    f"train_mae={train_mae:.1f}ms  val_mae={val_mae:.1f}ms"
                )

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

    # Save best model across folds
    model = TwoStageTTFTModel(
        in_features=3,
        gate_hidden=args.gate_hidden,
        expert_hidden=args.expert_hidden,
        dropout=args.dropout,
    )
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
            "feasibility_threshold": args.feasibility_threshold,
            "p_over_threshold": args.p_over_threshold,
        },
        out_path,
    )
    print(f"Saved best checkpoint (val_mae={best_overall_val_mae:.1f}ms) to {out_path}")


if __name__ == "__main__":
    main()
