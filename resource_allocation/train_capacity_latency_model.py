#!/usr/bin/env python3
"""
Train CapacityInspiredLatencyModel: ℓ(θ, λ) = a(θ) + b(θ)/(c(θ)−λ).

Uses raw load_rps (not log) so capacity c and load λ share units. Same data format
and k-fold CV as train_latency_model.py. Saves scaler; at inference, ∂ℓ/∂λ in RPS
is model.d_latency_d_load(x) * scaler.scale_[2] (scale for the load_rps feature).
"""

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from resource_allocation.capacity_inspired_latency_model import CapacityInspiredLatencyModel
from resource_allocation.train_latency_model import (
    METRIC_CHOICES,
    extract_xy,
    load_performance_data_files,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train CapacityInspiredLatencyModel (setup + load -> latency); use raw load_rps."
    )
    parser.add_argument("--input", "-i", type=str, nargs="+", required=True, help="Performance JSON(s)")
    parser.add_argument("--metric", "-m", type=str, choices=METRIC_CHOICES, default="p95_ttft")
    parser.add_argument("--output", type=str, default="resource_allocation/capacity_latency_model.pth")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--capacity-init", type=float, default=50.0, help="Initial c(θ) so c > load at start")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-throughput-load-ratio", "-r", type=float, default=0.99, metavar="R")
    parser.add_argument("--plot", type=str, default=None, metavar="PATH", help="Save y_true vs y_pred validation plot to PATH")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_performance_data_files(args.input)
    print(f"Loaded {len(data)} experiments from {len(args.input)} file(s)")

    # Raw load_rps for capacity model (no log)
    X, y_ms = extract_xy(
        data,
        metric=args.metric,
        log_load_rps=False,
        min_throughput_load_ratio=args.min_throughput_load_ratio,
    )
    print(f"Using {len(X)} experiments (throughput/load > {args.min_throughput_load_ratio}), target: {args.metric}")

    # Train on log(latency)
    y_target = np.log(y_ms).astype(np.float32)

    def mae_ms(pred_log, y_ms_true):
        pred_ms = np.exp(pred_log.detach().numpy())
        return float(np.abs(pred_ms - y_ms_true).mean())

    criterion = torch.nn.MSELoss()
    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    best_val_mae = float("inf")
    best_state = None
    best_scaler_mean = None
    best_scaler_scale = None
    best_y_val_ms = None
    best_X_val_s = None

    for fold, (idx_train, idx_val) in enumerate(kfold.split(X)):
        print(f"\n--- Fold {fold + 1}/{args.n_folds} ---")
        X_tr, X_val = X[idx_train], X[idx_val]
        y_tr = y_target[idx_train]
        y_val_ms = y_ms[idx_val]

        # Scale only setup (tp, thread_pct); leave load_rps raw so c(θ) and λ share units in a + b/(c−λ)
        scaler = StandardScaler()
        X_tr_s = np.column_stack([
            scaler.fit_transform(X_tr[:, :2]),
            X_tr[:, 2],
        ]).astype(np.float32)
        X_val_s = np.column_stack([
            scaler.transform(X_val[:, :2]),
            X_val[:, 2],
        ]).astype(np.float32)

        X_tr_t = torch.from_numpy(X_tr_s)
        y_tr_t = torch.from_numpy(y_tr).unsqueeze(1)
        X_val_t = torch.from_numpy(X_val_s.astype(np.float32))

        model = CapacityInspiredLatencyModel(setup_dim=2, hidden=args.hidden, capacity_init=args.capacity_init)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_fold_mae = float("inf")
        best_fold_state = None
        no_improve = 0

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_tr_t)
            loss = criterion(pred, y_tr_t)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_mae = mae_ms(model(X_val_t), y_val_ms)

            if val_mae < best_fold_mae:
                best_fold_mae = val_mae
                best_fold_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 50 == 0 or epoch == 0:
                train_mae = mae_ms(model(X_tr_t), y_ms[idx_train])
                print(f"  Epoch {epoch+1}/{args.epochs}  loss={loss.item():.4f}  train_mae={train_mae:.1f}ms  val_mae={val_mae:.1f}ms")

            if no_improve >= args.patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

        if best_fold_mae < best_val_mae:
            best_val_mae = best_fold_mae
            best_state = best_fold_state
            # Scaler was fit on setup only (first 2 cols); load is unscaled
            best_scaler_mean = scaler.mean_.tolist()
            best_scaler_scale = scaler.scale_.tolist()
            best_y_val_ms = y_val_ms.copy()
            best_X_val_s = X_val_s.copy()
            print(f"  New best (val_mae={best_fold_mae:.1f}ms)")

    mean_target_ms = float(np.mean(y_ms))
    print(f"\nBest validation MAE: {best_val_mae:.1f} ms")
    print(f"Target {args.metric}: mean = {mean_target_ms:.1f} ms  →  MAE/mean ≈ {100 * best_val_mae / mean_target_ms:.0f}% (lower is better)")

    model = CapacityInspiredLatencyModel(setup_dim=2, hidden=args.hidden, capacity_init=args.capacity_init)
    model.load_state_dict(best_state)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": best_scaler_mean,
            "scaler_scale": best_scaler_scale,
            "metric": args.metric,
            "log_target": True,
            "log_load_rps": False,
        },
        out_path,
    )
    print(f"Saved to {out_path}")
    print("Note: Only setup (cols 0,1) is scaled; load_rps (col 2) is raw. ∂ℓ/∂(load_rps) = model.d_latency_d_load(x).")

    if best_y_val_ms is not None:
        model.eval()
        with torch.no_grad():
            X_val_t = torch.from_numpy(best_X_val_s.astype(np.float32))
            y_pred_ms = model.forward_latency_ms(X_val_t).squeeze(1).numpy()
        y_true_ms = best_y_val_ms

        print("\nValidation set (best fold):")
        print(f"  Actual ({args.metric} ms):  mean = {np.mean(y_true_ms):.2f},  std = {np.std(y_true_ms):.2f}")
        print(f"  Predicted (ms):           mean = {np.mean(y_pred_ms):.2f},  std = {np.std(y_pred_ms):.2f}")

    if args.plot and best_y_val_ms is not None:
        fig, ax = plt.subplots()
        ax.scatter(y_true_ms, y_pred_ms, alpha=0.5, s=12)
        lims = [min(y_true_ms.min(), y_pred_ms.min()), max(y_true_ms.max(), y_pred_ms.max())]
        ax.plot(lims, lims, "k--", lw=1, label="y_pred = y_true")
        ax.set_xlabel("y_true (ms)")
        ax.set_ylabel("y_pred (ms)")
        ax.set_title(f"Validation: {args.metric} (MAE = {best_val_mae:.1f} ms)")
        ax.legend()
        ax.set_aspect("equal")
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Validation plot saved to {plot_path}")


if __name__ == "__main__":
    main()
