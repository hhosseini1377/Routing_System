#!/usr/bin/env python3
"""
Load the trained multi-head regression model, run prompts through it, and compute ROC-AUC per head.

ROC-AUC is computed by binarizing targets (above median = positive) and using model predictions
as the ranking score. This measures how well the model ranks "high vs low" scoring prompts.

Usage:
    python -m router_model.evaluate_multi_head_roc \
        --model router_model/model_checkpoints/multi_head_regression_best.pth \
        --data your_data.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from .train_multi_head_regression import (
    DEFAULT_SCORE_COLUMNS,
    DeBERTaMultiHeadRegression,
    MultiHeadRegressionDataset,
    load_dataset,
    load_multi_head_model,
)


def run_inference(model, loader, device):
    """Run model on all batches, return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            pred = model(input_ids=input_ids, attention_mask=attention_mask)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels.numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return preds, labels


def compute_roc_auc_per_head(preds: np.ndarray, labels: np.ndarray, score_cols: list):
    """
    Binarize labels (above median = 1) and compute ROC-AUC for each head.
    Handles edge cases (constant labels, NaN).
    """
    results = {}
    for i, col in enumerate(score_cols):
        y_true = labels[:, i]
        y_pred = preds[:, i]

        # Binarize: above median = positive class
        median = np.median(y_true)
        y_binary = (y_true >= median).astype(np.int32)

        # Skip if all same class (ROC-AUC undefined)
        if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
            results[col] = float("nan")
            continue

        try:
            auc = roc_auc_score(y_binary, y_pred)
            results[col] = float(auc)
        except ValueError:
            results[col] = float("nan")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-head model and compute ROC-AUC per head")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset (CSV/Parquet/Pickle)")
    parser.add_argument("--prompt-col", type=str, default="prompt")
    parser.add_argument("--score-cols", type=str, nargs="+", default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model, score_cols = load_multi_head_model(args.model, device=str(device))
    print(f"  Score columns: {score_cols}")

    print("Loading dataset...")
    df, score_cols = load_dataset(args.data, args.prompt_col, args.score_cols or score_cols)
    print(f"  Samples: {len(df)}")

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-large",
        truncation_side="left",
    )
    dataset = MultiHeadRegressionDataset(df, score_cols, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("Running inference...")
    preds, labels = run_inference(model, loader, device)

    print("Computing ROC-AUC per head (binary: above median = positive)...")
    roc_results = compute_roc_auc_per_head(preds, labels, score_cols)

    print("\n--- ROC-AUC per head ---")
    for col, auc in roc_results.items():
        print(f"  {col}: {auc:.4f}" if not np.isnan(auc) else f"  {col}: N/A (constant labels)")
    print(f"  Mean: {np.nanmean(list(roc_results.values())):.4f}")


if __name__ == "__main__":
    main()
