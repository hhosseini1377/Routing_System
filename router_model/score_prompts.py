#!/usr/bin/env python3
"""
Run the multi-head router model on prompts and save scores for each prompt.

Loads prompts from a pickle file (list of strings), runs them through the trained
multi-head regression model, and saves the scores to a pickle or parquet file.

Usage:
    python -m router_model.score_prompts \
        --model router_model/model_checkpoints/multi_head_regression_best.pth \
        --prompts datasets/routerbench_0shot_prompts.pkl \
        --output datasets/routerbench_0shot_scores.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from .train_multi_head_regression import load_multi_head_model


class PromptOnlyDataset(Dataset):
    """Dataset of prompts only (no labels) for inference."""

    def __init__(self, prompts: list, tokenizer, max_length: int = 512):
        self.prompts = [str(p) for p in prompts]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        text = self.prompts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def run_inference(model, loader, device):
    """Run model on all batches, return predictions."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pred = model(input_ids=input_ids, attention_mask=attention_mask)
            all_preds.append(pred.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def load_prompts(path: str) -> list:
    """Load prompts from pickle file (expects list of strings)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of prompts, got {type(data)}")
    return [str(p) for p in data]


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-head router on prompts and save scores"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="router_model/model_checkpoints/multi_head_regression_best.pth",
        help="Path to checkpoint (.pth)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="datasets/routerbench_0shot_prompts.pkl",
        help="Path to prompts pickle file (list of strings)",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Max number of prompts to process (default: all). Useful for quick testing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/routerbench_0shot_scores.pkl",
        help="Output path for scores (pickle or parquet)",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--format",
        type=str,
        choices=["pkl", "parquet"],
        default="pkl",
        help="Output format: pkl (pickle) or parquet",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model, score_cols = load_multi_head_model(args.model, device=str(device))
    print(f"  Score columns: {score_cols}")

    print("Loading prompts...")
    prompts = load_prompts(args.prompts)
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
        print(f"  Using first {len(prompts)} prompts (--max-prompts={args.max_prompts})")
    else:
        print(f"  Loaded {len(prompts)} prompts")

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-large",
        truncation_side="left",
    )
    dataset = PromptOnlyDataset(prompts, tokenizer, args.max_length)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print("Running inference...")
    scores = run_inference(model, loader, device)
    print(f"  Shape: {scores.shape}")

    # Build output: list of dicts with prompt and per-head scores
    results = []
    for i, prompt in enumerate(prompts):
        row = {"prompt": prompt, "idx": i}
        for j, col in enumerate(score_cols):
            row[col] = float(scores[i, j])
        results.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "parquet" or out_path.suffix.lower() in (".parquet", ".pq"):
        df = pd.DataFrame(results)
        df.to_parquet(out_path, index=False)
        print(f"Saved {len(results)} rows to {out_path} (parquet)")
    else:
        save_data = {
            "prompts": prompts,
            "scores": scores,
            "score_columns": score_cols,
            "results": results,
        }
        with open(out_path, "wb") as f:
            pickle.dump(save_data, f)
        print(f"Saved {len(results)} prompts with scores to {out_path} (pickle)")

    print("Done.")


if __name__ == "__main__":
    main()
