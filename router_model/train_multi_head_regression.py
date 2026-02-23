#!/usr/bin/env python3
"""
Train DeBERTa-v3-large with three regression heads to predict model scores from prompts.

Dataset columns: prompt, mistralai/mistral-7b-chat, zero-one-ai/Yi-34B-Chat, WizardLM/WizardLM-13B-V1.2

Supports multi-GPU via DataParallel (single node) or DistributedDataParallel.

Usage:
    # Single GPU
    python -m router_model.train_multi_head_regression --data data.csv --output checkpoints/

    # Multi-GPU (DataParallel)
    python -m router_model.train_multi_head_regression --data data.csv --output checkpoints/ --multi-gpu

    # From DataFrame (CSV, Parquet, or pickle)
    python -m router_model.train_multi_head_regression --data scores.parquet --output checkpoints/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup



# Default score columns (order matters for output heads)
DEFAULT_SCORE_COLUMNS = [
    "mistralai/mistral-7b-chat",
    "zero-one-ai/Yi-34B-Chat",
    "WizardLM/WizardLM-13B-V1.2",
]


def load_dataset(path: str, prompt_col: str = "prompt", score_cols: list = None):
    """Load dataset from CSV, Parquet, or pickle. Returns (df, score_cols)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif path.suffix.lower() in (".pkl", ".pickle"):
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Use .csv, .parquet, or .pkl")

    if prompt_col not in df.columns:
        raise ValueError(f"Column '{prompt_col}' not found. Available: {list(df.columns)}")

    if score_cols is None:
        # Try default columns first; else use all non-prompt columns if exactly 3
        score_cols = [c for c in DEFAULT_SCORE_COLUMNS if c in df.columns]
        if len(score_cols) != 3:
            other_cols = [c for c in df.columns if c != prompt_col]
            if len(other_cols) == 3:
                score_cols = other_cols
    else:
        for c in score_cols:
            if c not in df.columns:
                raise ValueError(f"Score column '{c}' not found. Available: {list(df.columns)}")

    if len(score_cols) != 3:
        raise ValueError(
            f"Expected 3 score columns, got {len(score_cols)}: {score_cols}. "
            f"Use --score-cols to specify."
        )

    return df[[prompt_col] + score_cols].rename(columns={prompt_col: "prompt"}), score_cols


class MultiHeadRegressionDataset(Dataset):
    """Dataset of prompts and 3 model scores."""

    def __init__(self, df: pd.DataFrame, score_cols: list, tokenizer, max_length: int = 512):
        self.prompts = df["prompt"].astype(str).tolist()
        self.labels = df[score_cols].astype(np.float32).values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        text = self.prompts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
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
            "labels": label,
        }


class DeBERTaMultiHeadRegression(nn.Module):
    """DeBERTa-v3-large with 3 regression heads (one per model score)."""

    def __init__(
        self,
        num_heads: int = 3,
        dropout_rate: float = 0.1,
        freeze_layers: int = 0,
        pooling: str = "cls",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.pooling = pooling

        self.transformer = AutoModel.from_pretrained(
            "microsoft/deberta-v3-large",
            torch_dtype=torch.float32,
        )

        if freeze_layers > 0:
            for i, layer in enumerate(self.transformer.encoder.layer):
                if i < freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

        hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_heads)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            pooled = last_hidden[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        pooled = self.dropout(pooled.float())
        out = torch.cat([h(pooled) for h in self.heads], dim=1)
        return out.squeeze(-1) if out.dim() == 3 else out


def train_epoch(model, loader, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    use_amp = scaler is not None

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                pred = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(pred, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        loss_val = loss.item()
        total_loss += loss_val
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        pred = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(pred, labels)
        total_loss += loss.item()
        all_preds.append(pred.cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    mae_per_head = np.abs(preds - labels).mean(axis=0)
    return total_loss / len(loader), mae_per_head


def main():
    parser = argparse.ArgumentParser(description="Train DeBERTa multi-head regression")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV/Parquet/Pickle")
    parser.add_argument("--output", type=str, default="router_model/model_checkpoints", help="Output dir")
    parser.add_argument("--prompt-col", type=str, default="prompt", help="Prompt column name")
    parser.add_argument("--score-cols", type=str, nargs="+", default=None,
                        help="Score columns (default: mistral-7b, Yi-34B, WizardLM-13B)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--freeze-layers", type=int, default=0, help="Freeze first N transformer layers")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--multi-gpu", action="store_true", help="Use DataParallel for multi-GPU")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading dataset...")
    df, score_cols = load_dataset(args.data, args.prompt_col, args.score_cols)
    print(f"  Loaded {len(df)} samples, score columns: {score_cols}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-large",
        truncation_side="left",
    )

    dataset = MultiHeadRegressionDataset(df, score_cols, tokenizer, args.max_length)

    n_val = int(len(dataset) * args.val_frac)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DeBERTaMultiHeadRegression(
        num_heads=3,
        dropout_rate=0.1,
        freeze_layers=args.freeze_layers,
        pooling=args.pooling,
    )

    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        val_loss, mae_per_head = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_MAE={mae_per_head}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            ckpt = {
                "model_state_dict": state,
                "score_cols": score_cols,
                "num_heads": 3,
                "epoch": epoch,
            }
            torch.save(ckpt, out_dir / "multi_head_regression_best.pth")
            print(f"  Saved best checkpoint")

    print(f"\nTraining done. Best checkpoint: {out_dir / 'multi_head_regression_best.pth'}")


def load_multi_head_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained multi-head model for inference."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DeBERTaMultiHeadRegression(
        num_heads=ckpt["num_heads"],
        dropout_rate=0.0,
        freeze_layers=0,
        pooling="cls",
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt.get("score_cols", DEFAULT_SCORE_COLUMNS)


if __name__ == "__main__":
    main()
