#!/usr/bin/env python3
"""
Compute the output quality score distribution for prompts using the router model.

Loads prompts from a dataset (e.g. datasets/lmsys_chat1m_prompts_100k_cleaned.pkl),
runs them through the router model in batches, and reports the distribution of
quality scores (min, max, mean, std, percentiles, optional histogram).

Usage:
    python -m profiler.quality_distribution \
        --dataset datasets/lmsys_chat1m_prompts_100k_cleaned.pkl \
        --model-path router_model/model_checkpoints/model_deberta_20260101-163459.pth \
        --output profiler/quality_distribution.json \
        --batch-size 32 \
        --max-prompts 10000
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add project root for imports
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from router_model.regression_models import TruncatedModel, load_tokenizer
from router_model.config import RouterModelConfig
from config import RouterConfig


def format_cnn_dailymail_prompt_qwen(
    article_text: str,
    use_chatml: bool = True,
    use_no_think: bool = True
) -> str:
    """
    Format a CNN/DailyMail article into a Qwen-style prompt for summarization.

    Args:
        article_text: The article text to summarize
        use_chatml: Whether to use ChatML format (default: True for Qwen models)
        use_no_think: Whether to use /no_think command to disable thinking mode (default: True)

    Returns:
        Formatted prompt string
    """
    user_content = (
        "Summarize the following article in a few sentences, focusing on the main points.\n\n"
        f"Article:\n{article_text}\n\n"
        "Summary:\n"
    )

    if use_chatml:
        if use_no_think:
            system_instruction = (
                "/no_think You are a helpful assistant that provides concise and accurate summaries "
                "of news articles. Output only the summary without any reasoning or explanation."
            )
        else:
            system_instruction = (
                "You are a helpful assistant that provides concise and accurate summaries "
                "of news articles. Output only the summary without any reasoning or explanation."
            )

        prompt = (
            "<|im_start|>system\n"
            f"{system_instruction}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_content}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        prompt = user_content

    return prompt


def load_prompts_from_pkl(path: str, max_prompts: int = None):
    """
    Load prompts from a pickle file.
    Supports: list of str, list of dict with 'prompt'/'text'/'content', or pandas DataFrame.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Avoid "truth value of DataFrame is ambiguous" - use explicit checks
    if data is None:
        return []
    if hasattr(data, "empty") and data.empty:
        return []
    if isinstance(data, (list, tuple)) and len(data) == 0:
        return []

    # Handle pandas DataFrame
    if hasattr(data, "columns") and hasattr(data, "iloc"):
        for col in ("prompt", "text", "content", "article", "prompts"):
            if col in data.columns:
                prompts = data[col].astype(str).tolist()
                break
        else:
            raise ValueError(
                f"DataFrame has no 'prompt', 'text', 'content', or 'article' column. "
                f"Columns: {list(data.columns)}"
            )
    else:
        first = data[0]
        if isinstance(first, str):
            prompts = list(data)
        elif isinstance(first, dict):
            prompts = []
            for item in data:
                p = item.get("prompt") or item.get("text") or item.get("content")
                if p is not None:
                    prompts.append(p if isinstance(p, str) else str(p))
        else:
            raise ValueError(f"Unsupported pkl format: list of {type(first)}")
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    return prompts


def get_quality_scores(model, tokenizer, prompts: list[str], batch_size: int, device: torch.device) -> np.ndarray:
    """Run prompts through the router model in batches; return 1D array of scores."""
    all_scores = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i : i + batch_size]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        if outputs.dim() == 1:
            scores = outputs.cpu().numpy()
        else:
            scores = outputs.squeeze(-1).cpu().numpy()
        scores = np.atleast_1d(scores)
        all_scores.append(scores)
    return np.concatenate(all_scores, axis=0)


def compute_distribution(scores: np.ndarray, num_bins: int = 20) -> dict:
    """Compute distribution stats and optional histogram."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return {"count": 0, "error": "No finite scores"}

    percentiles = [1, 5, 25, 50, 75, 95, 99]
    hist, bin_edges = np.histogram(scores, bins=num_bins)

    return {
        "count": int(len(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "percentiles": {f"p{p}": float(np.percentile(scores, p)) for p in percentiles},
        "histogram": {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute router model quality score distribution over a prompt dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/lmsys_chat1m_prompts_100k_cleaned.pkl",
        help="Path to prompts pickle (list of str or list of dict with 'prompt'/'text')",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to router checkpoint (.pth). Default: ROUTER_MODEL_PATH or router_model/model_checkpoints/...",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="profiler/quality_distribution.json",
        help="Path to save distribution JSON",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Max number of prompts to score (default: all)",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=20,
        help="Number of histogram bins",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model (cuda or cpu)",
    )
    parser.add_argument(
        "--save-scores",
        type=str,
        default=None,
        help="Optional path to save (index, score) or full (prompt, score) as JSON for analysis",
    )
    parser.add_argument(
        "--format-cnn-dailymail",
        action="store_true",
        help="Format each loaded prompt as CNN/DailyMail Qwen summarization prompt (ChatML + /no_think)",
    )
    parser.add_argument(
        "--no-chatml",
        action="store_true",
        help="When using --format-cnn-dailymail, use plain text instead of ChatML (default: ChatML)",
    )
    parser.add_argument(
        "--no-no-think",
        action="store_true",
        help="When using --format-cnn-dailymail, do not add /no_think to system instruction",
    )
    args = parser.parse_args()

    model_path = args.model_path or __import__("os").environ.get(
        "ROUTER_MODEL_PATH",
        "router_model/model_checkpoints/model_deberta_20260101-163459.pth",
    )
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = _ROOT / model_path
    if not model_path.exists():
        print(f"ERROR: Model checkpoint not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = _ROOT / dataset_path
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading prompts...")
    prompts = load_prompts_from_pkl(str(dataset_path), max_prompts=args.max_prompts)
    print(f"  Loaded {len(prompts)} prompts")

    if args.format_cnn_dailymail:
        prompts = [
            format_cnn_dailymail_prompt_qwen(
                p,
                use_chatml=not args.no_chatml,
                use_no_think=not args.no_no_think,
            )
            for p in prompts
        ]
        print("  Applied CNN/DailyMail Qwen format to all prompts")

    print("Loading router model...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    router_config = RouterModelConfig()
    model = TruncatedModel.load_model_from_checkpoint(
        model_path=str(model_path),
        model_name="deberta",
        pooling_strategy="cls",
        num_outputs=1,
        num_classes=2,
        router_model_config=router_config,
        device=str(device),
    )
    tokenizer = load_tokenizer(model_name=RouterConfig().model_name)
    print(f"  Model on {device}")

    print("Computing quality scores...")
    scores = get_quality_scores(model, tokenizer, prompts, args.batch_size, device)
    print(f"  Got {len(scores)} scores")

    # Also compute sigmoid-transformed scores (e.g. for probability-like interpretation)
    sigmoid_scores = 1.0 / (1.0 + np.exp(-scores))

    dist = compute_distribution(scores, num_bins=args.num_bins)
    dist["sigmoid_distribution"] = compute_distribution(sigmoid_scores, num_bins=args.num_bins)
    dist["dataset"] = str(dataset_path)
    dist["model_path"] = str(model_path)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dist, f, indent=2)
    print(f"Saved distribution to {out_path}")

    print("\nQuality score distribution:")
    print(f"  count:   {dist['count']}")
    print(f"  min:     {dist['min']:.4f}")
    print(f"  max:     {dist['max']:.4f}")
    print(f"  mean:    {dist['mean']:.4f}")
    print(f"  std:     {dist['std']:.4f}")
    print("  percentiles:")
    for k, v in dist["percentiles"].items():
        print(f"    {k}: {v:.4f}")

    if args.save_scores:
        save_path = Path(args.save_scores)
        if not save_path.is_absolute():
            save_path = _ROOT / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Save compact list of scores, sigmoid(scores), plus optional prompts (can be large)
        out_data = {
            "scores": scores.tolist(),
            "sigmoid_scores": sigmoid_scores.tolist(),
        }
        if len(prompts) == len(scores) and len(prompts) <= 10_000:
            out_data["prompts"] = prompts
        with open(save_path, "w") as f:
            json.dump(out_data, f, indent=0)
        print(f"Saved scores to {save_path}")


if __name__ == "__main__":
    main()
