#!/usr/bin/env python3
"""
Check GPU memory requirements for a model at different tensor-parallel (TP) levels.

For each TP level, tries loading the model with GPU memory utilization starting
at 0.9 and decreasing by 0.1 until the model loads successfully. Uses the same
engine config as model_server.py (AsyncEngineArgs, AsyncLLMEngine).

Usage:
    python -m profiler.check_model_memory --model Qwen/Qwen3-1.8B --tp 1 2 4
    python -m profiler.check_model_memory --model mistralai/Mistral-7B-v0.1 --tp 1 2 4 8 --util-min 0.2

Internal (subprocess) mode:
    python -m profiler.check_model_memory --_try-once --model X --tp T --gpu-memory-util U
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _try_load_once(model: str, tp: int, gpu_memory_util: float, **kwargs) -> bool:
    print('salam')
    """Try to load the model once. Returns True on success, False on OOM or other init failure."""
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs

    engine_kwargs = {
        "model": model,
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": 1,
        "max_model_len": int(kwargs.get("max_model_len", 4096)),
        "gpu_memory_utilization": gpu_memory_util,
        "dtype": "auto",
    }
    if kwargs.get("max_num_seqs"):
        engine_kwargs["max_num_seqs"] = int(kwargs["max_num_seqs"])
    if kwargs.get("max_num_batched_tokens"):
        engine_kwargs["max_num_batched_tokens"] = int(kwargs["max_num_batched_tokens"])
    print('salam')
    try:
        engine_args = AsyncEngineArgs(**engine_kwargs)
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        # No explicit stop - subprocess exit releases GPU memory; avoid engine.stop() (not on AsyncLLM)
        return True
    except Exception as e:
        err_str = str(e).lower()
        is_oom = "out of memory" in err_str or "cuda" in err_str and "memory" in err_str
        if is_oom:
            print(f"  [OOM] util={gpu_memory_util:.1f}: {e}", file=sys.stderr)
        else:
            print(f"  [ERROR] util={gpu_memory_util:.1f}: {e}", file=sys.stderr)
        return False


def _run_trial_subprocess(model: str, tp: int, gpu_memory_util: float, **kwargs) -> bool:
    """Run a single load trial in a subprocess so GPU memory is released on exit."""
    cmd = [
        sys.executable,
        "-m",
        "profiler.check_model_memory",
        "--_try-once",
        "--model", model,
        "--tp", str(tp),
        "--gpu-memory-util", str(gpu_memory_util),
    ]
    if kwargs.get("max_model_len"):
        cmd.extend(["--max-model-len", str(kwargs["max_model_len"])])
    if kwargs.get("max_num_seqs"):
        cmd.extend(["--max-num-seqs", str(kwargs["max_num_seqs"])])
    if kwargs.get("max_num_batched_tokens"):
        cmd.extend(["--max-num-batched-tokens", str(kwargs["max_num_batched_tokens"])])
    print('residim')
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    if result.returncode != 0 and result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check GPU memory requirements at different TP levels.")
    parser.add_argument("--model", type=str, required=True, help="Model name or path (e.g. Qwen/Qwen3-1.8B).")
    parser.add_argument(
        "--tp",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Tensor parallel sizes to test (default: 1 2 4).",
    )
    parser.add_argument(
        "--util-start",
        type=float,
        default=0.9,
        help="Starting GPU memory utilization (default: 0.9).",
    )
    parser.add_argument(
        "--util-step",
        type=float,
        default=0.1,
        help="Decrement step for GPU memory utilization (default: 0.1).",
    )
    parser.add_argument(
        "--util-min",
        type=float,
        default=0.1,
        help="Minimum utilization to try before giving up (default: 0.1).",
    )
    parser.add_argument("--max-model-len", type=int, default=2048, help="Max model length (default: 2048).")
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path. Default: min_gpu_util_<model>.json in project root (model name sanitized).",
    )

    # Internal: single trial (used by subprocess)
    parser.add_argument("--_try-once", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-memory-util", type=float, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    kwargs = {
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
    }

    if args._try_once:
        if args.gpu_memory_util is None:
            print("--_try-once requires --gpu-memory-util", file=sys.stderr)
            sys.exit(2)
        tp_val = args.tp[0] if isinstance(args.tp, list) else args.tp
        ok = _try_load_once(args.model, int(tp_val), args.gpu_memory_util, **kwargs)
        sys.exit(0 if ok else 1)

    print(f"Model: {args.model}")
    print(f"TP levels: {args.tp}")
    print(f"Util scan: {args.util_start} -> {args.util_min} (step -{args.util_step})")
    print("-" * 50)

    results = {}
    for tp in sorted(args.tp):
        util = args.util_start
        min_working = None
        while util >= args.util_min:
            print(f"TP={tp} util={util:.1f} ... ", end="", flush=True)
            ok = _run_trial_subprocess(args.model, tp, util, **kwargs)
            if ok:
                print("OK")
                min_working = util  # keep updating; lowest success = minimum
            else:
                print("FAIL")
                # Lower utilization => less GPU memory available. If we fail here,
                # we will not succeed at any lower utilization for this TP.
                break
            util = round(util - args.util_step, 1)
        results[tp] = min_working
        if min_working is None:
            print(f"TP={tp}: No working utilization in range [{args.util_min}, {args.util_start}]")

    print("-" * 50)
    print("Results (minimum GPU memory utilization that loads successfully):")
    for tp in sorted(results.keys()):
        v = results[tp]
        print(f"  TP={tp}: {v if v is not None else 'N/A'}")

    # Save to file with model name
    out_path = args.output
    if out_path is None:
        safe_name = re.sub(r"[^\w\-.]", "_", args.model)
        out_path = _PROJECT_ROOT / f"min_gpu_util_{safe_name}.json"
    else:
        out_path = Path(out_path)
    data = {"model": args.model, "min_gpu_util_per_tp": {str(k): v for k, v in sorted(results.items())}}
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
