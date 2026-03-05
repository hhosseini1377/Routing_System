#!/usr/bin/env python3
"""
Collect routing performance data at a predefined load.

Assumes the router and two model backends are already running (e.g. via scripts/start_services.sh).
Sends requests to the router at a target RPS (open-loop) for a given duration and collects
metrics: TTFTs, latency, throughput, success rate, backend distribution, etc. — same style as
single-model profiling (collect_performance_data.py).

Usage:
  # Start services first (in another terminal):
  #   ./scripts/start_services.sh

  python -m profiler.collect_routing_performance_data \
    --router-url http://127.0.0.1:8000 \
    --load-rps 20 \
    --duration 60 \
    --output routing_performance.json \
    --prompts-path datasets/lmsys_chat1m_prompts_100k_cleaned.pkl
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np


def load_prompts(prompts_path: Optional[str] = None, num_prompts: int = 50000) -> List[str]:
    """Load prompts from pickle (list of str, list of dict, or pandas DataFrame). Default prompts if missing."""
    if prompts_path and Path(prompts_path).exists():
        import pickle
        with open(prompts_path, "rb") as f:
            data = pickle.load(f)
        # Avoid "truth value of DataFrame is ambiguous"
        if data is None:
            return _default_prompts(num_prompts)
        if hasattr(data, "empty") and data.empty:
            return _default_prompts(num_prompts)
        if isinstance(data, (list, tuple)) and len(data) == 0:
            return _default_prompts(num_prompts)
        # Handle pandas DataFrame
        if hasattr(data, "columns") and hasattr(data, "iloc"):
            for col in ("prompt", "text", "content", "article", "prompts"):
                if col in data.columns:
                    prompts = data[col].astype(str).tolist()[:num_prompts]
                    break
            else:
                return _default_prompts(num_prompts)
        else:
            first = data[0]
            if isinstance(first, str):
                prompts = list(data)[:num_prompts]
            elif isinstance(first, dict):
                prompts = []
                for item in data[:num_prompts]:
                    p = item.get("prompt") or item.get("text") or item.get("content")
                    if p is not None:
                        prompts.append(p if isinstance(p, str) else str(p))
            else:
                prompts = _default_prompts(num_prompts)
        return prompts if prompts else _default_prompts(num_prompts)
    return _default_prompts(num_prompts)


def _default_prompts(n: int) -> List[str]:
    base = [
        "What is machine learning?",
        "Explain quantum computing.",
        "How does photosynthesis work?",
        "Describe the water cycle.",
        "What is the theory of relativity?",
    ]
    return (base * (n // len(base) + 1))[:n]


async def send_request(
    client: httpx.AsyncClient,
    router_url: str,
    prompt: str,
    scheduled_wall_time: float,
    request_timeout_sec: float = 120.0,
) -> Dict:
    """POST one request to router /generate; return result dict with state, latency_ms, ttft_ms, backend, etc.
    Sends start_time=scheduled_wall_time so router and model_server compute latency/TTFT/TBT from it."""
    payload = {
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 256,
        "start_time": scheduled_wall_time,
    }
    try:
        resp = await client.post(
            f"{router_url.rstrip('/')}/generate",
            json=payload,
            timeout=request_timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        # Use server-computed metrics (from scheduled_wall_time) when present
        latency_ms = data.get("latency_ms")
        if latency_ms is None:
            latency_ms = data.get("latency_sec", 0.0) * 1000.0
        ttft_ms = data.get("TTFT_ms")
        if ttft_ms is None:
            ttft_ms = data.get("TTFT", 0.0) * 1000.0
        backend = data.get("backend", "")
        time_to_choose_backend_sec = data.get("time_to_choose_backend", 0.0)
        time_between_tokens_ms = data.get("time_between_tokens_ms")
        tpot_ms = data.get("tpot_ms")
        if tpot_ms is None and time_between_tokens_ms:
            tpot_ms = sum(time_between_tokens_ms) / len(time_between_tokens_ms)
        return {
            "success": True,
            "state": "done",
            "latency_ms": float(latency_ms),
            "ttft_ms": float(ttft_ms),
            "backend": backend,
            "time_to_choose_backend_ms": time_to_choose_backend_sec * 1000,
            "tpot_ms": float(tpot_ms) if tpot_ms is not None else None,
            "scheduled_time": scheduled_wall_time,
            "prompt": prompt,
        }
    except Exception as e:
        latency_ms = (time.time() - scheduled_wall_time) * 1000
        return {
            "success": False,
            "state": "failed",
            "latency_ms": latency_ms,
            "ttft_ms": None,
            "backend": None,
            "time_to_choose_backend_ms": None,
            "scheduled_time": scheduled_wall_time,
            "prompt": prompt,
            "error": str(e),
        }


async def generate_load(
    router_url: str,
    target_rps: float,
    duration_sec: float,
    prompts: List[str],
    request_timeout_sec: float = 120.0,
    completion_timeout_sec: Optional[float] = None,
    max_concurrency: int = 100,
    max_pending_multiplier: int = 10,
    arrival_process: str = "exponential",
) -> Tuple[List[Dict], List[float], bool]:
    """
    Open-loop load at target_rps for duration_sec. Returns (results, request_times, completion_timed_out).
    Timed-out requests are appended to results with state "timed_out".

    arrival_process: "exponential" (Poisson, inter-arrival ~ Exp(1/target_rps)) or "uniform" (equal spacing).
    """
    if target_rps <= 0:
        return [], [], False

    results: List[Dict] = []
    request_times: List[float] = []
    completion_timed_out = False
    min_pending = int(target_rps * duration_sec) + 100
    max_pending = max(max_concurrency * max_pending_multiplier, min_pending)
    task_infos: List[Tuple[asyncio.Task, float, str]] = []
    loop = asyncio.get_running_loop()
    loop_start = loop.time()
    wall_start = time.time()
    prompt_idx = 0
    sent_count = 0
    # Next scheduled arrival time (seconds from loop_start). For exponential: cumulative sum of Exp(1/target_rps).
    next_arrival_elapsed = 0.0

    # httpx defaults to max_connections=100. At target_rps with multi-second
    # latencies we exceed that; requests queue waiting for a connection before
    # being sent, inflating "time to reach router". Raise limit to avoid queueing.
    limits = httpx.Limits(max_connections=max(500, int(target_rps * 20)))
    async with httpx.AsyncClient(timeout=request_timeout_sec, limits=limits) as client:

        async def bounded_send(prompt: str, scheduled_wall_time: float):
            r = await send_request(client, router_url, prompt, scheduled_wall_time, request_timeout_sec)
            results.append(r)

        while loop.time() - loop_start < duration_sec:
            elapsed = loop.time() - loop_start
            if elapsed >= next_arrival_elapsed:
                prompt = prompts[prompt_idx % len(prompts)]
                prompt_idx += 1
                sent_count += 1
                scheduled_wall_time = time.time()
                request_times.append(scheduled_wall_time)
                task = asyncio.create_task(bounded_send(prompt, scheduled_wall_time))
                task_infos.append((task, scheduled_wall_time, prompt))
                # Schedule next arrival
                if arrival_process == "exponential":
                    next_arrival_elapsed += float(np.random.exponential(scale=1.0 / target_rps))
                else:
                    next_arrival_elapsed = sent_count / target_rps
                if len(task_infos) >= max_pending:
                    tasks_only = [t for t, _, _ in task_infos]
                    _, pending_set = await asyncio.wait(tasks_only, return_when=asyncio.FIRST_COMPLETED)
                    task_infos = [(t, sw, p) for (t, sw, p) in task_infos if t in pending_set]
            else:
                await asyncio.sleep(max(0.0, min(0.01, next_arrival_elapsed - elapsed)))

    if task_infos:
        tasks_only = [t for t, _, _ in task_infos]
        if completion_timeout_sec is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(asyncio.gather(*tasks_only, return_exceptions=True)),
                    timeout=completion_timeout_sec,
                )
            except asyncio.TimeoutError:
                timeout_wall = time.time()
                pending_infos = [(t, sw, p) for (t, sw, p) in task_infos if (not t.done()) or t.cancelled()]
                if pending_infos:
                    completion_timed_out = True
                    for t, scheduled_wall_time, prompt in pending_infos:
                        if not t.done():
                            t.cancel()
                        results.append({
                            "state": "timed_out",
                            "success": False,
                            "scheduled_time": scheduled_wall_time,
                            "prompt": prompt,
                            "latency_ms": (timeout_wall - scheduled_wall_time) * 1000,
                            "ttft_ms": None,
                            "backend": None,
                            "time_to_choose_backend_ms": None,
                            "error": "completion_timeout",
                        })
                    await asyncio.gather(*tasks_only, return_exceptions=True)
                    print(f"  Completion timeout after {completion_timeout_sec}s ({len(pending_infos)} requests timed out)")
                else:
                    await asyncio.gather(*tasks_only, return_exceptions=True)
        else:
            await asyncio.gather(*tasks_only, return_exceptions=False)

    return results, request_times, completion_timed_out


def compute_statistics(results: List[Dict], request_times: List[float]) -> Dict:
    """Compute throughput, latency, TTFT, success rate, backend counts."""
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if r.get("state") == "failed" or (not r.get("success", False) and r.get("state") != "timed_out")]
    timed_out = [r for r in results if r.get("state") == "timed_out"]
    total = len(results)

    out = {
        "total_requests": total,
        "failed_requests": len(failed),
        "timed_out_requests": len(timed_out),
        "success_rate": len(successful) / total if total else 0.0,
        "throughput_rps": 0.0,
        "avg_latency_ms": 0.0,
        "p50_latency_ms": 0.0,
        "p99_latency_ms": 0.0,
        "min_latency_ms": 0.0,
        "max_latency_ms": 0.0,
        "std_latency_ms": 0.0,
        "avg_ttft_ms": 0.0,
        "p50_ttft_ms": 0.0,
        "p99_ttft_ms": 0.0,
        "avg_time_to_choose_backend_ms": 0.0,
        "avg_tbt_ms": 0.0,
        "p50_tbt_ms": 0.0,
        "p99_tbt_ms": 0.0,
        "backend_counts": {},
    }

    if not successful:
        return out

    latencies = [r["latency_ms"] for r in successful]
    ls = sorted(latencies)
    if request_times:
        time_span = request_times[-1] - request_times[0]
        out["throughput_rps"] = len(successful) / max(time_span, 0.1)
    out["avg_latency_ms"] = float(np.mean(latencies))
    out["p50_latency_ms"] = float(ls[len(ls) // 2])
    out["p99_latency_ms"] = float(ls[min(int(len(ls) * 0.99), len(ls) - 1)]) if len(ls) > 1 else float(ls[0])
    out["min_latency_ms"] = float(min(latencies))
    out["max_latency_ms"] = float(max(latencies))
    out["std_latency_ms"] = float(np.std(latencies))

    ttfts = [r["ttft_ms"] for r in successful if r.get("ttft_ms") is not None]
    if ttfts:
        ttfts_sorted = sorted(ttfts)
        out["avg_ttft_ms"] = float(np.mean(ttfts))
        out["p50_ttft_ms"] = float(ttfts_sorted[len(ttfts_sorted) // 2])
        out["p99_ttft_ms"] = float(ttfts_sorted[min(int(len(ttfts_sorted) * 0.99), len(ttfts_sorted) - 1)]) if len(ttfts_sorted) > 1 else float(ttfts_sorted[0])

    choose_times = [r["time_to_choose_backend_ms"] for r in successful if r.get("time_to_choose_backend_ms") is not None]
    if choose_times:
        out["avg_time_to_choose_backend_ms"] = float(np.mean(choose_times))

    tpots = [r["tpot_ms"] for r in successful if r.get("tpot_ms") is not None]
    if tpots:
        tpots_sorted = sorted(tpots)
        out["avg_tbt_ms"] = float(np.mean(tpots))
        out["p50_tbt_ms"] = float(tpots_sorted[len(tpots_sorted) // 2])
        out["p99_tbt_ms"] = float(tpots_sorted[min(int(len(tpots_sorted) * 0.99), len(tpots_sorted) - 1)])

    backends: Dict[str, int] = {}
    for r in successful:
        b = r.get("backend") or "unknown"
        backends[b] = backends.get(b, 0) + 1
    out["backend_counts"] = backends

    return out


def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    return obj


async def main():
    parser = argparse.ArgumentParser(
        description="Collect routing performance at a fixed load (router + 2 backends must be running)"
    )
    parser.add_argument("--router-url", type=str, default="http://127.0.0.1:8000", help="Router base URL")
    parser.add_argument("--load-rps", type=float, default=20.0, help="Target requests per second (open-loop)")
    parser.add_argument("--duration", type=int, default=60, help="Load duration in seconds")
    parser.add_argument("--output", type=str, default="routing_performance.json", help="Output JSON path")
    parser.add_argument("--prompts-path", type=str, default="datasets/lmsys_chat1m_prompts_100k_cleaned.pkl", help="Pickle with list of prompts")
    parser.add_argument("--request-timeout-sec", type=float, default=120.0, help="Per-request HTTP timeout")
    parser.add_argument("--completion-timeout-sec", type=float, default=None, help="Max wait for in-flight requests after send phase (default: duration*3)")
    parser.add_argument("--max-concurrency", type=int, default=100, help="Max concurrent requests")
    parser.add_argument("--arrival-process", type=str, default="exponential", choices=["exponential", "uniform"],
                        help="Inter-arrival distribution: exponential (Poisson) or uniform (equal spacing)")
    args = parser.parse_args()

    completion_timeout = args.completion_timeout_sec
    if completion_timeout is None:
        completion_timeout = max(args.duration * 3, 300)

    num_prompts_needed = int(args.load_rps * args.duration) + 5000
    print("Loading prompts...")
    prompts = load_prompts(args.prompts_path, num_prompts=num_prompts_needed)
    print(f"  Loaded {len(prompts)} prompts")

    print(f"Sending load to {args.router_url} at {args.load_rps} RPS for {args.duration}s (arrival: {args.arrival_process})...")
    results, request_times, completion_timed_out = await generate_load(
        router_url=args.router_url,
        target_rps=args.load_rps,
        duration_sec=float(args.duration),
        prompts=prompts,
        request_timeout_sec=args.request_timeout_sec,
        completion_timeout_sec=completion_timeout,
        max_concurrency=args.max_concurrency,
        arrival_process=args.arrival_process,
    )

    stats = compute_statistics(results, request_times)
    stats["completion_timed_out"] = completion_timed_out

    # Same format as collect_performance_data: ttfts_ms and tbt_lists_ms (no prompts)
    successful = [r for r in results if r.get("success", False)]
    ttfts_ms = [r["ttft_ms"] for r in successful if r.get("ttft_ms") is not None]
    tpot_ms_list = [r["tpot_ms"] for r in successful if r.get("tpot_ms") is not None]

    # List of results: one element per request (per prompt), with ttft_ms, tpot_ms, backend, latency_ms, time_to_choose_backend_ms
    results_list = [
        {
            "ttft_ms": r.get("ttft_ms"),
            "tpot_ms": r.get("tpot_ms"),
            "backend": r.get("backend"),
            "latency_ms": r.get("latency_ms"),
            "time_to_choose_backend_ms": r.get("time_to_choose_backend_ms"),
        }
        for r in results
    ]

    payload = {
        "config": {
            "router_url": args.router_url,
            "load_rps": args.load_rps,
            "duration_sec": args.duration,
            "request_timeout_sec": args.request_timeout_sec,
            "completion_timeout_sec": completion_timeout,
            "arrival_process": args.arrival_process,
        },
        "performance": _to_native(stats),
        "ttfts_ms": _to_native(ttfts_ms),
        "tpot_ms_list": _to_native(tpot_ms_list),
        "results": _to_native(results_list),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"  Total requests:  {stats['total_requests']}")
    print(f"  Success rate:    {stats['success_rate']:.2%}")
    print(f"  Throughput:      {stats['throughput_rps']:.2f} RPS")
    print(f"  Avg latency:     {stats['avg_latency_ms']:.1f} ms")
    print(f"  P99 latency:    {stats['p99_latency_ms']:.1f} ms")
    print(f"  Avg TTFT:        {stats['avg_ttft_ms']:.1f} ms")
    print(f"  P99 TTFT:        {stats['p99_ttft_ms']:.1f} ms")
    print(f"  Backend counts:  {stats['backend_counts']}")


if __name__ == "__main__":
    asyncio.run(main())
