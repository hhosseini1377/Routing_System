#!/usr/bin/env python3
"""
Data collection script for performance regression models.

Tests a single model server with different resource allocations and load levels
to collect training data for regression models.

Usage:
    python collect_performance_data.py --model-name "Qwen/Qwen2-1.5B-Instruct" \
        --output performance_data.json \
        --memory-range 0.2 0.8 --thread-range 10 90 --load-range 1 50
"""

import asyncio
import json
import os
import signal
import subprocess
import time
import argparse
import sys
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import httpx
import numpy as np
from pathlib import Path
import warnings

def _to_native_json(obj):
    """Convert numpy types to native Python so json.dump works."""
    if isinstance(obj, dict):
        return {k: _to_native_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native_json(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class PerformanceDataCollector:
    """Collects performance data for a single model under different configurations."""
    
    def __init__(
        self,
        model_name: str,
        root_dir: str = "/data/gpfs/projects/punim2662/routing_system",
        port: int = 8001,
        warmup_duration: int = 10,
        test_duration: int = 60,
    ):
        self.model_name = model_name
        self.root_dir = root_dir
        self.port = port
        self.warmup_duration = warmup_duration
        self.test_duration = test_duration
        self.model_url = f"http://127.0.0.1:{port}"
        self.metrics_url = f"{self.model_url}/metrics"
        # Map from (thread_perc, memory_util, load_rps, tensor_parallel_size, max_model_len?, max_num_seqs?, max_num_batched_tokens?) -> result dict
        self.results: Dict[Tuple, Dict] = {}
        self.service_process = None
        
    async def fetch_metrics(self, timeout_sec: float = 15.0, max_retries: int = 3) -> Dict:
        """Fetch current metrics from the model server. Retries on failure (e.g. after timeouts)."""
        last_err = None
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout_sec) as client:
                    resp = await client.get(self.metrics_url)
                    resp.raise_for_status()
                    return resp.json()
            except Exception as e:
                last_err = e
                print(f"Warning: Failed to fetch metrics (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(3)
        return {}

    async def clear_engine(self, timeout_sec: float = 60.0, max_retries: int = 3) -> bool:
        """Abort all in-flight requests and reset engine. Returns True if successful."""
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout_sec) as client:
                    resp = await client.post(f"{self.model_url}/clear")
                    resp.raise_for_status()
                    data = resp.json()
                    if not data.get("cleared", True):
                        raise RuntimeError(data.get("error", "clear returned cleared=False"))
                    return True
            except Exception as e:
                print(f"  Warning: Failed to clear engine (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(5)
        return False
    
    async def send_request(
        self, client: httpx.AsyncClient, prompt: str, scheduled_wall_time: float, timeout_sec: Optional[float] = None
    ) -> Dict:
        """Send a single request to the model."""
        payload = {
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 256,
            "start_time": scheduled_wall_time,
        }
        try:
            timeout = timeout_sec if timeout_sec is not None else 120.0
            resp = await client.post(f"{self.model_url}/generate", json=payload, timeout=timeout)
            resp.raise_for_status()
            response_data = resp.json()
            elapsed = (time.time() - scheduled_wall_time) * 1000
            result = {
                "success": True,
                "state": "done",
                "latency_ms": elapsed,
                "prompt": prompt,
            }
            # Extract TTFT and TPOT (avg TBT per prompt) if available
            if "TTFT" in response_data:
                result["ttft_ms"] = response_data["TTFT"] * 1000  # Convert seconds to ms
            if "tpot_ms" in response_data:
                result["tpot_ms"] = response_data["tpot_ms"]
            elif "time_between_tokens_ms" in response_data:
                tbts = response_data["time_between_tokens_ms"]
                result["tpot_ms"] = sum(tbts) / len(tbts) if tbts else 0.0
            return result
        except Exception as e:
            elapsed = (time.time() - scheduled_wall_time) * 1000
            return {
                "success": False,
                "state": "failed",
                "latency_ms": elapsed,
                "error": str(e),
                "prompt": prompt,
            }
    
    async def generate_load(
        self,
        target_rps: float,
        duration_sec: int,
        prompts: List[str],
        max_concurrency: int = 100,
        max_pending_multiplier: int = 10,
        completion_timeout_sec: Optional[float] = None,
        request_timeout_sec: float = 120.0,
    ) -> Tuple[List[Dict], List[float], bool]:
        """
        Generate open-loop load at target RPS for specified duration.

        This schedules request *arrivals* at the target rate (open-loop), instead
        of waiting for each request to finish before sending the next (closed-loop).

        Args:
            target_rps: Target arrival rate (requests/sec)
            duration_sec: How long to generate arrivals
            prompts: Prompt list to cycle through
            max_concurrency: Max in-flight requests (caps pressure on the server/client)
            max_pending_multiplier: Max pending tasks = max_concurrency * multiplier
            completion_timeout_sec: Max time to wait for requests to complete after send
                phase. If None, wait indefinitely. When set, harsh setups won't block forever.
            request_timeout_sec: Per-request HTTP timeout (seconds).

        Returns:
            (results, request_times, completion_timed_out):
              - results: list of per-request dicts (success/latency/etc.)
              - request_times: wall-clock timestamps when requests were scheduled
              - completion_timed_out: True if we stopped waiting before all requests finished
        """
        if target_rps <= 0:
            return [], [], False

        results: List[Dict] = []
        request_times: List[float] = []
        completion_timed_out = False

        semaphore = asyncio.Semaphore(max_concurrency)
        # Need enough headroom to schedule all open-loop arrivals (target_rps * duration_sec)
        min_pending_for_open_loop = int(target_rps * duration_sec) + 100
        max_pending = max(max_concurrency * max_pending_multiplier, min_pending_for_open_loop)
        # Track (task, scheduled_wall_time, prompt) so we can add timed_out results for cancelled tasks
        task_infos: List[Tuple[asyncio.Task, float, str]] = []

        loop = asyncio.get_running_loop()
        loop_start = loop.time()          # monotonic, good for scheduling/sleeps
        wall_start = time.time()          # wall-clock, good for logging/metrics

        prompt_idx = 0
        sent_count = 0

        async with httpx.AsyncClient(timeout=request_timeout_sec) as client:

            async def bounded_send(prompt: str, scheduled_wall_time: float):
                async with semaphore:
                    r = await self.send_request(client, prompt, scheduled_wall_time, timeout_sec=request_timeout_sec)
                    # Attach when it was scheduled (arrival time)
                    r["scheduled_time"] = scheduled_wall_time
                    results.append(r)

            while (loop.time() - loop_start) < duration_sec:
                now_loop = loop.time()
                elapsed = now_loop - loop_start

                # Ideal arrival time for the next request (open-loop)
                next_arrival_elapsed = sent_count / target_rps

                if elapsed >= next_arrival_elapsed:
                    prompt = prompts[prompt_idx % len(prompts)]
                    prompt_idx += 1
                    sent_count += 1

                    scheduled_wall_time = time.time()
                    request_times.append(scheduled_wall_time)

                    task = asyncio.create_task(bounded_send(prompt, scheduled_wall_time))
                    task_infos.append((task, scheduled_wall_time, prompt))

                    # Prevent unbounded growth of tasks waiting on the semaphore
                    # if len(task_infos) >= max_pending:
                    #     tasks_only = [t for t, _, _ in task_infos]
                    #     done_set, pending_set = await asyncio.wait(tasks_only, return_when=asyncio.FIRST_COMPLETED)
                    #     task_infos = [(t, sw, p) for (t, sw, p) in task_infos if t in pending_set]
                else:
                    # Sleep until next ideal arrival (or a small slice)
                    await asyncio.sleep(max(0.0, next_arrival_elapsed - elapsed))

            # Wait for scheduled requests to finish, with optional timeout
            if task_infos:
                tasks_only = [t for t, _, _ in task_infos]
                if completion_timeout_sec is not None:
                    try:
                        await asyncio.wait_for(
                            # Shield so wait_for timeout doesn't auto-cancel the tasks.
                            asyncio.shield(asyncio.gather(*tasks_only, return_exceptions=True)),
                            timeout=completion_timeout_sec,
                        )
                    except asyncio.TimeoutError:
                        timeout_wall = time.time()
                        # Anything still running (or cancelled) at this point is a timeout.
                        timeout_infos = [(t, sw, p) for (t, sw, p) in task_infos if (not t.done()) or t.cancelled()]
                        if timeout_infos:
                            completion_timed_out = True
                            timed_out_count = 0
                            for t, scheduled_wall_time, prompt in timeout_infos:
                                if not t.done():
                                    t.cancel()
                                    results.append({
                                        "state": "timed_out",
                                        "success": False,
                                        "scheduled_time": scheduled_wall_time,
                                        "prompt": prompt,
                                        "latency_ms": (timeout_wall - scheduled_wall_time) * 1000,
                                        "error": "completion_timeout",
                                    })
                                    timed_out_count += 1
                            await asyncio.gather(*tasks_only, return_exceptions=True)
                            print(f"  Completion timeout after {completion_timeout_sec}s ({timed_out_count} requests timed out)")
                        else:
                            # Everything already finished; continue quietly.
                            await asyncio.gather(*tasks_only, return_exceptions=True)
                else:
                    await asyncio.gather(*tasks_only, return_exceptions=False)

        return results, request_times, completion_timed_out
    
    def start_model_server(
        self,
        memory_util: float,
        thread_percentage: int,
        startup_timeout_sec: int = 180,
        tensor_parallel_size: int = 2,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
    ) -> subprocess.Popen:
        """Start model server with specified configuration."""
        # Stop any existing server on this port
        subprocess.run(
            f"pkill -f 'uvicorn.*port.*{self.port}' || true",
            shell=True,
            cwd=self.root_dir
        )
        time.sleep(2)

        env_parts = [
            f"cd {self.root_dir} && ",
            f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={thread_percentage} ",
            f"MODEL_NAME=\"{self.model_name}\" ",
            f"TENSOR_PARALLEL_SIZE={tensor_parallel_size} ",
            f"GPU_MEMORY_UTILIZATION={memory_util} ",
            f"UVICORN_PORT={self.port} ",
        ]
        if max_model_len is not None:
            env_parts.append(f"MAX_MODEL_LEN={max_model_len} ")
        if max_num_seqs is not None:
            env_parts.append(f"MAX_NUM_SEQS={max_num_seqs} ")
        if max_num_batched_tokens is not None:
            env_parts.append(f"MAX_NUM_BATCHED_TOKENS={max_num_batched_tokens} ")

        cmd = [
            "bash", "-c",
            "".join(env_parts) + "uvicorn servers.model_server:app --host 0.0.0.0 --port " + str(self.port)
        ]
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # own process group so we can kill entire tree (uvicorn + vLLM workers)
        )
        
        # Wait for server to be ready
        print(f"  Waiting for server to start...")
        deadline = time.time() + startup_timeout_sec
        while time.time() < deadline:
            # If the process crashed, surface logs immediately.
            if proc.poll() is not None:
                try:
                    out, err = proc.communicate(timeout=2)
                except Exception:
                    out, err = "", ""
                msg = [
                    "Model server process exited during startup.",
                    f"Exit code: {proc.returncode}",
                ]
                if err:
                    msg.append("---- stderr (last 4000 chars) ----")
                    msg.append(err[-4000:])
                if out:
                    msg.append("---- stdout (last 4000 chars) ----")
                    msg.append(out[-4000:])
                raise RuntimeError("\n".join(msg))

            try:
                resp = httpx.get(f"{self.model_url}/metrics", timeout=2.0)
                if resp.status_code == 200:
                    print(f"  Server ready!")
                    break
                else:
                    print(resp.status_code)
            except:
                pass
            time.sleep(1)
        else:
            # Still running but not ready (or stuck). Try to surface partial logs.
            msg = [
                f"Server failed to become ready within {startup_timeout_sec}s.",
                "It may still be loading, or it may be stuck.",
                f"Try increasing --startup-timeout-sec, lowering tensor parallelism, or checking GPU memory.",
            ]
            raise RuntimeError("\n".join(msg))
        
        return proc
    
    def stop_model_server(self):
        """Stop the model server and all child processes (uvicorn, vLLM workers)."""
        if self.service_process and self.service_process.poll() is None:
            pid = self.service_process.pid
            try:
                # Kill entire process group (bash + uvicorn + vLLM worker children)
                if hasattr(os, "killpg") and pid is not None:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGTERM)
                else:
                    self.service_process.terminate()
            except (ProcessLookupError, OSError):
                self.service_process.terminate()
            try:
                self.service_process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    if hasattr(os, "killpg") and pid is not None:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                try:
                    self.service_process.kill()
                    self.service_process.wait(timeout=3)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    pass
            self.service_process = None

        # Fallback: kill by port (catches any process using it, including orphaned workers)
        subprocess.run(
            f"fuser -k {self.port}/tcp 2>/dev/null || true",
            shell=True,
            cwd=self.root_dir,
        )
        subprocess.run(
            f"pkill -9 -f 'uvicorn.*port.*{self.port}' || true",
            shell=True,
            cwd=self.root_dir,
        )
        time.sleep(2)
    
    def compute_statistics(self, results: List[Dict], request_times: List[float]) -> Dict:
        """Compute performance statistics from results.
        total_requests = number of submitted requests (done + failed + timed_out).
        Each result has state: 'done' | 'failed' | 'timed_out'.
        """
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if r.get("state") == "failed" or (not r.get("success", False) and r.get("state") != "timed_out")]
        timed_out = [r for r in results if r.get("state") == "timed_out"]
        # total_requests = all submitted (should equal len(results))
        total_requests = len(results)

        if not successful:
            return {
                "success_rate": 0.0,
                "throughput_rps": 0.0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "total_requests": total_requests,
                "failed_requests": len(failed),
                "timed_out_requests": len(timed_out),
            }

        latencies = [r["latency_ms"] for r in successful]
        latencies_sorted = sorted(latencies)

        # Compute throughput (successful completions per second)
        if len(request_times) > 1:
            time_span = request_times[-1] - request_times[0]
            throughput_rps = len(successful) / max(time_span, 0.1)
        else:
            throughput_rps = 0.0

        return {
            "success_rate": len(successful) / total_requests,
            "throughput_rps": throughput_rps,
            "avg_latency_ms": np.mean(latencies),
            "p50_latency_ms": latencies_sorted[len(latencies_sorted) // 2],
            "p99_latency_ms": latencies_sorted[int(len(latencies_sorted) * 0.99)] if len(latencies_sorted) > 1 else latencies_sorted[0],
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies),
            "std_latency_ms": np.std(latencies),
            "total_requests": total_requests,
            "failed_requests": len(failed),
            "timed_out_requests": len(timed_out),
        }
    
    async def run_experiment(
        self,
        memory_util: float,
        thread_percentage: int,
        load_rps: float,
        prompts: List[str],
        max_concurrency: int = 100,
        startup_timeout_sec: int = 180,
        tensor_parallel_size: int = 2,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        completion_timeout_sec: Optional[float] = None,
        request_timeout_sec: float = 90.0,
        skip_server_start: bool = False,
        skip_server_stop: bool = False,
        skip_warmup: bool = False,
        warmup_rps: Optional[float] = None,
    ) -> Optional[Dict]:
        """Run a single experiment configuration.

        When skip_server_start/skip_server_stop/skip_warmup are True, assumes server is
        already running (used when testing multiple loads on the same engine setup).

        warmup_rps: RPS for warmup (default 5). Use low RPS to warm model without stressing it.
        """
        # Cap how long we wait for completions so harsh setups don't block forever
        if completion_timeout_sec is None:
            completion_timeout_sec = max(self.test_duration * 3, 300)
        warmup_completion_timeout = 10
        warmup_load_rps = warmup_rps if warmup_rps is not None else 5.0

        print(f"\n{'='*80}")
        print(f"Testing: mem={memory_util:.2f}, threads={thread_percentage}%, load={load_rps:.1f} RPS"
              f", max_model_len={max_model_len}, max_num_seqs={max_num_seqs}, max_num_batched_tokens={max_num_batched_tokens}")
        print(f"{'='*80}")
        
        try:
            # Start server (unless caller manages lifecycle for multiple loads)
            if not skip_server_start:
                self.service_process = self.start_model_server(
                    memory_util,
                    thread_percentage,
                    startup_timeout_sec=startup_timeout_sec,
                    tensor_parallel_size=tensor_parallel_size,
                    max_model_len=max_model_len,
                    max_num_seqs=max_num_seqs,
                    max_num_batched_tokens=max_num_batched_tokens,
                )
            
            # Warmup (skip when server already warm from previous load test)
            if not skip_warmup:
                print(f"  Warmup ({self.warmup_duration}s @ {warmup_load_rps:.1f} RPS, completion timeout {warmup_completion_timeout}s)...")
                await self.generate_load(
                    warmup_load_rps,
                    self.warmup_duration,
                    prompts,
                    max_concurrency=max_concurrency,
                    completion_timeout_sec=warmup_completion_timeout,
                    request_timeout_sec=request_timeout_sec,
                )
                await asyncio.sleep(2)
            else:
                await asyncio.sleep(2)  # Brief pause to let queue drain between loads
            
            # Collect metrics before test
            metrics_before = await self.fetch_metrics()
            
            # Run test
            print(f"  Running test ({self.test_duration}s, completion timeout {completion_timeout_sec}s)...")
            results, request_times, completion_timed_out = await self.generate_load(
                load_rps,
                self.test_duration,
                prompts,
                max_concurrency=max_concurrency,
                completion_timeout_sec=completion_timeout_sec,
                request_timeout_sec=request_timeout_sec,
            )
            
            # Collect metrics after test
            await asyncio.sleep(1)
            metrics_after = await self.fetch_metrics()
            
            # Compute statistics (uses whatever completed; partial if timed out)
            stats = self.compute_statistics(results, request_times)
            
            # Collect all TTFTs and TPOTs (one per request)
            ttfts_ms: List[float] = []
            tpot_ms_list: List[float] = []
            for r in results:
                if r.get("success"):
                    if "ttft_ms" in r:
                        ttfts_ms.append(r["ttft_ms"])
                    if "tpot_ms" in r:
                        tpot_ms_list.append(r["tpot_ms"])
            
            # Extract queue metrics
            queue_before = metrics_before.get("in_flight_requests", 0)
            queue_after = metrics_after.get("in_flight_requests", 0)
            avg_queue = (queue_before + queue_after) / 2
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "memory_utilization": memory_util,
                    "thread_percentage": thread_percentage,
                    "load_rps": load_rps,
                    "model_name": self.model_name,
                    "max_model_len": max_model_len,
                    "max_num_seqs": max_num_seqs,
                    "max_num_batched_tokens": max_num_batched_tokens,
                },
                "performance": {
                    **stats,
                    "avg_queue_size": avg_queue,
                    "max_queue_size": metrics_after.get("in_flight_requests", 0),
                    "completion_timed_out": completion_timed_out,
                },
                "ttfts_ms": ttfts_ms,
                "tpot_ms_list": tpot_ms_list,
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
            }
            
            print(f"  Results:")
            print(f"    Throughput: {stats['throughput_rps']:.2f} RPS")
            print(f"    Avg Latency: {stats['avg_latency_ms']:.1f}ms")
            print(f"    P99 Latency: {stats['p99_latency_ms']:.1f}ms")
            print(f"    Success Rate: {stats['success_rate']:.1%}")
            
            return result
            
        except Exception as e:
            print(f"  ERROR: {e}")
            return {
                "failed": True,
                "error": str(e),
                "config": {
                    "memory_utilization": memory_util,
                    "thread_percentage": thread_percentage,
                    "load_rps": load_rps,
                    "model_name": self.model_name,
                    "max_model_len": max_model_len,
                    "max_num_seqs": max_num_seqs,
                    "max_num_batched_tokens": max_num_batched_tokens,
                },
            }
        finally:
            if not skip_server_stop:
                self.stop_model_server()
    
    def load_results(self, output_file: str):
        """Load existing results from JSON file if it exists."""
        if Path(output_file).exists():
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
                    for item in data:
                        setup = item["setup"]
                        key = (
                            int(setup["thread_percentage"]),
                            float(setup["memory_util"]),
                            float(setup["load_rps"]),
                            int(setup["tensor_parallel_size"]),
                            setup.get("max_model_len"),
                            setup.get("max_num_seqs"),
                            setup.get("max_num_batched_tokens"),
                        )
                        self.results[key] = item["result"]
                    print(f"Loaded {len(self.results)} existing results from {output_file}")
            except Exception as e:
                print(f"Warning: Failed to load existing results: {e}")
    
    def save_results(self, output_file: str, incremental: bool = False):
        """Save collected results to JSON file. Map keys (setup tuples) become 'setup' objects.
        
        Args:
            output_file: Path to output JSON file
            incremental: If True, only print a message (for incremental saves after each experiment)
        """
        serializable = []
        for k, v in self.results.items():
            setup = {
                "thread_percentage": _to_native_json(k[0]),
                "memory_util": _to_native_json(k[1]),
                "load_rps": _to_native_json(k[2]),
                "tensor_parallel_size": _to_native_json(k[3]),
            }
            # Extended key (max_model_len, max_num_seqs, max_num_batched_tokens); old files have 4-tuple keys
            if len(k) >= 7:
                setup["max_model_len"] = _to_native_json(k[4])
                setup["max_num_seqs"] = _to_native_json(k[5])
                setup["max_num_batched_tokens"] = _to_native_json(k[6])
            serializable.append({"setup": setup, "result": _to_native_json(v)})
        with open(output_file, "w") as f:
            json.dump(serializable, f, indent=2)
        if incremental:
            print(f"  Saved {len(self.results)} experiment(s) to {output_file}")
        else:
            print(f"\n\nResults saved to {output_file}")
            print(f"Total experiments: {len(self.results)}")


def load_prompts(prompts_path: str = None, num_prompts: int = 1000) -> List[str]:
    """Load prompts for testing."""
    if prompts_path and Path(prompts_path).exists():
        import pickle
        with open(prompts_path, 'rb') as f:
            data = pickle.load(f)
            return data[:num_prompts]
    else:
        warnings.warn(f"Prompts file not found: {prompts_path}")
        # Default prompts if file not found
        return [
            "What is machine learning?",
            "Explain quantum computing.",
            "How does photosynthesis work?",
            "Describe the water cycle.",
            "What is the theory of relativity?",
        ] * (num_prompts // 5 + 1)


async def main():
    parser = argparse.ArgumentParser(description="Collect performance data for regression models")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name (HuggingFace ID)")
    parser.add_argument("--output", type=str, default="performance_data.json",
                        help="Output JSON file")
    parser.add_argument("--port", type=int, default=8001,
                        help="Port for model server")
    parser.add_argument("--memory-range", type=float, nargs=2, default=[0.2, 0.8],
                        metavar=("MIN", "MAX"), help="Memory utilization range")
    parser.add_argument("--memory-steps", type=int, default=4,
                        help="Number of memory utilization steps")
    parser.add_argument("--thread-range", type=int, nargs=2, default=[10, 90],
                        metavar=("MIN", "MAX"), help="Thread percentage range")
    parser.add_argument("--thread-steps", type=int, default=4,
                        help="Number of thread percentage steps")
    parser.add_argument("--load-range", type=float, nargs=2, default=[1.0, 50.0],
                        metavar=("MIN", "MAX"), help="Load (RPS) range")
    parser.add_argument("--load-steps", type=int, default=5,
                        help="Number of load steps")
    parser.add_argument("--warmup-duration", type=int, default=10,
                        help="Warmup duration in seconds")
    parser.add_argument("--warmup-rps", type=float, default=5.0,
                        help="RPS for warmup (default 5). Use low RPS to warm model without stressing it.")
    parser.add_argument("--test-duration", type=int, default=60,
                        help="Test duration in seconds")
    parser.add_argument("--prompts-path", type=str, default=None,
                        help="Path to prompts pickle file")
    parser.add_argument("--max-experiments", type=int, default=None,
                        help="Maximum number of experiments to run")
    parser.add_argument("--max-concurrency", type=int, default=100,
                        help="Max in-flight requests during load generation")
    parser.add_argument("--startup-timeout-sec", type=int, default=180,
                        help="Seconds to wait for model server /metrics to become ready")
    parser.add_argument("--tensor-parallel-size", type=int, default=2,
                        help="TENSOR_PARALLEL_SIZE for vLLM (must be <= visible GPU count). Ignored if --tensor-parallel-sizes or --tensor-parallel-size-range is set.")
    parser.add_argument("--tensor-parallel-sizes", type=int, nargs="+", default=None,
                        metavar=("SIZE", "..."), help="Exact tensor_parallel_size values to sweep, e.g. --tensor-parallel-sizes 1 2 4")
    parser.add_argument("--tensor-parallel-size-range", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"), help="Sweep tensor_parallel_size over [MIN, MAX] with --tensor-parallel-size-steps")
    parser.add_argument("--tensor-parallel-size-steps", type=int, default=1,
                        help="Number of tensor_parallel_size steps when using --tensor-parallel-size-range (default 1)")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="MAX_MODEL_LEN for vLLM (default 2048). Ignored if --max-model-len-range is set.")
    parser.add_argument("--max-model-len-range", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"), help="Sweep max_model_len over [MIN, MAX] with --max-model-len-steps")
    parser.add_argument("--max-model-len-steps", type=int, default=1,
                        help="Number of max_model_len steps when using --max-model-len-range (default 1)")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="MAX_NUM_SEQS for vLLM (optional). Ignored if --max-num-seqs-range is set.")
    parser.add_argument("--max-num-seqs-range", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"), help="Sweep max_num_seqs over [MIN, MAX] with --max-num-seqs-steps")
    parser.add_argument("--max-num-seqs-steps", type=int, default=1,
                        help="Number of max_num_seqs steps when using --max-num-seqs-range (default 1)")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        help="MAX_NUM_BATCHED_TOKENS for vLLM (optional). Ignored if --max-num-batched-tokens-range is set.")
    parser.add_argument("--max-num-batched-tokens-range", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"), help="Sweep max_num_batched_tokens over [MIN, MAX] with --max-num-batched-tokens-steps")
    parser.add_argument("--max-num-batched-tokens-steps", type=int, default=1,
                        help="Number of max_num_batched_tokens steps when using --max-num-batched-tokens-range (default 1)")
    parser.add_argument("--completion-timeout-sec", type=float, default=None,
                        help="Max seconds to wait for requests to complete after send phase (default: max(test_duration*3, 300)). Harsh setups will stop early.")
    parser.add_argument("--request-timeout-sec", type=float, default=90.0,
                        help="Per-request HTTP timeout in seconds (default: 90)")
    
    args = parser.parse_args()

    # Optional sweep: tensor_parallel_size, max_model_len, max_num_seqs, max_num_batched_tokens
    # Sort TP descending so we can skip smaller TP when larger TP fails (infeasibility heuristic)
    if args.tensor_parallel_sizes is not None:
        tensor_parallel_size_values = sorted(set(args.tensor_parallel_sizes), reverse=True)
    elif args.tensor_parallel_size_range is not None and args.tensor_parallel_size_steps > 1:
        tensor_parallel_size_values = sorted(
            np.linspace(
                args.tensor_parallel_size_range[0], args.tensor_parallel_size_range[1],
                args.tensor_parallel_size_steps, dtype=int
            ).tolist(),
            reverse=True,
        )
    else:
        tensor_parallel_size_values = [args.tensor_parallel_size]

    if args.max_model_len_range is not None and args.max_model_len_steps > 1:
        max_model_len_values = np.linspace(
            args.max_model_len_range[0], args.max_model_len_range[1],
            args.max_model_len_steps, dtype=int
        ).tolist()
    else:
        max_model_len_values = [args.max_model_len]

    if args.max_num_seqs_range is not None and args.max_num_seqs_steps > 1:
        max_num_seqs_values = np.linspace(
            args.max_num_seqs_range[0], args.max_num_seqs_range[1],
            args.max_num_seqs_steps, dtype=int
        ).tolist()
    else:
        max_num_seqs_values = [args.max_num_seqs]

    if args.max_num_batched_tokens_range is not None and args.max_num_batched_tokens_steps > 1:
        max_num_batched_tokens_values = np.linspace(
            args.max_num_batched_tokens_range[0], args.max_num_batched_tokens_range[1],
            args.max_num_batched_tokens_steps, dtype=int
        ).tolist()
    else:
        max_num_batched_tokens_values = [args.max_num_batched_tokens]

    # Generate parameter grid
    memory_values = np.linspace(args.memory_range[0], args.memory_range[1], args.memory_steps)
    thread_values = np.linspace(args.thread_range[0], args.thread_range[1], args.thread_steps, dtype=int)
    load_values = np.linspace(args.load_range[0], args.load_range[1], args.load_steps)

    total_experiments = (
        len(memory_values) * len(thread_values) * len(load_values)
        * len(tensor_parallel_size_values) * len(max_model_len_values) * len(max_num_seqs_values) * len(max_num_batched_tokens_values)
    )
    if args.max_experiments:
        total_experiments = min(total_experiments, args.max_experiments)

    print(f"\n{'='*80}")
    print(f"PERFORMANCE DATA COLLECTION")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Memory range: {args.memory_range[0]:.2f} - {args.memory_range[1]:.2f} ({args.memory_steps} steps)")
    print(f"Thread range: {args.thread_range[0]} - {args.thread_range[1]}% ({args.thread_steps} steps)")
    print(f"Load range: {args.load_range[0]:.1f} - {args.load_range[1]:.1f} RPS ({args.load_steps} steps)")
    print(f"tensor_parallel_size: {tensor_parallel_size_values}")
    print(f"max_model_len: {max_model_len_values}")
    print(f"max_num_seqs: {max_num_seqs_values}")
    print(f"max_num_batched_tokens: {max_num_batched_tokens_values}")
    print(f"Total experiments: {total_experiments}")
    print(f"Estimated time: ~{total_experiments * (args.warmup_duration + args.test_duration + 5) / 60:.1f} minutes")
    print(f"{'='*80}\n")
    
    # Load prompts
    prompts = load_prompts(args.prompts_path)
    print(f"Loaded {len(prompts)} prompts")
    
    # Create collector
    collector = PerformanceDataCollector(
        model_name=args.model_name,
        port=args.port,
        warmup_duration=args.warmup_duration,
        test_duration=args.test_duration,
    )
    
    # Load existing results if file exists (for resuming)
    collector.load_results(args.output)
    
    # Run experiments: group by engine setup so we start server once per setup,
    # then run all load levels without restarting.
    experiment_count = len(collector.results)  # Start from existing count
    for memory_util in memory_values:
        for thread_perc in thread_values:
            for max_model_len in max_model_len_values:
                for max_num_seqs in max_num_seqs_values:
                    for max_num_batched_tokens in max_num_batched_tokens_values:

                        for tensor_parallel_size in tensor_parallel_size_values:
                            # print(tensor_parallel_size)
                            # Collect loads we need to run for this engine setup
                            loads_to_run = []
                            for load_rps in load_values:
                                if args.max_experiments and experiment_count >= args.max_experiments:
                                    break
                                setup_key = (
                                    thread_perc,
                                    float(memory_util),
                                    float(load_rps),
                                    tensor_parallel_size,
                                    max_model_len,
                                    max_num_seqs,
                                    max_num_batched_tokens,
                                )
                                if setup_key in collector.results:
                                    print(f"\nSkipping (already exists): mem={memory_util:.2f}, threads={thread_perc}%, tp={tensor_parallel_size}, load={load_rps:.1f} RPS, max_model_len={max_model_len}, max_num_seqs={max_num_seqs}, max_num_batched_tokens={max_num_batched_tokens}")
                                    experiment_count += 1
                                    continue
                                loads_to_run.append((load_rps, setup_key))

                            if not loads_to_run:
                                continue

                            # Run each load with a fresh server (restart for stability)
                            for load_rps, setup_key in loads_to_run:
                                if args.max_experiments and experiment_count >= args.max_experiments:
                                    break

                                result = await collector.run_experiment(
                                    memory_util=memory_util,
                                    thread_percentage=thread_perc,
                                    load_rps=load_rps,
                                    prompts=prompts,
                                    max_concurrency=args.max_concurrency,
                                    startup_timeout_sec=args.startup_timeout_sec,
                                    tensor_parallel_size=tensor_parallel_size,
                                    max_model_len=max_model_len,
                                    max_num_seqs=max_num_seqs,
                                    max_num_batched_tokens=max_num_batched_tokens,
                                    completion_timeout_sec=args.completion_timeout_sec,
                                    request_timeout_sec=args.request_timeout_sec,
                                    skip_server_start=False,
                                    skip_server_stop=False,
                                    skip_warmup=False,
                                    warmup_rps=args.warmup_rps,
                                )
                                collector.results[setup_key] = result
                                experiment_count += 1
                                collector.save_results(args.output, incremental=True)

                                if result.get("failed"):
                                    print(f"\nProgress: {experiment_count}/{total_experiments} (failed)")
                                else:
                                    print(f"\nProgress: {experiment_count}/{total_experiments}")

                            if args.max_experiments and experiment_count >= args.max_experiments:
                                break
                    if args.max_experiments and experiment_count >= args.max_experiments:
                        break
                if args.max_experiments and experiment_count >= args.max_experiments:
                    break
            if args.max_experiments and experiment_count >= args.max_experiments:
                break
        if args.max_experiments and experiment_count >= args.max_experiments:
            break
    
    # Final save (redundant but ensures everything is saved)
    collector.save_results(args.output)
    
    print(f"\n✓ Data collection complete!")
    print(f"  Collected {len(collector.results)} experiments")
    print(f"  Next step: Train regression models with train_performance_models.py")


if __name__ == "__main__":
    asyncio.run(main())
