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
import subprocess
import time
import argparse
import sys
from typing import Dict, List, Tuple
from datetime import datetime
import httpx
import numpy as np
from pathlib import Path


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
        self.results = []
        self.service_process = None
        
    async def fetch_metrics(self) -> Dict:
        """Fetch current metrics from the model server."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(self.metrics_url)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            print(f"Warning: Failed to fetch metrics: {e}")
            return {}
    
    async def send_request(self, client: httpx.AsyncClient, prompt: str) -> Dict:
        """Send a single request to the model."""
        payload = {
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 256,
            "start_time": time.time(),
        }
        try:
            start_time = time.time()
            resp = await client.post(f"{self.model_url}/generate", json=payload, timeout=120.0)
            resp.raise_for_status()
            elapsed = time.time() - start_time
            return {
                "success": True,
                "latency_ms": elapsed * 1000,
                "response": resp.json(),
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "success": False,
                "latency_ms": elapsed * 1000,
                "error": str(e),
            }
    
    async def generate_load(
        self,
        target_rps: float,
        duration_sec: int,
        prompts: List[str],
        max_concurrency: int = 100,
        max_pending_multiplier: int = 10,
    ) -> List[Dict]:
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

        Returns:
            (results, request_times):
              - results: list of per-request dicts (success/latency/etc.)
              - request_times: wall-clock timestamps when requests were scheduled
        """
        if target_rps <= 0:
            return [], []

        results: List[Dict] = []
        request_times: List[float] = []

        semaphore = asyncio.Semaphore(max_concurrency)
        max_pending = max_concurrency * max_pending_multiplier
        tasks: List[asyncio.Task] = []

        loop = asyncio.get_running_loop()
        loop_start = loop.time()          # monotonic, good for scheduling/sleeps
        wall_start = time.time()          # wall-clock, good for logging/metrics

        prompt_idx = 0
        sent_count = 0

        async with httpx.AsyncClient(timeout=120.0) as client:

            async def bounded_send(prompt: str, scheduled_wall_time: float):
                async with semaphore:
                    r = await self.send_request(client, prompt)
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

                    scheduled_wall_time = wall_start + next_arrival_elapsed
                    request_times.append(scheduled_wall_time)

                    tasks.append(asyncio.create_task(bounded_send(prompt, scheduled_wall_time)))

                    # Prevent unbounded growth of tasks waiting on the semaphore
                    if len(tasks) >= max_pending:
                        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        tasks = list(pending)
                else:
                    # Sleep until next ideal arrival (or a small slice)
                    sleep_for = max(0.0, next_arrival_elapsed - elapsed)
                    await asyncio.sleep(min(0.01, sleep_for))

            # Wait for all scheduled requests to finish
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=False)

        return results, request_times
    
    def start_model_server(
        self,
        memory_util: float,
        thread_percentage: int,
        startup_timeout_sec: int = 180,
        tensor_parallel_size: int = 2,
    ) -> subprocess.Popen:
        """Start model server with specified configuration."""
        # Stop any existing server on this port
        subprocess.run(
            f"pkill -f 'uvicorn.*port.*{self.port}' || true",
            shell=True,
            cwd=self.root_dir
        )
        time.sleep(2)
        
        # Start new server
        cmd = [
            "bash", "-c",
            f"cd {self.root_dir} && "
            f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={thread_percentage} "
            f"MODEL_NAME=\"{self.model_name}\" "
            f"TENSOR_PARALLEL_SIZE={tensor_parallel_size} "
            f"GPU_MEMORY_UTILIZATION={memory_util} "
            f"MAX_MODEL_LEN=2048 "
            f"UVICORN_PORT={self.port} "
            f"uvicorn servers.model_server:app --host 0.0.0.0 --port {self.port}"
        ]
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
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
        """Stop the model server."""
        if self.service_process:
            self.service_process.terminate()
            try:
                self.service_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.service_process.kill()
        
        subprocess.run(
            f"pkill -f 'uvicorn.*port.*{self.port}' || true",
            shell=True,
            cwd=self.root_dir
        )
        time.sleep(2)
    
    def compute_statistics(self, results: List[Dict], request_times: List[float]) -> Dict:
        """Compute performance statistics from results."""
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]
        
        if not successful:
            return {
                "success_rate": 0.0,
                "throughput_rps": 0.0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "total_requests": len(results),
                "failed_requests": len(failed),
            }
        
        latencies = [r["latency_ms"] for r in successful]
        latencies_sorted = sorted(latencies)
        
        # Compute throughput
        if len(request_times) > 1:
            time_span = request_times[-1] - request_times[0]
            throughput_rps = len(successful) / max(time_span, 0.1)
        else:
            throughput_rps = 0.0
        
        return {
            "success_rate": len(successful) / len(results),
            "throughput_rps": throughput_rps,
            "avg_latency_ms": np.mean(latencies),
            "p50_latency_ms": latencies_sorted[len(latencies_sorted) // 2],
            "p99_latency_ms": latencies_sorted[int(len(latencies_sorted) * 0.99)] if len(latencies_sorted) > 1 else latencies_sorted[0],
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies),
            "std_latency_ms": np.std(latencies),
            "total_requests": len(results),
            "failed_requests": len(failed),
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
    ) -> Dict:
        """Run a single experiment configuration."""
        print(f"\n{'='*80}")
        print(f"Testing: mem={memory_util:.2f}, threads={thread_percentage}%, load={load_rps:.1f} RPS")
        print(f"{'='*80}")
        
        try:
            # Start server
            self.service_process = self.start_model_server(
                memory_util,
                thread_percentage,
                startup_timeout_sec=startup_timeout_sec,
                tensor_parallel_size=tensor_parallel_size,
            )
            
            # Warmup
            print(f"  Warmup ({self.warmup_duration}s)...")
            await self.generate_load(load_rps, self.warmup_duration, prompts, max_concurrency=max_concurrency)
            await asyncio.sleep(2)
            
            # Collect metrics before test
            metrics_before = await self.fetch_metrics()
            
            # Run test
            print(f"  Running test ({self.test_duration}s)...")
            results, request_times = await self.generate_load(load_rps, self.test_duration, prompts, max_concurrency=max_concurrency)
            
            # Collect metrics after test
            await asyncio.sleep(1)
            metrics_after = await self.fetch_metrics()
            
            # Compute statistics
            stats = self.compute_statistics(results, request_times)
            
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
                },
                "performance": {
                    **stats,
                    "avg_queue_size": avg_queue,
                    "max_queue_size": metrics_after.get("in_flight_requests", 0),
                },
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
            return None
        finally:
            self.stop_model_server()
    
    def save_results(self, output_file: str):
        """Save collected results to JSON file."""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
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
                        help="TENSOR_PARALLEL_SIZE for vLLM (must be <= visible GPU count)")
    
    args = parser.parse_args()
    
    # Generate parameter grid
    memory_values = np.linspace(args.memory_range[0], args.memory_range[1], args.memory_steps)
    thread_values = np.linspace(args.thread_range[0], args.thread_range[1], args.thread_steps, dtype=int)
    load_values = np.linspace(args.load_range[0], args.load_range[1], args.load_steps)
    
    total_experiments = len(memory_values) * len(thread_values) * len(load_values)
    if args.max_experiments:
        total_experiments = min(total_experiments, args.max_experiments)
    
    print(f"\n{'='*80}")
    print(f"PERFORMANCE DATA COLLECTION")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Memory range: {args.memory_range[0]:.2f} - {args.memory_range[1]:.2f} ({args.memory_steps} steps)")
    print(f"Thread range: {args.thread_range[0]} - {args.thread_range[1]}% ({args.thread_steps} steps)")
    print(f"Load range: {args.load_range[0]:.1f} - {args.load_range[1]:.1f} RPS ({args.load_steps} steps)")
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
    
    # Run experiments
    experiment_count = 0
    for memory_util in memory_values:
        for thread_perc in thread_values:
            for load_rps in load_values:
                if args.max_experiments and experiment_count >= args.max_experiments:
                    break
                
                result = await collector.run_experiment(
                    memory_util=memory_util,
                    thread_percentage=thread_perc,
                    load_rps=load_rps,
                    prompts=prompts,
                    max_concurrency=args.max_concurrency,
                    startup_timeout_sec=args.startup_timeout_sec,
                    tensor_parallel_size=args.tensor_parallel_size,
                )
                
                if result:
                    collector.results.append(result)
                    experiment_count += 1
                    print(f"\nProgress: {experiment_count}/{total_experiments}")
            
            if args.max_experiments and experiment_count >= args.max_experiments:
                break
        if args.max_experiments and experiment_count >= args.max_experiments:
            break
    
    # Save results
    collector.save_results(args.output)
    
    print(f"\n✓ Data collection complete!")
    print(f"  Collected {len(collector.results)} experiments")
    print(f"  Next step: Train regression models with train_performance_models.py")


if __name__ == "__main__":
    asyncio.run(main())
