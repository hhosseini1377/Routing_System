#!/usr/bin/env python3
"""
Script to send 100 requests to the router endpoint.
"""

import asyncio
import json
import time
from typing import Dict, List
import httpx


ROUTER_URL = "http://127.0.0.1:8000/generate"
NUM_REQUESTS = 1000
PROMPTS_PATH = "datasets/lmsys_chat1m_prompts_100k_cleaned.pkl"

async def send_single_request(
    client: httpx.AsyncClient,
    request_id: int,
    prompt: str = "What is machine learning?",
    temperature: float = 0.7,
    max_tokens: int = 256,
    start_time: float = None,
) -> Dict:
    """Send a single request and process the JSON response."""
    
    start_time = start_time or time.time()

    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        response = await client.post(ROUTER_URL, json=payload, timeout=60.0)
        response.raise_for_status()

        response_data = response.json()
        backend = response_data.get("backend", "")
        response_text = response_data.get("response_text", "")
        time_to_choose_backend = response_data.get("time_to_choose_backend", 0.0)
        ttft = response_data.get("TTFT", 0.0)
        avg_time_between_tokens = response_data.get("avg_time_between_tokens", 0.0)
        elapsed_time = time.time() - start_time
        
        return {
            "request_id": request_id,
            "success": True,
            "elapsed_time": elapsed_time,
            "backend": backend,
            "response_length": len(response_text),
            "response_text": response_text,
            "status_code": response.status_code,
            "TTFT": ttft,
            "avg_time_between_tokens": avg_time_between_tokens,
            "time_to_choose_backend": time_to_choose_backend,
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "request_id": request_id,
            "success": False,
            "elapsed_time": elapsed_time,
            "error": str(e),
        }


async def send_requests_concurrent(
    num_requests: int = NUM_REQUESTS,
    concurrent_limit: int = 100,
    prompts: List[str] = None,
) -> List[Dict]:
    """Send multiple requests with concurrency limit."""
    results = []
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def bounded_request(request_id: int):
            async with semaphore:
                return await send_single_request(client, request_id, prompt=prompts[request_id] if prompts else "What is machine learning?", start_time=start_time)
        
        # Create all tasks
        tasks = [bounded_request(i) for i in range(num_requests)]
        
        # Execute with progress tracking
        print(f"Sending {num_requests} requests (max {concurrent_limit} concurrent)...")
        start_time = time.time()
        
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            result = await task
            results.append(result)
            if i % 10 == 0 or i == num_requests:
                print(f"Completed {i}/{num_requests} requests...")
        
        total_time = time.time() - start_time
        print(f"\nAll requests completed in {total_time:.2f} seconds")
    
    return results


def send_requests_sequential(
    num_requests: int = NUM_REQUESTS,
    prompt: str = "What is machine learning?",
) -> List[Dict]:
    """Send requests sequentially (simpler, but slower)."""
    results = []
    
    with httpx.Client(timeout=120.0) as client:
        print(f"Sending {num_requests} requests sequentially...")
        start_time = time.time()
        
        for i in range(num_requests):
            payload = {
                "prompt": prompt,
                "temperature": 0.7,
                "max_tokens": 256,
            }
            
            request_start = time.time()
            try:
                response = client.post(ROUTER_URL, json=payload, timeout=60.0)
                response.raise_for_status()
                
                response_data = response.json()
                print(response_data)
                backend = response_data.get("backend", "")
                response_text = response_data.get("response", "")
                time_to_choose_backend = response_data.get("time_to_choose_backend", 0.0)
                ttft = response_data.get("TTFT", 0.0)
                avg_time_between_tokens = response_data.get("avg_time_between_tokens", 0.0)
                elapsed = time.time() - request_start
                results.append({
                    "request_id": i,
                    "success": True,
                    "elapsed_time": elapsed,
                    "backend": backend,
                    "response_length": len(response_text),
                    "response_text": response_text,
                    "status_code": response.status_code,
                    "TTFT": ttft,
                    "avg_time_between_tokens": avg_time_between_tokens,
                    "time_to_choose_backend": time_to_choose_backend,
                })
            except Exception as e:
                elapsed = time.time() - request_start
                results.append({
                    "request_id": i,
                    "success": False,
                    "elapsed_time": elapsed,
                    "error": str(e),
                })
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_requests} requests...")
        
        total_time = time.time() - start_time
        print(f"\nAll requests completed in {total_time:.2f} seconds")
    
    return results


def print_statistics(results: List[Dict]):
    """Print statistics about the requests."""
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Total requests: {len(results)}")
    print(f"Successful: {len(successful)} ({100*len(successful)/len(results):.1f}%)")
    print(f"Failed: {len(failed)} ({100*len(failed)/len(results):.1f}%)")
    
    unique_backends = list(set([r["backend"] for r in successful]))
    for backend in unique_backends:
        times = [r["elapsed_time"] for r in successful if r["backend"] == backend]
        print(f"count for {backend}: {len(times)}")
        print(f"\nResponse Times for {backend}:")
        print(f"  Min: {min(times):.2f}")
        print(f"  Max: {max(times):.2f}")
        print(f"  Avg: {sum(times)/len(times):.2f}")
        print(f"  Median: {sorted(times)[len(times)//2]:.2f}")

    if successful:
        times = [r["elapsed_time"] for r in successful]
        print(f"\nResponse Times (seconds):")
        print(f"  Min: {min(times):.2f}")
        print(f"  Max: {max(times):.2f}")
        print(f"  Avg: {sum(times)/len(times):.2f}")
        print(f"  Median: {sorted(times)[len(times)//2]:.2f}")
    
    if failed:
        print(f"\nErrors encountered:")
        error_counts = {}
        for r in failed:
            error = r.get("error", "Unknown error")
            error_counts[error] = error_counts.get(error, 0) + 1
        for error, count in error_counts.items():
            print(f"  {error}: {count}")


async def main():
    """Main function."""
    import sys
    
    # Check command line arguments
    mode = "concurrent"
    num_requests = NUM_REQUESTS
    concurrent_limit = 10

    with open(PROMPTS_PATH, 'rb') as f:
        import pickle
        data = pickle.load(f)
    prompts = data[:num_requests]
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "sequential":
            mode = "sequential"
        elif sys.argv[1] == "concurrent":
            mode = "concurrent"
        else:
            num_requests = int(sys.argv[1])
    
    if len(sys.argv) > 2 and mode == "concurrent":
        concurrent_limit = int(sys.argv[2])
    
    print(f"Mode: {mode}")
    print(f"Number of requests: {num_requests}")
    if mode == "concurrent":
        print(f"Concurrent limit: {concurrent_limit}")
    print()
    
    if mode == "concurrent":
        results = await send_requests_concurrent(num_requests=num_requests, concurrent_limit=concurrent_limit, prompts=prompts)
    else:
        results = send_requests_sequential(num_requests=num_requests)

    print_statistics(results)
    
    # rewrite results to file and append to the file
    output_file = "request_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults appended to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())