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
NUM_REQUESTS = 100


async def send_single_request(
    client: httpx.AsyncClient,
    request_id: int,
    prompt: str = "What is machine learning?",
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> Dict:
    """Send a single request and process the SSE response."""
    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    start_time = time.time()
    try:
        async with client.stream("POST", ROUTER_URL, json=payload, timeout=60.0) as response:
            response.raise_for_status()
            
            full_text = ""
            last_event = None
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    try:
                        event_data = json.loads(data_str)
                        last_event = event_data
                        if "text" in event_data:
                            full_text += event_data["text"]
                    except json.JSONDecodeError:
                        pass
            
            elapsed_time = time.time() - start_time
            
            return {
                "request_id": request_id,
                "success": True,
                "elapsed_time": elapsed_time,
                "text_length": len(full_text),
                "last_event": last_event,
                "status_code": response.status_code,
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
    concurrent_limit: int = 10,
    prompt: str = "What is machine learning?",
) -> List[Dict]:
    """Send multiple requests with concurrency limit."""
    results = []
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def bounded_request(request_id: int):
            async with semaphore:
                return await send_single_request(client, request_id, prompt=prompt)
        
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
                with client.stream("POST", ROUTER_URL, json=payload, timeout=60.0) as response:
                    response.raise_for_status()
                    
                    full_text = ""
                    last_event = None
                    
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                event_data = json.loads(data_str)
                                last_event = event_data
                                if "text" in event_data:
                                    full_text += event_data["text"]
                            except json.JSONDecodeError:
                                pass
                    
                    elapsed = time.time() - request_start
                    results.append({
                        "request_id": i,
                        "success": True,
                        "elapsed_time": elapsed,
                        "text_length": len(full_text),
                        "last_event": last_event,
                        "status_code": response.status_code,
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
        results = await send_requests_concurrent(num_requests, concurrent_limit)
    else:
        results = send_requests_sequential(num_requests)
    
    print_statistics(results)
    
    # Save results to file
    output_file = "request_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
