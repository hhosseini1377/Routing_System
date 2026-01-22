import asyncio
import json
import os
import random
import uuid
from typing import Dict, Optional, List, Tuple

from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from router_model.regression_models import TruncatedModel, load_tokenizer
from router_model.config import RouterModelConfig
from config import RouterConfig
import time

def format_sse(data: str) -> str:
    return f"data: {data}\n\n"


class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 256
    request_id: Optional[str] = None


MODEL_A_URL = os.environ.get("MODEL_A_URL", "http://127.0.0.1:8001")
MODEL_B_URL = os.environ.get("MODEL_B_URL", "http://127.0.0.1:8002")
threshold = 0.5
ROUTER_MODEL_PATH = os.environ.get("ROUTER_MODEL_PATH", "router_model/model_checkpoints/model_deberta_20260101-163459.pth")

# Batching configuration
BATCH_SIZE = int(os.environ.get("ROUTER_BATCH_SIZE", "4"))  # Process up to 4 requests at once
BATCH_TIMEOUT_MS = int(os.environ.get("ROUTER_BATCH_TIMEOUT_MS", "20"))  # Max 20ms wait for batch

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _batch_processor_task

    router_model_config = RouterModelConfig()
    print("Starting up the router...")
    app.state.router_model = TruncatedModel.load_model_from_checkpoint(
        model_path=ROUTER_MODEL_PATH,
        model_name="deberta",
        pooling_strategy="cls",
        num_outputs=1,
        num_classes=2,
        router_model_config=router_model_config
    )
    router_config = RouterConfig()
    app.state.tokenizer = load_tokenizer(model_name=router_config.model_name)
    
    # Start batch processor task
    _batch_processor_task = asyncio.create_task(batch_processor())
    print(f"Router batching enabled: batch_size={BATCH_SIZE}, timeout_ms={BATCH_TIMEOUT_MS}")
    
    yield
    
    print("Shutting down the router...")
    # Cancel batch processor
    if _batch_processor_task:
        _batch_processor_task.cancel()
        try:
            await _batch_processor_task
        except asyncio.CancelledError:
            pass

app = FastAPI(lifespan=lifespan)
_request_routes: Dict[str, str] = {}
_request_lock = asyncio.Lock()

# Batching infrastructure
_batch_queue: asyncio.Queue = asyncio.Queue()
_batch_processor_task: Optional[asyncio.Task] = None


def choose_backend_batched(prompts: List[str]) -> List[str]:
    """Process a batch of prompts and return backends."""
    import torch
    
    router_model = app.state.router_model
    tokenizer = app.state.tokenizer
    device = next(router_model.parameters()).device
    
    # Tokenize all prompts
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    # Process batch
    with torch.no_grad():
        outputs = router_model(input_ids, attention_mask)
    
    # Extract scores (handle both 1D and 2D outputs)
    if outputs.dim() == 1:
        scores = outputs.cpu().numpy()
    else:
        scores = outputs.squeeze(-1).cpu().numpy()
    
    # Ensure scores is at least 1D
    import numpy as np
    scores = np.atleast_1d(scores)
    
    # Convert scores to backends
    backends = [
        MODEL_A_URL if score > threshold else MODEL_B_URL 
        for score in scores
    ]
    
    return backends


async def batch_processor():
    """Background task that processes batches from the queue."""
    while True:
        batch_items: List[Tuple[str, asyncio.Future]] = []
        
        try:
            # Wait for first item (no timeout - block until at least one arrives)
            prompt, future = await _batch_queue.get()
            batch_items.append((prompt, future))
            
            # Try to collect more items up to BATCH_SIZE (with timeout for batching)
            # Use asyncio.wait_for to wait up to BATCH_TIMEOUT_MS for additional items
            try:
                async def collect_additional_items():
                    while len(batch_items) < BATCH_SIZE:
                        prompt, future = await _batch_queue.get()
                        batch_items.append((prompt, future))
                
                await asyncio.wait_for(collect_additional_items(), timeout=BATCH_TIMEOUT_MS / 1000.0)
            except asyncio.TimeoutError:
                # Timeout expired, process what we have
                pass
        except Exception as e:
            # Handle unexpected errors in queue operations
            print(f"Error in batch processor queue: {e}")
            continue
        
        # Process the batch
        try:
            prompts = [item[0] for item in batch_items]
            backends = choose_backend_batched(prompts)
            
            # Set results for all futures
            for (_, future), backend in zip(batch_items, backends):
                if not future.done():
                    future.set_result(backend)
        except Exception as e:
            # Set exception for all futures if batch processing fails
            print(f"Error processing batch: {e}")
            for _, future in batch_items:
                if not future.done():
                    future.set_exception(e)


async def choose_backend(prompt: str) -> str:
    """Choose backend for a prompt using batched inference."""
    # Create a future for this request
    future = asyncio.Future()
    
    # Add to batch queue
    await _batch_queue.put((prompt, future))
    
    # Wait for result
    return await future

@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    request_id = req.request_id or str(uuid.uuid4())
    payload = req.model_dump()
    payload["request_id"] = request_id
    payload["start_time"] = time.time()
    backend = await choose_backend(payload["prompt"])
    time_to_choose_backend = time.time() - payload["start_time"]
    async with _request_lock:
        _request_routes[request_id] = backend

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(f"{backend}/generate", json=payload)
            resp.raise_for_status()
            resp_data = resp.json()

            response_text = resp_data.get("response_text", "")
            ttft = resp_data.get("TTFT", 0.0)
            avg_time_between_tokens = resp_data.get("avg_time_between_tokens", 0.0)
            
            # Add the backend to the response text and return a json response
            return JSONResponse(content={"backend": backend, "response_text": response_text, "time_to_choose_backend": time_to_choose_backend, "TTFT": ttft, "avg_time_between_tokens": avg_time_between_tokens})
    finally:
        async with _request_lock:
            _request_routes.pop(request_id, None)

@app.post("/abort")
async def abort(req: dict):
    request_id = req.get("request_id")
    if not request_id:
        return JSONResponse({"aborted": False, "error": "request_id required"}, status_code=400)

    aborted = await abort_request(request_id)
    status = 200 if aborted else 404
    return JSONResponse({"aborted": aborted, "request_id": request_id}, status_code=status)


async def abort_request(request_id: str) -> bool:
    async with _request_lock:
        backend = _request_routes.get(request_id)

    if not backend:
        return False

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{backend}/abort", json={"request_id": request_id})
        return resp.status_code == 200
