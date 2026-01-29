import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from collections import deque
from statistics import median

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
import time

def format_sse(data: str) -> str:
    return f"data: {data}\n\n"


class ModelMetrics:
    """Track per-model performance metrics."""
    
    def __init__(self, window_size: int = 300):  # 5-minute window
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)  # end-to-end latencies
        self.ttfts = deque(maxlen=window_size)  # time-to-first-token
        self.success_count = 0
        self.failure_count = 0
        self.total_requests = 0
        self.in_flight_requests = 0
        self.start_time = time.time()
        self.request_times = deque(maxlen=window_size)  # for throughput calculation
        
    def record_success(self, latency_ms: float, ttft_ms: float):
        self.latencies.append(latency_ms)
        self.ttfts.append(ttft_ms)
        self.success_count += 1
        self.total_requests += 1
        self.request_times.append(time.time())
    
    def record_failure(self):
        self.failure_count += 1
        self.total_requests += 1
    
    def get_throughput_rps(self) -> float:
        """Requests per second in current window."""
        if not self.request_times:
            return 0.0
        time_span = self.request_times[-1] - self.request_times[0]
        if time_span < 1:
            return float(len(self.request_times))
        return len(self.request_times) / time_span
    
    def get_metrics_dict(self) -> dict:
        """Return metrics as dictionary."""
        success_rate = self.success_count / max(1, self.total_requests)
        
        if self.latencies:
            latencies_sorted = sorted(self.latencies)
            p50_lat = latencies_sorted[len(latencies_sorted) // 2]
            p99_lat = latencies_sorted[int(len(latencies_sorted) * 0.99)]
            p99_lat = latencies_sorted[-1] if len(latencies_sorted) == 1 else p99_lat
            avg_lat = sum(self.latencies) / len(self.latencies)
        else:
            p50_lat = p99_lat = avg_lat = 0.0
        
        if self.ttfts:
            avg_ttft = sum(self.ttfts) / len(self.ttfts)
        else:
            avg_ttft = 0.0
        
        uptime_sec = time.time() - self.start_time
        
        return {
            "success_rate": success_rate,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_requests": self.total_requests,
            "throughput_rps": self.get_throughput_rps(),
            "in_flight_requests": self.in_flight_requests,
            "avg_latency_ms": avg_lat,
            "p50_latency_ms": p50_lat,
            "p99_latency_ms": p99_lat,
            "avg_ttft_ms": avg_ttft,
            "uptime_sec": uptime_sec,
            "timestamp": time.time()
        }


class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 256
    request_id: Optional[str] = None
    start_time: Optional[float] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up the model server...")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-1.8B")
    tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "8192"))
    gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))

    engine_args = AsyncEngineArgs(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    try:
        app.state.engine = AsyncLLMEngine.from_engine_args(engine_args)
    except Exception as e:
        print(f"Failed to initialize the model engine: {e}")
        raise e
    app.state.model_name = model_name
    app.state.metrics = ModelMetrics()
    app.state.gpu_memory_utilization = gpu_memory_utilization
    app.state.tensor_parallel_size = tensor_parallel_size

    yield
    
    print("Shutting down the model server...")
    try:
        await app.state.engine.stop()
        print("Model engine stopped successfully")
    except Exception as e:
        print(f"Error stopping the engine: {e}")
    
app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    request_id = req.request_id or str(uuid.uuid4())
    sampling = SamplingParams(temperature=req.temperature, max_tokens=req.max_tokens)
    start_time = req.start_time 
    engine: AsyncLLMEngine = app.state.engine
    metrics: ModelMetrics = app.state.metrics

    # Track in-flight request
    metrics.in_flight_requests += 1
    request_start = time.time()
    
    try:
        results = engine.generate(
            req.prompt,
            sampling_params=sampling,
            request_id=request_id,
        )

        last_token_time = start_time
        time_between_tokens = []
        is_first_token = True
        try:
            async for out in results:
                if await request.is_disconnected():
                    await engine.abort(request_id)
                    metrics.record_failure()
                    return
                
                current_time = time.time()
                if is_first_token:
                    TTFT = current_time - req.start_time
                    is_first_token = False
                else:
                    time_between_tokens.append(current_time - last_token_time)
                
                last_token_time = current_time
                text = out.outputs[0].text if out.outputs else ""
                
            # Compute metrics
            end_to_end_latency = time.time() - start_time
            metrics.record_success(
                latency_ms=end_to_end_latency * 1000,
                ttft_ms = TTFT * 1000
            )
            
            payload = {
                "request_id": request_id,
                "model": app.state.model_name,
                "response_text": text,
                "TTFT": TTFT,
                "avg_time_between_tokens": sum(time_between_tokens) / len(time_between_tokens) if time_between_tokens else 0.0
            }

        except asyncio.CancelledError:
            await engine.abort(request_id)
            metrics.record_failure()
            raise
    finally:
        metrics.in_flight_requests -= 1

    return JSONResponse(content=payload)

@app.post("/abort")
async def abort(req: dict):
    request_id = req.get("request_id")
    if not request_id:
        return JSONResponse({"aborted": False, "error": "request_id required"}, status_code=400)

    engine: AsyncLLMEngine = app.state.engine
    await engine.abort(request_id)
    return JSONResponse({"aborted": True, "request_id": request_id})

@app.get("/metrics")
async def get_metrics():
    """Get current model server metrics."""
    metrics: ModelMetrics = app.state.metrics
    return JSONResponse(content=metrics.get_metrics_dict())