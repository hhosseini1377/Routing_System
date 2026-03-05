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
        self.success_count = 0
        self.failure_count = 0
        self.total_requests = 0
        self.in_flight_requests = 0
        self.start_time = time.time()
        self.request_times = deque(maxlen=window_size)  # for throughput calculation
        self.ttfts_ms = deque(maxlen=window_size)  # time to first token, ms
        self.tpot_ms_list: deque = deque(maxlen=window_size)  # TPOT (avg TBT per request), ms; one value per request
        self.token_counts = deque(maxlen=window_size)  # output tokens per request

    def record_success(
        self,
        latency_ms: float,
        ttft_ms: float,
        tbt_ms_list: Optional[list] = None,
        num_output_tokens: Optional[int] = None,
    ):
        self.latencies.append(latency_ms)
        self.ttfts_ms.append(ttft_ms)
        if tbt_ms_list:
            tpot_ms = sum(tbt_ms_list) / len(tbt_ms_list)
            self.tpot_ms_list.append(tpot_ms)
        if num_output_tokens is not None:
            self.token_counts.append(num_output_tokens)
        elif tbt_ms_list is not None:
            # Infer from TBT list: first token + one per TBT interval
            self.token_counts.append(len(tbt_ms_list) + 1)
        else:
            self.token_counts.append(0)
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
        if time_span <= 0:
            return 0.0
        return len(self.request_times) / time_span

    def get_token_throughput(self) -> float:
        """Output tokens per second in current window."""
        if not self.token_counts or not self.request_times:
            return 0.0
        time_span = self.request_times[-1] - self.request_times[0]
        if time_span <= 0:
            return 0.0
        total_tokens = sum(self.token_counts)
        return total_tokens / time_span
    
    def get_metrics_dict(self) -> dict:
        """Return metrics as dictionary."""
        success_rate = self.success_count / max(1, self.total_requests)
        # Latency metrics
        if self.latencies:
            latencies_sorted = sorted(self.latencies)
            p50_lat = latencies_sorted[len(latencies_sorted) // 2]
            p99_lat = latencies_sorted[int(len(latencies_sorted) * 0.99)]
            p99_lat = latencies_sorted[-1] if len(latencies_sorted) == 1 else p99_lat
            avg_lat = sum(self.latencies) / len(self.latencies)
        else:
            p50_lat = p99_lat = avg_lat = 0.0
        
        # TTFT metrics
        if self.ttfts_ms:
            ttfts_sorted = sorted(self.ttfts_ms)
            avg_ttft = sum(self.ttfts_ms) / len(self.ttfts_ms)
            p50_ttft = ttfts_sorted[len(ttfts_sorted) // 2]
            p99_ttft = ttfts_sorted[min(int(len(ttfts_sorted) * 0.99), len(ttfts_sorted) - 1)]
            p99_ttft = ttfts_sorted[-1] if len(ttfts_sorted) == 1 else p99_ttft
        else:
            avg_ttft = p50_ttft = p99_ttft = 0.0

        if self.tpot_ms_list:
            tpots_sorted = sorted(self.tpot_ms_list)
            avg_tpot = sum(self.tpot_ms_list) / len(self.tpot_ms_list)
            p50_tpot = tpots_sorted[len(tpots_sorted) // 2]
            p99_tpot = tpots_sorted[min(int(len(tpots_sorted) * 0.99), len(tpots_sorted) - 1)]
        else:
            avg_tpot = p50_tpot = p99_tpot = 0.0
        
        uptime_sec = time.time() - self.start_time
        
        total_tokens = sum(self.token_counts) if self.token_counts else 0

        return {
            "success_rate": success_rate,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_requests": self.total_requests,
            "throughput_rps": self.get_throughput_rps(),
            "token_throughput": self.get_token_throughput(),
            "total_output_tokens": total_tokens,
            "in_flight_requests": self.in_flight_requests,
            "avg_latency_ms": avg_lat,
            "p50_latency_ms": p50_lat,
            "p99_latency_ms": p99_lat,
            "avg_ttft_ms": avg_ttft,
            "p50_ttft_ms": p50_ttft,
            "p99_ttft_ms": p99_ttft,
            "avg_tpot_ms": avg_tpot,
            "p50_tpot_ms": p50_tpot,
            "p99_tpot_ms": p99_tpot,
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
    pipeline_parallel_size = int(os.environ.get("PIPELINE_PARALLEL_SIZE", "1"))
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "8192"))
    max_num_seqs_env = os.environ.get("MAX_NUM_SEQS")
    max_num_batched_tokens_env = os.environ.get("MAX_NUM_BATCHED_TOKENS")
    gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))

    engine_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if max_num_seqs_env:
        engine_kwargs["max_num_seqs"] = int(max_num_seqs_env)
    if max_num_batched_tokens_env:
        engine_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens_env)

    engine_args = AsyncEngineArgs(**engine_kwargs)
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
    start_time = req.start_time if req.start_time is not None else time.time()
    engine: AsyncLLMEngine = app.state.engine
    metrics: ModelMetrics = app.state.metrics

    # Track in-flight request
    metrics.in_flight_requests += 1
    
    try:
        results = engine.generate(
            req.prompt,
            sampling_params=sampling,
            request_id=request_id,
        )

        last_token_time = start_time
        time_between_tokens = []
        is_first_token = True
        TTFT = 0.0  # in case loop has zero iterations
        last_out = None
        disconnect_check_task = None

        async def check_disconnect():
            """Periodically check if client disconnected; abort if so."""
            while True:
                await asyncio.sleep(0.5)
                if await request.is_disconnected():
                    await engine.abort(request_id)
                    return

        try:
            disconnect_check_task = asyncio.create_task(check_disconnect())
            async for out in results:
                last_out = out
                if await request.is_disconnected():
                    await engine.abort(request_id)
                    metrics.record_failure()
                    return
                
                current_time = time.time()
                if is_first_token:
                    TTFT = current_time - start_time
                    is_first_token = False
                else:
                    time_between_tokens.append(current_time - last_token_time)
                
                last_token_time = current_time
                text = out.outputs[0].text if out.outputs else ""
                
            # Cancel disconnect checker
            if disconnect_check_task:
                disconnect_check_task.cancel()
                try:
                    await disconnect_check_task
                except asyncio.CancelledError:
                    pass

            # Check if request was aborted (client timeout/disconnect)
            was_aborted = (
                last_out is not None
                and last_out.finished
                and last_out.outputs
                and str(last_out.outputs[0].finish_reason) == "abort"
            )
            if was_aborted:
                metrics.record_failure()
                return JSONResponse(
                    content={"error": "aborted", "request_id": request_id},
                    status_code=499,
                )
                
            # Compute metrics for successful completion
            end_to_end_latency = time.time() - start_time
            tbt_ms_list = [t * 1000 for t in time_between_tokens]
            metrics.record_success(
                latency_ms=end_to_end_latency * 1000,
                ttft_ms=TTFT * 1000,
                tbt_ms_list=tbt_ms_list,
            )

            # Latency from start_time (client scheduled time) to completion
            latency_sec = time.time() - start_time
            tpot_ms = (sum(tbt_ms_list) / len(tbt_ms_list)) if tbt_ms_list else 0.0
            payload = {
                "request_id": request_id,
                "model": app.state.model_name,
                "response_text": text,
                "TTFT": TTFT,
                "tpot_ms": tpot_ms,
                "latency_sec": latency_sec,
            }

        except asyncio.CancelledError:
            if disconnect_check_task:
                disconnect_check_task.cancel()
                try:
                    await disconnect_check_task
                except asyncio.CancelledError:
                    pass
            await engine.abort(request_id)
            metrics.record_failure()
            raise
        except Exception:
            if disconnect_check_task:
                disconnect_check_task.cancel()
                try:
                    await disconnect_check_task
                except asyncio.CancelledError:
                    pass
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


@app.post("/clear")
async def clear():
    """Abort all in-flight requests and reset state. Use between loads to recover from timeouts."""
    engine: AsyncLLMEngine = app.state.engine
    try:
        # vLLM v1: pause_generation aborts in-flight, clear_cache=False avoids reset failures
        await engine.pause_generation(wait_for_inflight_requests=False, clear_cache=False)
        await engine.resume_generation()
        # Reinitialize metrics so next load starts with clean stats
        app.state.metrics = ModelMetrics()
        return JSONResponse({"cleared": True})
    except Exception as e:
        return JSONResponse({"cleared": False, "error": str(e)}, status_code=500)

@app.get("/metrics")
async def get_metrics():
    """Get current model server metrics."""
    metrics: ModelMetrics = app.state.metrics
    try:
        data = metrics.get_metrics_dict()
        # JSON cannot serialize NaN/Inf; replace with 0 to avoid 500
        def _json_safe(v):
            if isinstance(v, float) and (v != v or abs(v) == float('inf')):
                return 0.0
            return v
        data = {k: _json_safe(v) for k, v in data.items()}
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
