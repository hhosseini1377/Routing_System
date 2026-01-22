import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
import time

def format_sse(data: str) -> str:
    return f"data: {data}\n\n"


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
                return
            
            current_time = time.time()
            if is_first_token:
                TTFT = current_time - req.start_time
                is_first_token = False
            else:
                time_between_tokens.append(current_time - last_token_time)
            
            last_token_time = current_time
            text = out.outputs[0].text if out.outputs else ""
            
        payload = {
            "request_id": request_id,
            "model": app.state.model_name,
            "response_text": text,
            "TTFT": TTFT,
            "avg_time_between_tokens": sum(time_between_tokens) / len(time_between_tokens) if time_between_tokens else 0.0
        }
    except asyncio.CancelledError:
        await engine.abort(request_id)
        raise

    return JSONResponse(content=payload)

@app.post("/abort")
async def abort(req: dict):
    request_id = req.get("request_id")
    if not request_id:
        return JSONResponse({"aborted": False, "error": "request_id required"}, status_code=400)

    engine: AsyncLLMEngine = app.state.engine
    await engine.abort(request_id)
    return JSONResponse({"aborted": True, "request_id": request_id})
