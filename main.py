import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs


def format_sse(data: str) -> str:
    # Minimal Server-Sent Events formatting
    return f"data: {data}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    # Create the engine once per process.
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-7B-Instruct",  # change me
        # tensor_parallel_size=1,
        # max_model_len=8192,
        # gpu_memory_utilization=0.9,
    )
    app.state.engine = AsyncLLMEngine.from_engine_args(engine_args)

    yield

    # --- shutdown ---
    # vLLM manages its internal background loop; on shutdown you mainly want to
    # stop accepting requests and let the process exit cleanly.
    # (If you keep long-running tasks, cancel them here.)
    # Some versions expose explicit shutdown methods; if yours does, call it.
    # Otherwise, just let process shutdown clean up resources.
    # app.state.engine.shutdown()  # only if available in your vLLM version


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(req: dict, request: Request):
    prompt: str = req["prompt"]
    sampling = SamplingParams(
        temperature=req.get("temperature", 0.7),
        max_tokens=req.get("max_tokens", 256),
    )

    engine: AsyncLLMEngine = app.state.engine
    request_id = req.get("request_id") or str(uuid.uuid4())

    # AsyncLLMEngine.generate returns an async generator of outputs. :contentReference[oaicite:1]{index=1}
    results = engine.generate(prompt, sampling_params=sampling, request_id=request_id)

    async def stream():
        try:
            async for out in results:
                # If client disconnects, abort the generation. :contentReference[oaicite:2]{index=2}
                if await request.is_disconnected():
                    await engine.abort(request_id)
                    return

                # vLLM output objects contain token/text info; common pattern:
                # out.outputs[-1].text (depends on vLLM version / API)
                text = out.outputs[0].text if out.outputs else ""
                yield format_sse(text)

            yield format_sse("[DONE]")

        except asyncio.CancelledError:
            # If FastAPI cancels the handler (client dropped / server shutdown)
            # make sure we abort the engine request.
            await engine.abort(request_id)
            raise

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/abort")
async def abort(req: dict):
    request_id = req["request_id"]
    engine: AsyncLLMEngine = app.state.engine
    await engine.abort(request_id)
    return JSONResponse({"aborted": True, "request_id": request_id})
