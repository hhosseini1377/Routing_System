import asyncio
import json
import os
import random
import uuid
from typing import AsyncGenerator, Dict, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel


def format_sse(data: str) -> str:
    return f"data: {data}\n\n"


class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 256
    request_id: Optional[str] = None


MODEL_A_URL = os.environ.get("MODEL_A_URL", "http://127.0.0.1:8001")
MODEL_B_URL = os.environ.get("MODEL_B_URL", "http://127.0.0.1:8002")


app = FastAPI()
_request_routes: Dict[str, str] = {}
_request_lock = asyncio.Lock()


def choose_backend() -> str:
    return random.choice([MODEL_A_URL, MODEL_B_URL])


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    request_id = req.request_id or str(uuid.uuid4())
    backend = choose_backend()

    async with _request_lock:
        _request_routes[request_id] = backend

    payload = req.model_dump()
    payload["request_id"] = request_id

    async def stream() -> AsyncGenerator[str, None]:
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{backend}/generate", json=payload) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_text():
                        if await request.is_disconnected():
                            await abort_request(request_id)
                            return
                        if chunk:
                            yield chunk
        finally:
            async with _request_lock:
                _request_routes.pop(request_id, None)

    return StreamingResponse(stream(), media_type="text/event-stream")


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
