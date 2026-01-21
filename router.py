import asyncio
import json
import os
import random
import uuid
from typing import Dict, Optional

from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from router_model.regression_models import TruncatedModel, load_tokenizer
from router_model.config import RouterModelConfig
from config import RouterConfig

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

@asynccontextmanager
async def lifespan(app: FastAPI):

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
    yield
    print("Shutting down the router...")

app = FastAPI(lifespan=lifespan)
_request_routes: Dict[str, str] = {}
_request_lock = asyncio.Lock()


def choose_backend(prompt: str) -> str:
    import torch
    
    router_model = app.state.router_model
    device = next(router_model.parameters()).device
    
    input_ids = app.state.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = (input_ids != app.state.tokenizer.pad_token_id).to(device)
    
    with torch.no_grad():
        outputs = router_model(input_ids, attention_mask)
    
    score = outputs.item() if outputs.dim() == 0 else outputs.squeeze().item()
    if score > threshold:
        return MODEL_A_URL
    else:
        return MODEL_B_URL

@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    request_id = req.request_id or str(uuid.uuid4())
    payload = req.model_dump()
    payload["request_id"] = request_id
    backend = choose_backend(payload["prompt"])

    async with _request_lock:
        _request_routes[request_id] = backend


    try:
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(f"{backend}/generate", json=payload)
            resp.raise_for_status()
            response_text = resp.text
            # Add the backend to the response text and return a json response
            return JSONResponse(content={"backend": backend, "response": response_text})
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
