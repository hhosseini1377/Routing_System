import asyncio
import json
import os
import random
import uuid
from typing import Dict, Optional, List
from collections import deque

import numpy as np
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from router_model.regression_models import TruncatedModel, load_tokenizer
from router_model.config import RouterModelConfig
from config import RouterConfig
from .backend_selector import BackendSelector
import time

def format_sse(data: str) -> str:
    return f"data: {data}\n\n"


def _meets_slo(
    success: bool,
    ttft_ms: Optional[float],
    tbt_list_ms: List[float],
    slo_ttft_ms: float,
    slo_tbt_p95_ms: float,
) -> bool:
    """Request meets SLO if: success AND TTFT < threshold AND p95(TBTs) < threshold."""
    if not success:
        return False
    if ttft_ms is None or ttft_ms >= slo_ttft_ms:
        return False
    if tbt_list_ms:
        p95_tbt = float(np.percentile(tbt_list_ms, 95))
        if p95_tbt >= slo_tbt_p95_ms:
            return False
    return True


class RouterMetrics:
    """Track system-level goodput metrics.

    Goodput = throughput * SLO attainment rate.
    SLO: request is good if TTFT < slo_ttft_ms AND p95(TBTs) < slo_tbt_p95_ms.
    """

    def __init__(
        self,
        window_size: int = 1000,
        slo_ttft_ms: float = 500.0,
        slo_tbt_p95_ms: float = 100.0,
    ):
        self.window_size = window_size
        self.slo_ttft_ms = slo_ttft_ms
        self.slo_tbt_p95_ms = slo_tbt_p95_ms
        self.request_latencies = deque(maxlen=window_size)  # end-to-end latencies
        self.request_backends = deque(maxlen=window_size)  # which backend was chosen
        self.request_outcomes = deque(maxlen=window_size)  # success/failure
        self.request_times = deque(maxlen=window_size)  # timestamps
        self.ttfts_ms = deque(maxlen=window_size)  # TTFT in ms per request (None if failed)
        self.tbt_lists_ms = deque(maxlen=window_size)  # list of TBT lists per request
        self.start_time = time.time()

    def record_request(
        self,
        latency_ms: float,
        backend: str,
        success: bool,
        ttft_ms: Optional[float] = None,
        time_between_tokens_ms: Optional[List[float]] = None,
    ):
        self.request_latencies.append(latency_ms)
        self.request_backends.append(backend)
        self.request_outcomes.append(success)
        self.request_times.append(time.time())
        self.ttfts_ms.append(ttft_ms if success else None)
        self.tbt_lists_ms.append(time_between_tokens_ms if success else [])

    def get_goodput(self) -> float:
        """Goodput = throughput * SLO attainment rate (SLO-compliant RPS)."""
        if not self.request_outcomes:
            return 0.0
        slo_compliant = sum(
            _meets_slo(
                outcome,
                ttft,
                list(tbts) if tbts else [],
                self.slo_ttft_ms,
                self.slo_tbt_p95_ms,
            )
            for outcome, ttft, tbts in zip(
                self.request_outcomes, self.ttfts_ms, self.tbt_lists_ms
            )
        )
        slo_attainment_rate = slo_compliant / len(self.request_outcomes)
        if self.request_times:
            time_span = self.request_times[-1] - self.request_times[0]
            throughput_rps = len(self.request_outcomes) / max(1, time_span)
        else:
            throughput_rps = 0.0
        return throughput_rps * slo_attainment_rate
    
    def get_metrics_dict(self, detail: str = "full") -> dict:
        """Return aggregated system metrics.

        detail: "full" returns ttfts_ms, tbt_lists_ms, chosen_backends as lists;
                "summary" returns avg_ttft_ms, avg_tbt_ms, backend_counts instead.
        """
        if not self.request_outcomes:
            empty = {
                "goodput": 0.0,
                "total_requests": 0,
                "success_rate": 0.0,
                "slo_compliant_count": 0,
                "slo_attainment_rate": 0.0,
                "throughput_rps": 0.0,
                "avg_latency_ms": 0.0,
                "slo_ttft_ms": self.slo_ttft_ms,
                "slo_tbt_p95_ms": self.slo_tbt_p95_ms,
                "timestamp": time.time(),
            }
            if detail == "full":
                empty["ttfts_ms"] = []
                empty["tbt_lists_ms"] = []
                empty["chosen_backends"] = []
            else:
                empty["avg_ttft_ms"] = 0.0
                empty["avg_tbt_ms"] = 0.0
                empty["backend_counts"] = {}
            return empty

        success_rate = sum(self.request_outcomes) / len(self.request_outcomes)
        slo_compliant = sum(
            _meets_slo(
                outcome,
                ttft,
                list(tbts) if tbts else [],
                self.slo_ttft_ms,
                self.slo_tbt_p95_ms,
            )
            for outcome, ttft, tbts in zip(
                self.request_outcomes, self.ttfts_ms, self.tbt_lists_ms
            )
        )
        slo_attainment_rate = slo_compliant / len(self.request_outcomes)
        avg_latency = sum(self.request_latencies) / len(self.request_latencies)

        # Count by backend
        backend_a_count = sum(1 for b in self.request_backends if "8001" in b)
        backend_b_count = sum(1 for b in self.request_backends if "8002" in b)

        if self.request_times:
            time_span = self.request_times[-1] - self.request_times[0]
            throughput_rps = len(self.request_outcomes) / max(1, time_span)
        else:
            throughput_rps = 0.0

        goodput = throughput_rps * slo_attainment_rate

        result = {
            "goodput": goodput,
            "total_requests": len(self.request_outcomes),
            "success_rate": success_rate,
            "slo_compliant_count": slo_compliant,
            "slo_attainment_rate": slo_attainment_rate,
            "throughput_rps": throughput_rps,
            "slo_ttft_ms": self.slo_ttft_ms,
            "slo_tbt_p95_ms": self.slo_tbt_p95_ms,
            "avg_latency_ms": avg_latency,
            "backend_a_routed": backend_a_count,
            "backend_b_routed": backend_b_count,
            "uptime_sec": time.time() - self.start_time,
            "timestamp": time.time(),
        }

        if detail == "full":
            result["ttfts_ms"] = list(self.ttfts_ms)
            result["tbt_lists_ms"] = [list(tbts) for tbts in self.tbt_lists_ms]
            result["chosen_backends"] = list(self.request_backends)
        else:
            ttfts_valid = [t for t in self.ttfts_ms if t is not None]
            result["avg_ttft_ms"] = sum(ttfts_valid) / len(ttfts_valid) if ttfts_valid else 0.0
            all_tbts = [tbt for tbts in self.tbt_lists_ms for tbt in tbts]
            result["avg_tbt_ms"] = sum(all_tbts) / len(all_tbts) if all_tbts else 0.0
            result["backend_counts"] = dict(
                (b, sum(1 for x in self.request_backends if x == b))
                for b in set(self.request_backends)
            )

        return result

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 256
    request_id: Optional[str] = None
    start_time: Optional[float] = None  # client scheduled_wall_time for latency/TTFT/TBT


class BatchConfigUpdate(BaseModel):
    """Optional fields; only provided fields are updated."""
    batch_size: Optional[int] = None
    batch_timeout_ms: Optional[int] = None


MODEL_A_URL = os.environ.get("MODEL_A_URL", "http://127.0.0.1:8001")
MODEL_B_URL = os.environ.get("MODEL_B_URL", "http://127.0.0.1:8002")
threshold = 0.5
ROUTER_MODEL_PATH = os.environ.get("ROUTER_MODEL_PATH", "router_model/model_checkpoints/model_deberta_20260101-163459.pth")

# Batching configuration
BATCH_SIZE = int(os.environ.get("ROUTER_BATCH_SIZE", "4"))  # Process up to 4 requests at once
BATCH_TIMEOUT_MS = int(os.environ.get("ROUTER_BATCH_TIMEOUT_MS", "20"))  # Max 20ms wait for batch

# Multi-router: comma-separated devices, e.g. "cuda:0,cuda:1". Prompts are randomly routed to one.
ROUTER_MODEL_DEVICES = [d.strip() for d in os.environ.get("ROUTER_MODEL_DEVICES", "cuda:0").split(",") if d.strip()]
if not ROUTER_MODEL_DEVICES:
    ROUTER_MODEL_DEVICES = ["cuda:0"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    router_model_config = RouterModelConfig()
    router_config = RouterConfig()
    tokenizer = load_tokenizer(model_name=router_config.model_name)
    app.state.metrics = RouterMetrics(
        slo_ttft_ms=float(os.environ.get("SLO_TTFT_MS", "500")),
        slo_tbt_p95_ms=float(os.environ.get("SLO_TBT_P95_MS", "100")),
    )
    app.state.on_the_fly_requests = 0

    backend_selectors: List[BackendSelector] = []
    for i, device in enumerate(ROUTER_MODEL_DEVICES):
        print(f"Loading router model {i + 1}/{len(ROUTER_MODEL_DEVICES)} on {device}...")
        router_model = TruncatedModel.load_model_from_checkpoint(
            model_path=ROUTER_MODEL_PATH,
            model_name="deberta",
            pooling_strategy="cls",
            num_outputs=1,
            num_classes=2,
            router_model_config=router_model_config,
            device=device,
        )
        sel = BackendSelector(
            model=router_model,
            tokenizer=tokenizer,
            model_a_url=MODEL_A_URL,
            model_b_url=MODEL_B_URL,
            threshold=threshold,
            batch_size=BATCH_SIZE,
            batch_timeout_ms=BATCH_TIMEOUT_MS,
        )
        await sel.start()
        backend_selectors.append(sel)

    app.state.backend_selectors = backend_selectors
    print(f"Router: {len(backend_selectors)} model(s) on devices {ROUTER_MODEL_DEVICES}, batch_size={BATCH_SIZE}, timeout_ms={BATCH_TIMEOUT_MS}")

    yield

    print("Shutting down the router...")
    for sel in backend_selectors:
        await sel.stop()

app = FastAPI(lifespan=lifespan)
_request_routes: Dict[str, str] = {}
_request_lock = asyncio.Lock()


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    request_id = req.request_id or str(uuid.uuid4())
    # Use client-provided start_time (scheduled_wall_time) for latency/TTFT/TBT; fallback to now
    request_start = req.start_time if req.start_time is not None else time.time()
    payload = req.model_dump(exclude={"start_time"})
    payload["request_id"] = request_id
    payload["start_time"] = request_start
    loop = asyncio.get_running_loop()
    router_start_time = loop.time()
    selector = random.choice(app.state.backend_selectors)
    backend = await selector.choose_backend(payload["prompt"])
    time_to_choose_backend = loop.time() - router_start_time

    async with _request_lock:
        _request_routes[request_id] = backend
        app.state.on_the_fly_requests += 1
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(f"{backend}/generate", json=payload)
            resp.raise_for_status()
            resp_data = resp.json()

            response_text = resp_data.get("response_text", "")
            ttft = resp_data.get("TTFT", 0.0)
            time_between_tokens_ms = resp_data.get("time_between_tokens_ms", [])
            avg_time_between_tokens = resp_data.get("avg_time_between_tokens") or (
                (sum(time_between_tokens_ms) / len(time_between_tokens_ms) / 1000.0) if time_between_tokens_ms else 0.0
            )
            # End-to-end latency from scheduled_wall_time (request_start) to response received
            latency_sec = time.time() - request_start
            latency_ms = latency_sec * 1000
            ttft_ms = ttft * 1000.0

            app.state.metrics.record_request(
                latency_ms=latency_ms,
                backend=backend,
                success=True,
                ttft_ms=ttft_ms,
                time_between_tokens_ms=time_between_tokens_ms,
            )

            return JSONResponse(content={
                "backend": backend,
                "response_text": response_text,
                "time_to_choose_backend": time_to_choose_backend,
                "TTFT": ttft,
                "TTFT_ms": ttft_ms,
                "avg_time_between_tokens": avg_time_between_tokens,
                "time_between_tokens_ms": time_between_tokens_ms,
                "latency_sec": latency_sec,
                "latency_ms": latency_ms,
            })
    except Exception as e:
        latency_ms = (time.time() - request_start) * 1000
        app.state.metrics.record_request(
            latency_ms=latency_ms,
            backend=backend,
            success=False,
        )
        raise
    finally:
        async with _request_lock:
            _request_routes.pop(request_id, None)
        app.state.on_the_fly_requests -= 1
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


@app.get("/metrics")
async def get_metrics(detail: str = "full"):
    """Get system-level goodput metrics.

    detail: "full" returns ttfts_ms, tbt_lists_ms, chosen_backends as lists;
            "summary" returns avg_ttft_ms, avg_tbt_ms, backend_counts instead.
    """
    metrics: RouterMetrics = app.state.metrics
    if detail not in ("full", "summary"):
        return JSONResponse(
            {"error": "detail must be 'full' or 'summary'"},
            status_code=400,
        )
    return JSONResponse(content=metrics.get_metrics_dict(detail=detail))


@app.get("/config")
async def get_config():
    """Get current router batch config."""
    selectors: List[BackendSelector] = app.state.backend_selectors
    sel = selectors[0]
    return JSONResponse(content={
        "batch_size": sel.batch_size,
        "batch_timeout_ms": sel.batch_timeout_ms,
        "router_model_devices": ROUTER_MODEL_DEVICES,
        "num_router_models": len(selectors),
    })


@app.patch("/config")
async def update_config(req: BatchConfigUpdate):
    """Update router batch config at runtime. Takes effect on next batch cycle."""
    if req.batch_size is not None and not 1 <= req.batch_size <= 64:
        return JSONResponse(
            {"error": "batch_size must be between 1 and 64"},
            status_code=400,
        )
    if req.batch_timeout_ms is not None and not 1 <= req.batch_timeout_ms <= 2000:
        return JSONResponse(
            {"error": "batch_timeout_ms must be between 1 and 2000"},
            status_code=400,
        )
    selectors: List[BackendSelector] = app.state.backend_selectors
    for sel in selectors:
        if req.batch_size is not None:
            sel.batch_size = req.batch_size
        if req.batch_timeout_ms is not None:
            sel.batch_timeout_ms = req.batch_timeout_ms
    sel = selectors[0]
    return JSONResponse(content={
        "batch_size": sel.batch_size,
        "batch_timeout_ms": sel.batch_timeout_ms,
    })
