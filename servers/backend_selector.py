"""
Backend selection logic for the router.

Provides batched inference to choose which model backend (A or B) should
handle each prompt. Uses a queue and background processor for batching.
"""

import asyncio
from typing import List, Tuple, Optional
import time
import numpy as np
import torch


class BackendSelector:
    """
    Selects backends for prompts using batched router model inference.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_a_url: str,
        model_b_url: str,
        threshold: float = 0.5,
        batch_size: int = 4,
        batch_timeout_ms: int = 20,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_a_url = model_a_url
        self.model_b_url = model_b_url
        self.threshold = threshold
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._batch_processor_task: Optional[asyncio.Task] = None

    def _choose_backend_batched(self, prompts: List[str]) -> List[str]:
        """Process a batch of prompts and return backends."""
        device = next(self.model.parameters()).device

        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

        if outputs.dim() == 1:
            scores = outputs.cpu().numpy()
        else:
            scores = outputs.squeeze(-1).cpu().numpy()
        scores = np.atleast_1d(scores)

        backends = [
            self.model_a_url if score > self.threshold else self.model_b_url
            for score in scores
        ]
        return backends

    async def _batch_processor(self) -> None:
        """Background task that processes batches from the queue."""
        while True:
            batch_items: List[Tuple[str, asyncio.Future]] = []

            try:
                prompt, future = await self._batch_queue.get()
                batch_items.append((prompt, future))

                try:
                    async def collect_additional_items():
                        while len(batch_items) < self.batch_size:
                            prompt, future = await self._batch_queue.get()
                            batch_items.append((prompt, future))

                    await asyncio.wait_for(
                        collect_additional_items(),
                        timeout=self.batch_timeout_ms / 1000.0,
                    )
                except asyncio.TimeoutError:
                    pass
            except Exception as e:
                print(f"Error in batch processor queue: {e}")
                continue

            try:
                prompts = [item[0] for item in batch_items]
                # Run synchronous tokenization + model inference in a thread
                # so we don't block the asyncio event loop (which would prevent
                # FastAPI from accepting new connections).
                loop = asyncio.get_running_loop()
                backends = await loop.run_in_executor(
                    None, self._choose_backend_batched, prompts
                )
                for (_, future), backend in zip(batch_items, backends):
                    if not future.done():
                        future.set_result(backend)
            except Exception as e:
                print(f"Error processing batch: {e}")
                for _, future in batch_items:
                    if not future.done():
                        future.set_exception(e)

    async def start(self) -> None:
        """Start the batch processor background task."""
        self._batch_processor_task = asyncio.create_task(self._batch_processor())

    async def stop(self) -> None:
        """Stop the batch processor background task."""
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass
            self._batch_processor_task = None

    async def choose_backend(self, prompt: str) -> str:
        """Choose backend for a prompt using batched inference."""
        future = asyncio.Future()
        await self._batch_queue.put((prompt, future))
        return await future
