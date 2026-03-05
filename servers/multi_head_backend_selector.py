"""
Multi-head backend selection logic for the router.

Uses a multi-head regression model that outputs a score per model.
Selects the model with the highest score for each prompt.
Provides batched inference via a queue and background processor.
"""

import asyncio
from typing import List, Tuple, Optional

import numpy as np
import torch


class MultiHeadBackendSelector:
    """
    Selects backends for prompts using batched multi-head router model inference.
    Each head outputs a score for one model; the model with the highest score is selected.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_urls: List[str],
        batch_size: int = 4,
        batch_timeout_ms: int = 20,
    ):
        """
        Args:
            model: DeBERTaMultiHeadRegression model (num_heads must match len(model_urls)).
            tokenizer: Tokenizer for the model.
            model_urls: List of backend URLs, one per head. Order must match model head order.
            batch_size: Max prompts per batch.
            batch_timeout_ms: Max wait time to fill a batch.
        """
        if len(model_urls) < 2:
            raise ValueError("model_urls must have at least 2 entries")
        self.model = model
        self.tokenizer = tokenizer
        self.model_urls = model_urls
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._batch_processor_task: Optional[asyncio.Task] = None

    def _choose_backend_batched(self, prompts: List[str]) -> List[str]:
        """Process a batch of prompts and return backends (URL with highest score per prompt)."""
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

        # outputs: (batch_size, num_heads)
        scores = outputs.cpu().numpy()
        scores = np.atleast_2d(scores)

        # argmax along head dimension -> index of best model
        best_indices = np.argmax(scores, axis=1)
        backends = [self.model_urls[idx] for idx in best_indices]
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
        """Choose backend for a prompt using batched inference (model with highest score)."""
        future = asyncio.Future()
        await self._batch_queue.put((prompt, future))
        return await future
