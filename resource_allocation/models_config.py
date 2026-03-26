"""
Centralized configuration for the routing models.

This module is intended to be the single source of truth for:
- canonical model order (must match the RouterBench score columns)
- the score column names inside `routerbench_0shot_scores.pkl`
- the corresponding per-model memory + performance file names
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ModelSpec:
    """
    Canonical spec for one backend model used by the router.

    Notes
    -----
    - `score_column` is the column key inside routerbench_0shot_scores.pkl.
    - `memory_filename` is located under `profiler/model_memory/`.
    - `performance_filename` is located under the project root.
    """

    model_id: str
    display_name: str
    score_column: str
    memory_filename: str
    performance_filename: str


# Canonical routing order. This order must match the score columns order after
# we reorder using `score_columns` metadata from the scores pickle.
ROUTING_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="mistral-7b",
        display_name="Mistral 7B",
        score_column="mistralai/mistral-7b-chat",
        memory_filename="min_gpu_util_mistralai_Mistral-7B-v0.1.json",
        performance_filename="performance_data_mistral_7b_final.json",
    ),
    ModelSpec(
        model_id="yi-34b",
        display_name="Yi 34B",
        score_column="zero-one-ai/Yi-34B-Chat",
        memory_filename="min_gpu_util_01-ai_Yi-34B.json",
        performance_filename="performance_data_yi34b_final.json",
    ),
    ModelSpec(
        model_id="vicuna-13b",
        # RouterBench uses WizardLM as the 13B column; we keep the historical
        # naming "Vicuna 13B" for display to match the rest of the pipeline.
        display_name="Vicuna 13B",
        score_column="WizardLM/WizardLM-13B-V1.2",
        memory_filename="min_gpu_util_lmsys_vicuna-13b-v1.5.json",
        performance_filename="performance_data_vicuna_13b_final.json",
    ),
)


def get_score_columns() -> tuple[str, ...]:
    """RouterBench score columns in canonical routing order."""

    return tuple(m.score_column for m in ROUTING_MODELS)


def get_display_names() -> tuple[str, ...]:
    """Human-readable backend names in canonical routing order."""

    return tuple(m.display_name for m in ROUTING_MODELS)


def get_scores_path(root: Path | str) -> Path:
    """Path to `routerbench_0shot_scores.pkl`."""

    root = Path(root)
    return root / "datasets" / "routerbench_0shot_scores.pkl"


def get_model_memory_paths(root: Path | str) -> list[Path]:
    """Paths to per-model memory config JSON files."""

    root = Path(root)
    mem_dir = root / "profiler" / "model_memory"
    return [mem_dir / m.memory_filename for m in ROUTING_MODELS]


def get_performance_data_paths(root: Path | str) -> list[Path]:
    """
    Paths to per-model performance JSON files.

    These are expected to live under the project root (not under profiler/).
    """

    root = Path(root)
    return [root / m.performance_filename for m in ROUTING_MODELS]


def iter_models() -> Iterable[ModelSpec]:
    """Iterate over canonical routing models."""

    return iter(ROUTING_MODELS)


def model_count() -> int:
    return len(ROUTING_MODELS)

