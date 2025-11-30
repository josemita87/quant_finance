"""Enums for LLM adapters."""

from __future__ import annotations

from typing import NamedTuple
from enum import Enum


class LLMAdapter(Enum):
    """Available LLM adapters."""

    OPENAI = 'OpenAI'
    ANTHROPIC = 'Anthropic'


class ModelInfo(NamedTuple):
    """Information structure for LLM models."""

    id: str
    input_cost: float
    output_cost: float
    cached_cost: float


class LLMModel(Enum):
    """Base enum for LLM models."""
