"""Anthropic model definitions and pricing info."""
from __future__ import annotations
from llm.enums import LLMModel, ModelInfo


class AnthropicModel(LLMModel):
    """Anthropic Claude models with rough pricing placeholders (per token)."""

    # Current dated model IDs (avoid 404s); adjust costs as needed in your app
    SONNET_4 = ModelInfo(
        id='claude-sonnet-4-20250514',
        input_cost=3.0 / 1_000_000,
        output_cost=15.0 / 1_000_000,
        cached_cost=0.0,
    )
