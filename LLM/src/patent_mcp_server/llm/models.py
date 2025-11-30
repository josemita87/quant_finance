"""Models for LLM requests."""

from __future__ import annotations
from typing import Optional, List, Dict, Type
from pydantic import BaseModel, ConfigDict, Field


class LLMRequest(BaseModel):
    """A provider-agnostic request to an LLM."""

    model_config = ConfigDict(strict=True)

    messages: Optional[List[Dict[str, str]]] = Field(
        default_factory=list, description='History of conversation messages for the LLM'
    )
    instructions: str = Field(description='Instructions for the LLM')
    input: str = Field(description='Input to the LLM')
    output_model: Type[BaseModel] = Field(description='Output model for the LLM')
    temperature: float = Field(default=0.7, description='Temperature for the LLM')
    max_tokens: int = Field(
        default=3000, ge=500, description='Maximum tokens for the LLM (minimum 500 tokens)'
    )



class TokenUsage(BaseModel):
    """Represents the usage of tokens for a specific model."""

    model: str = Field(default='not_applicable', description='Name of the LLM model used')
    input_tokens: int = Field(default=0, description='Number of input tokens used')
    output_tokens: int = Field(default=0, description='Number of output tokens used')
    cached_tokens: int = Field(default=0, description='Number of cached tokens used')
    cost: float = Field(default=0.0, description='Cost of the tokens used')

    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add two token usage objects together.

        Args:
            other: Another TokenUsage object to add to this one.

        Returns:
            TokenUsage: A new TokenUsage object with combined values.
        """
        return TokenUsage(
            model=self.model,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cost=self.cost + other.cost,
        )
