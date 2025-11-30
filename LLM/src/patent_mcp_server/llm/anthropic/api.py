"""Anthropic API client integration."""
from __future__ import annotations
import asyncio
from typing import Tuple, TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, SecretStr, ValidationError
import anthropic

from patent_mcp_server.llm.models import TokenUsage
from patent_mcp_server.llm.exceptions import (
    LLMConnectionError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseError,
    LLMWebSearchError,
)
from patent_mcp_server.llm.anthropic.models import AnthropicRequestDTO

if TYPE_CHECKING:
    from llm.anthropic.enums import AnthropicModel
    from llm.models import LLMRequest
    from server.main import USPTOMCPServer # noqa: F401


class AnthropicAPI:
    """Anthropic (Claude) API client for LLM interactions with MCP server."""

    def __init__(
        self,
        api_key: SecretStr,
        mcp_server: 'USPTOMCPServer',
        *,
        anthropic_version: str = "2023-06-01",
    ):
        """Initialize the Anthropic API client with MCP server.

        Args:
            api_key: Anthropic API key.
            mcp_server: USPTOMCPServer instance to use for MCP tools.
            anthropic_version: Anthropic API version header.
        """
        try:
            # Store MCP server reference
            self.mcp_server = mcp_server

            # Initialize Anthropic client
            self.client = anthropic.AsyncAnthropic(
                api_key=api_key.get_secret_value(),
                default_headers={"anthropic-version": anthropic_version},
            )
        except Exception as e:
            raise LLMRequestError(
                message=f"Failed to initialize Anthropic client: {e}", status_code=400
            ) from e

    async def __call__(self, request: LLMRequest) -> Tuple[BaseModel, TokenUsage]:
        """Process an LLM request and return (parsed_output, token_usage)."""

        try:
            return await self._messages(
                AnthropicRequestDTO(
                    **request.model_dump(),
                    public_url=self.mcp_server.public_url,
                    name=self.mcp_server.name
                )
            )

        except Exception as e:
            if isinstance(e, anthropic.APIConnectionError):
                raise LLMConnectionError(
                    message=f"Failed to connect to Anthropic API: {e}", status_code=503
                ) from e
            elif isinstance(e, anthropic.RateLimitError):
                raise LLMRateLimitError(
                    message=f"Anthropic API rate limit exceeded: {e}", status_code=429
                ) from e
            elif isinstance(e, anthropic.BadRequestError):
                raise LLMRequestError(
                    message=f"Invalid request to Anthropic API: {e}", status_code=400
                ) from e
            elif isinstance(e, anthropic.APIStatusError):
                raise LLMResponseError(
                    message=f"Anthropic API returned error: {e}.",
                    status_code=e.status_code,
                    response=e,
                ) from e
            elif isinstance(e, ValueError) and "refused" in str(e).lower():
                raise LLMWebSearchError(
                    message=f"Anthropic model refused the request: {e}.", status_code=400
                ) from e
            else:
                raise LLMResponseError(
                    message=f"Unexpected error in Anthropic API: {e}.",
                    status_code=500,
                ) from e

    async def _messages(self, r: AnthropicRequestDTO) -> Tuple[BaseModel, TokenUsage]:
        """Perform Claude Messages API call using precomputed args and parse response."""

        resp = await self.client.beta.messages.create(
                model=r.api_model.value.id,
                messages=r.api_messages,
                system=r.instructions,
                max_tokens=r.max_tokens,
                temperature=r.temperature,
                mcp_servers=r.mcp_servers,
                extra_headers=r.api_extra_headers,
            )

        last_message = resp.content[-1].text if resp.content else ""

        try:
            parsed: BaseModel = r.output_model.model_validate_json(last_message)
        except (ValidationError, ValueError):
            parsed = r.output_model(response=last_message)

        tokens_usage = self.calculate_tokens(usage=resp.usage, model=r.api_model)
        return parsed, tokens_usage

    @staticmethod
    def calculate_tokens(usage: Any, model: AnthropicModel) -> TokenUsage:
        """Convert Claude usage → TokenUsage. Pricing is app-specific; cost=0 here."""
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        cached_tokens = getattr(usage, "cache_creation_input_tokens", 0)
        cost = (
            model.value.input_cost * (input_tokens - cached_tokens)
            + model.value.output_cost * output_tokens
            + model.value.cached_cost * cached_tokens
        )

        return TokenUsage(
            model=model.value.id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost=cost,
        )
