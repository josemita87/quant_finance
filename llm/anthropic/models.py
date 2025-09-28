"""DTOs and helpers for Anthropic request composition."""
from __future__ import annotations
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from llm.models import LLMRequest
from llm.anthropic.enums import AnthropicModel


class AnthropicRequestDTO(LLMRequest):
    """Minimal DTO for Anthropic Messages API.

    Inherits from provider-agnostic LLMRequest and exposes only the Anthropic-
    specific arguments you need to pass into `client.messages.create(...)`.
    """

    public_url: str
    name: str

    # ---- Properties directly mapped to Anthropic arguments ----

    @property
    def api_model(self) -> AnthropicModel:
        """Anthropic model id.

        Choose based on behavior/web_search; adjust mapping as needed.
        """
        return AnthropicModel.SONNET_4


    @property
    def api_messages(self) -> List[Dict[str, Any]]:
        """Anthropic 'messages' array (role/content).

        We keep only user/assistant from history if provided and append
        current input as a user turn.
        """
        msgs: List[Dict[str, Any]] = []
        if self.messages:
            for m in self.messages:
                role = m.get("role")
                if role in {"user", "assistant"}:
                    msgs.append({"role": role, "content": m.get("content", "")})
        # Always include the current user input as the latest turn
        msgs.append({"role": "user", "content": self.input})
        return msgs

    @property
    def mcp_servers(self) -> Optional[List[Dict[str, Any]]]:
        """Anthropic Remote MCP connector configuration.

        Present only if URL/Name are set.
        """
        if not self.public_url:
            return None

        entry: Dict[str, Any] = {
            "type": "url",
            "url": self.public_url,
            "name": self.name,
        }

        return [entry]

    @property
    def api_extra_headers(self) -> Optional[Dict[str, str]]:
        """Extra headers for Anthropic requests (e.g., MCP beta header).

        Return {} when not needed so callers can pass it directly.
        """
        return {"anthropic-beta": "mcp-client-2025-04-04"}
