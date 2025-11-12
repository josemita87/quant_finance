"""Domain-specific agent definition for USPTO legal workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, List

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class LawyerAgent:
    """Encapsulates prompt templates for the USPTO-focused legal agent.

    The class currently exposes the system prompt and a basic user-input template.
    Future enhancements can expand on this boundary to include richer context
    assembly, tool routing, and history management without forcing changes to
    downstream LLM adapters.
    """

    class LawyerResponse(BaseModel):
        response: str = Field(
            default_factory=str,
            description="The response from the USPTO MCP Server"
        )
        sources: Optional[List[str]] = Field(
            default=None,
            description="The sources used to generate the response"
        )

    system_template: str = """
        <system>

        <role>
        You are a legal research assistant specializing in USPTO regulations,
        procedures, and data.
        </role>

        <instructions>
        Provide thorough, practical guidance and cite relevant sources when possible. Ask clarifying questions when the request lacks critical details.
        </instructions>

        <context>
        You are connected to the USPTO MCP Server, which provides access to USPTO regulations, procedures, structured data and unstructured data. You can use the tools provided to you to get the information you need.
        </context>

        <output>
        You MUST return your final answer as STRICT JSON only (no prose before or after).
        The JSON MUST validate against the following JSON Schema (draft-07 compatible):

        {
          "type": "object",
          "required": ["response", "sources"],
          "additionalProperties": false,
          "properties": {
            "response": {
              "type": "string",
              "description": "The response from the USPTO MCP Server"
            },
            "sources": {
              "type": "array",
              "items": { "type": "string" },
              "description": "The sources used to generate the response"
            }
          }
        }

        Return ONLY that JSON object.
        </output>

        </system>
    """
    input_template: str = "Client request:\n{user_input}\n\nConversation history:\n{conversation_history}"
