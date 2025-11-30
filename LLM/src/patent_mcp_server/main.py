#!/usr/bin/env python3
"""
USPTO Patent MCP Server with Anthropic Integration
"""

import asyncio
import sys
from pydantic import BaseModel, Field

from patent_mcp_server.llm.anthropic.api import AnthropicAPI
from patent_mcp_server.llm.models import LLMRequest
from patent_mcp_server.server.main import USPTOMCPServer
from patent_mcp_server.settings import settings
from patent_mcp_server.domain.lawyer import LawyerAgent



async def main():
    """Main entry point for the USPTO Patent MCP Server."""
    try:

        # Initialize domain agent boundary
        agent = LawyerAgent()

        # Initialize MCP server & Ngrok tunnel
        mcp_server = USPTOMCPServer(
            ngrok_auth_token=settings.ngrok_auth_token,
            uspto_api_key=settings.uspto_api_key
        )

        # Get the public endpoint
        endpoint = mcp_server.run()
        if endpoint:
            print(f"MCP endpoint available at: {endpoint}")

        # Initialize Anthropic API
        anthropic_api = AnthropicAPI(
            api_key=settings.anthropic_api_key,
            mcp_server=mcp_server
        )

        # Initialize conversation history
        conv_history = []

        # Initialize conversation loop
        try:
            while True:
                user_input = input("\n👤 You: ")

                request = LLMRequest(
                    messages=conv_history,
                    instructions=agent.system_template,
                    input=user_input.format(user_input=user_input, conversation_history=conv_history),
                    output_model=agent.LawyerResponse
                )

                parsed_response, usage = await anthropic_api(request=request)
                print(f"🤖 Claude: {parsed_response.response}")
                print(f"$ {usage.cost} cost")
                conv_history.append({'role': 'user', 'content': user_input})
                conv_history.append({'role': 'assistant', 'content': parsed_response.response})

        except KeyboardInterrupt:
            raise KeyboardInterrupt
        finally:
            # Clean shutdown
            mcp_server.stop()

    except Exception as e:
        raise e


if __name__ == "__main__":
    asyncio.run(main())
