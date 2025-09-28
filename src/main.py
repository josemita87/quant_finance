#!/usr/bin/env python3
"""
USPTO Patent MCP Server with Anthropic Integration
"""

import asyncio
import sys
import logging
from pathlib import Path
from pydantic import BaseModel, Field
# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from llm.anthropic.api import AnthropicAPI
from llm.models import LLMRequest
from server.main import USPTOMCPServer
from src.settings import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('uspto_patent_main')


class ResponseModel(BaseModel):
    response: str = Field(default_factory=str, description="The response from the USPTO MCP Server")


async def main():
    """Main entry point for the USPTO Patent MCP Server."""
    try:
        logger.info("Starting USPTO Patent MCP Server")

        # Initialize MCP server & Ngrok tunnel
        mcp_server = USPTOMCPServer(ngrok_auth_token=settings.ngrok_auth_token)
        mcp_server.run()  # This now returns immediately

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
                    instructions="You are a helpful assistant that can answer questions about the USPTO.",
                    input = user_input,
                    output_model = ResponseModel
                )

                response = await anthropic_api(request=request)
                print(f"🤖 Claude: {response}")

                conv_history.append({'role': 'user', 'content': user_input})
                conv_history.append({'role': 'assistant', 'content': response[0].content})

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            # Clean shutdown
            mcp_server.stop()
            logger.info("Server stopped")

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
