"""
Settings configuration using Pydantic for seamless credential loading.

This module provides a Pydantic BaseSettings class that automatically loads
credentials from the .credentials.env file with proper validation and type hints.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import SecretStr, Field, validator
from pydantic_settings import BaseSettings


class CredentialsSettings(BaseSettings):
    """Pydantic settings for loading and validating credentials."""

    # Anthropic API Configuration
    anthropic_api_key: SecretStr = Field(
        ...,
        description="Anthropic API key for Claude access",
        env="ANTHROPIC_API_KEY"
    )

    # USPTO API Configuration
    uspto_api_key: Optional[SecretStr] = Field(
        default=None,
        description="USPTO API key for enhanced rate limits",
        env="USPTO_API_KEY"
    )

    # Ngrok Configuration
    ngrok_auth_token: Optional[SecretStr] = Field(
        default=None,
        description="Ngrok auth token for authenticated tunnels",
        env="NGROK_AUTH_TOKEN"
    )

    # MCP Server Configuration
    mcp_use_ngrok: bool = Field(
        default=False,
        description="Enable ngrok tunnel for MCP server",
        env="MCP_USE_NGROK"
    )
    mcp_host: str = Field(
        default="localhost",
        description="MCP server host",
        env="MCP_HOST"
    )
    mcp_port: int = Field(
        default=8000,
        description="MCP server port",
        env="MCP_PORT"
    )

    class Config:
        """Pydantic configuration for settings."""
        env_file = ".credentials.env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = CredentialsSettings()
