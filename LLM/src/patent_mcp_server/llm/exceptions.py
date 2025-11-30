"""Exceptions for LLM adapters."""

from __future__ import annotations

from typing import Any, Optional


class AdapterError(Exception):
    """Base exception for adapter errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        """Initialize the adapter error.

        Args:
            message: Error message describing the adapter issue.
            status_code: Optional HTTP status code, defaults to 500.
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code or 500


class LLMAdapterError(AdapterError):
    """Base exception for LLM adapter errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        """Initialize the LLM adapter error.

        Args:
            message: Error message describing the LLM adapter issue.
            status_code: Optional HTTP status code, defaults to 500.
        """
        super().__init__(message=message, status_code=status_code or 500)


class LLMConnectionError(LLMAdapterError):
    """Raised when there's a problem connecting to the LLM service."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        """Initialize the LLM connection error.

        Args:
            message: Error message describing the connection issue.
            status_code: Optional HTTP status code, defaults to 503.
        """
        super().__init__(message=message, status_code=status_code or 503)


class LLMRateLimitError(LLMAdapterError):
    """Raised when the LLM service rate limit is exceeded."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        """Initialize the LLM rate limit error.

        Args:
            message: Error message describing the rate limit issue.
            status_code: Optional HTTP status code, defaults to 429.
        """
        super().__init__(message=message, status_code=status_code or 429)


class LLMRequestError(LLMAdapterError):
    """Raised when the request to the LLM service is malformed or invalid."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        """Initialize the LLM request error.

        Args:
            message: Error message describing the request issue.
            status_code: Optional HTTP status code, defaults to 400.
        """
        super().__init__(message=message, status_code=status_code or 400)


class LLMResponseError(LLMAdapterError):
    """Raised for non-200 API responses from the LLM service."""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Any = None):
        """Initialize the LLM response error.

        Args:
            message: Error message describing the response issue.
            status_code: Optional HTTP status code, defaults to 500.
            response: Optional response object that caused the error.
        """
        super().__init__(message=message, status_code=status_code or 500)
        self.response = response


class LLMWebSearchError(LLMAdapterError):
    """Raised when the websearch output model is incompatible."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        """Initialize the LLM web search error.

        Args:
            message: Error message describing the web search issue.
            status_code: Optional HTTP status code, defaults to 400.
        """
        super().__init__(message=message, status_code=status_code or 400)
