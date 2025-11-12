"""USPTO Open Data Protocol (ODP) API client."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import httpx
from pydantic import SecretStr


class ApiUsptoClient:
    """Asynchronous client for the USPTO Open Data Protocol (ODP)."""

    BASE_URL = "https://api.uspto.gov"
    USER_AGENT = "patent-mcp-server/1.0"
    DEFAULT_TIMEOUT = 30.0

    def __init__(self, api_key: Optional[SecretStr], *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout
        api_key_value = api_key.get_secret_value() if api_key else None

        headers = {"User-Agent": self.USER_AGENT}
        if api_key_value:
            headers["X-API-KEY"] = api_key_value

        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers=headers,
            http2=True,
            follow_redirects=True,
            timeout=self._timeout,
        )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_params(params: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if not params:
            return None

        cleaned: Dict[str, Any] = {}
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, bool):
                cleaned[key] = str(value).lower()
            elif isinstance(value, (list, tuple, set)):
                cleaned[key] = ",".join(str(item) for item in value)
            else:
                cleaned[key] = value
        return cleaned or None

    @staticmethod
    def _clean_payload(payload: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if payload is None:
            return None
        return {key: value for key, value in payload.items() if value is not None}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            response = await self._client.request(
                method,
                path if path.startswith("/") else f"/{path}",
                params=self._clean_params(params),
                json=self._clean_payload(json),
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            raise ValueError(f"HTTP error: {status_code} - {exc.response.text}") from exc
        except httpx.RequestError as exc:
            raise ValueError(f"Unexpected error: {exc}") from exc

    async def get(self, path: str, *, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return await self._request("GET", path, params=params)

    async def post(
        self,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await self._request("POST", path, params=params, json=json)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    APPLICATIONS_PATH = "/api/v1/patent/applications"
    STATUS_CODES_PATH = "/api/v1/patent/status-codes"
    DATASETS_PATH = "/api/v1/datasets/products"

    async def get_application(self, app_num: str) -> Dict[str, Any]:
        return await self.get(f"{self.APPLICATIONS_PATH}/{app_num}")

    async def get_application_resource(self, app_num: str, resource: str) -> Dict[str, Any]:
        return await self.get(f"{self.APPLICATIONS_PATH}/{app_num}/{resource}")

    async def search_applications(self, **params: Any) -> Dict[str, Any]:
        return await self.get(f"{self.APPLICATIONS_PATH}/search", params=params)

    async def search_applications_post(self, **payload: Any) -> Dict[str, Any]:
        pagination = payload.pop("pagination", None)
        if pagination is None and {
            payload.get("offset"),
            payload.get("limit"),
        } != {None}:
            pagination = {
                "offset": payload.pop("offset", 0),
                "limit": payload.pop("limit", 25),
            }
        if pagination:
            payload["pagination"] = pagination
        return await self.post(f"{self.APPLICATIONS_PATH}/search", json=payload)

    async def download_applications(self, **params: Any) -> Dict[str, Any]:
        return await self.get(f"{self.APPLICATIONS_PATH}/search/download", params=params)

    async def download_applications_post(self, **payload: Any) -> Dict[str, Any]:
        pagination = payload.pop("pagination", None)
        if pagination is None and {
            payload.get("offset"),
            payload.get("limit"),
        } != {None}:
            pagination = {
                "offset": payload.pop("offset", 0),
                "limit": payload.pop("limit", 25),
            }
        if pagination:
            payload["pagination"] = pagination
        return await self.post(f"{self.APPLICATIONS_PATH}/search/download", json=payload)

    async def get_status_codes(self, **params: Any) -> Dict[str, Any]:
        return await self.get(self.STATUS_CODES_PATH, params=params)

    async def get_status_codes_post(self, **payload: Any) -> Dict[str, Any]:
        return await self.post(self.STATUS_CODES_PATH, json=payload)

    async def search_datasets(self, **params: Any) -> Dict[str, Any]:
        return await self.get(f"{self.DATASETS_PATH}/search", params=params)

    async def get_dataset_product(self, product_id: str, **params: Any) -> Dict[str, Any]:
        return await self.get(f"{self.DATASETS_PATH}/{product_id}", params=params)

    async def close(self) -> None:
        await self._client.aclose()
