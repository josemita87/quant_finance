"""USPTO Public Search client (ppubs.uspto.gov)."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import httpx


logger = logging.getLogger(__name__)


class PpubsClient:
    """Asynchronous client for the USPTO Public Search API."""

    BASE_URL = "https://ppubs.uspto.gov"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    )
    DEFAULT_LIMIT = 500

    def __init__(self, search_query_path: Optional[Path] = None) -> None:
        self._headers = {
            "X-Requested-With": "XMLHttpRequest",
            "User-Agent": self.USER_AGENT,
            "Origin": self.BASE_URL,
            "Referer": f"{self.BASE_URL}/pubwebapp/",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
            "Priority": "u=1, i",
        }

        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers=self._headers,
            http2=True,
            follow_redirects=True,
        )
        self._session: Dict[str, Any] = {}
        self._case_id: Optional[str] = None
        self._access_token: Optional[str] = None

        script_dir = Path(__file__).parent.parent
        default_path = script_dir / "json" / "search_query.json"
        path = search_query_path or default_path
        with path.open("r", encoding="utf-8") as file:
            self._search_template = json.load(file)

    # ------------------------------------------------------------------
    # Session handling
    # ------------------------------------------------------------------

    async def ensure_session(self) -> Dict[str, Any]:
        if self._case_id:
            return self._session

        self._client.cookies = httpx.Cookies()
        await self._client.get("/pubwebapp/")

        response = await self._client.post(
            "/api/users/me/session",
            json=-1,
            headers={**self._headers, "X-Access-Token": "null"},
        )

        if response.status_code != 200:
            raise ValueError(
                f"Failed to establish session: {response.status_code} - {response.text}"
            )

        self._session = response.json()
        self._case_id = self._session["userCase"]["caseId"]
        self._access_token = response.headers.get("X-Access-Token")
        if self._access_token:
            self._client.headers["X-Access-Token"] = self._access_token

        logger.debug("ppubs session established with case_id=%s", self._case_id)
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        require_session: bool = True,
        **kwargs: Any,
    ) -> httpx.Response:
        if require_session:
            await self.ensure_session()

        response = await self._client.request(method, path, **kwargs)

        if response.status_code == 403 and require_session:
            logger.debug("ppubs session expired, refreshing")
            await self.ensure_session()
            response = await self._client.request(method, path, **kwargs)

        if response.status_code == 429:
            wait_time = int(response.headers.get("x-rate-limit-retry-after-seconds", 5)) + 1
            logger.warning("ppubs rate limited, backing off for %s seconds", wait_time)
            await asyncio.sleep(wait_time)
            response = await self._client.request(method, path, **kwargs)

        return response

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    def _prepare_query_payload(
        self,
        *,
        query: str,
        start: int,
        limit: int,
        sort: str,
        default_operator: str,
        sources: Iterable[str],
        expand_plurals: bool,
        british_equivalents: bool,
    ) -> Dict[str, Any]:
        payload = json.loads(json.dumps(self._search_template))
        payload["start"] = start
        payload["pageCount"] = limit
        payload["sort"] = sort
        payload["query"].update(
            {
                "caseId": self._case_id,
                "op": default_operator,
                "q": query,
                "queryName": query,
                "userEnteredQuery": query,
                "databaseFilters": [
                    {"databaseName": source, "countryCodes": []} for source in sources
                ],
                "plurals": expand_plurals,
                "britishEquivalents": british_equivalents,
            }
        )
        return payload

    @staticmethod
    def _first_record(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for key in ("patents", "docs"):
            records = result.get(key)
            if records:
                return records[0]
        return None

    async def run_query(
        self,
        query: str,
        *,
        start: int = 0,
        limit: int = DEFAULT_LIMIT,
        sort: str = "date_publ desc",
        default_operator: str = "OR",
        sources: Optional[Iterable[str]] = None,
        expand_plurals: bool = True,
        british_equivalents: bool = True,
    ) -> Dict[str, Any]:
        await self.ensure_session()

        payload = self._prepare_query_payload(
            query=query,
            start=start,
            limit=limit,
            sort=sort,
            default_operator=default_operator,
            sources=sources or ("US-PGPUB", "USPAT", "USOCR"),
            expand_plurals=expand_plurals,
            british_equivalents=british_equivalents,
        )

        counts_response = await self._request(
            "POST",
            "/api/searches/counts",
            json=payload["query"],
        )

        if counts_response.status_code >= 400:
            raise ValueError(
                f"Search counts failed: {counts_response.status_code} - {counts_response.text}"
            )

        search_response = await self._request(
            "POST",
            "/api/searches/searchWithBeFamily",
            json=payload,
        )

        if search_response.status_code != 200:
            raise ValueError(
                f"Search query failed: {search_response.status_code} - {search_response.text}"
            )

        result = search_response.json()
        if result.get("error"):
            error = result["error"]
            return {
                "error": True,
                "errorCode": error.get("errorCode"),
                "message": error.get("errorMessage"),
            }
        return result

    async def get_document(self, guid: str, source_type: str) -> Dict[str, Any]:
        await self.ensure_session()

        response = await self._request(
            "GET",
            f"/api/patents/highlight/{guid}",
            params={
                "queryId": 1,
                "source": source_type,
                "includeSections": True,
                "uniqueId": None,
            },
        )

        if response.status_code != 200:
            raise ValueError(
                f"Document request failed: {response.status_code} - {response.text}"
            )

        return response.json()

    async def get_document_by_number(self, patent_number: str) -> Dict[str, Any]:
        record = await self._find_patent_record(patent_number)
        if not isinstance(record, dict):
            return {"error": True, "message": f"Patent {patent_number} not found"}
        if record.get("error"):
            return record
        return await self.get_document(record["guid"], record["type"])

    async def download_patent_pdf(self, patent_number: str) -> Dict[str, Any]:
        record = await self._find_patent_record(patent_number)
        if not isinstance(record, dict):
            return {"error": True, "message": f"Patent {patent_number} not found"}
        if record.get("error"):
            return record

        image_location = record.get(
            "imageLocation",
            record.get("document_structure", {}).get("image_location"),
        )
        page_count = record.get(
            "pageCount",
            record.get("document_structure", {}).get("page_count"),
        )

        if not image_location or not page_count:
            return {
                "error": True,
                "message": "Missing image location or page count information",
            }

        return await self._download_pdf(
            guid=record["guid"],
            image_location=image_location,
            page_count=page_count,
            document_type=record["type"],
        )

    async def _find_patent_record(self, patent_number: str) -> Optional[Dict[str, Any]]:
        queries = [f'patentNumber:"{patent_number}"', f'"{patent_number}".pn.']
        for query in queries:
            result = await self.run_query(query=query, limit=1, sources=("USPAT",))
            if result.get("error"):
                return result
            record = self._first_record(result)
            if record:
                return record
        return None

    async def _download_pdf(
        self,
        *,
        guid: str,
        image_location: str,
        page_count: int,
        document_type: str,
    ) -> Dict[str, Any]:
        await self.ensure_session()

        page_keys = [f"{image_location}/{index:0>8}.tif" for index in range(1, page_count + 1)]

        response = await self._client.post(
            "/api/print/imageviewer",
            json={
                "caseId": self._case_id,
                "pageKeys": page_keys,
                "patentGuid": guid,
                "saveOrPrint": "save",
                "source": document_type,
            },
        )

        if response.status_code >= 500:
            raise ValueError(
                f"PDF save request failed: {response.status_code} - {response.text}"
            )

        print_job_id = response.text.strip()

        while True:
            poll_response = await self._client.post(
                "/api/print/print-process",
                json=[print_job_id],
            )

            if poll_response.status_code != 200:
                raise ValueError(
                    f"Print job status failed: {poll_response.status_code} - {poll_response.text}"
                )

            poll_data = poll_response.json()[0]
            if poll_data.get("printStatus") == "COMPLETED":
                pdf_name = poll_data.get("pdfName")
                break

            await asyncio.sleep(1)

        download_request = self._client.build_request(
            "GET",
            f"/api/internal/print/save/{pdf_name}",
        )

        download_response = await self._client.send(download_request, stream=True)
        if download_response.status_code != 200:
            raise ValueError(
                f"PDF download failed: {download_response.status_code} - {download_response.text}"
            )

        content = await download_response.aread()
        b64_content = base64.b64encode(content).decode("utf-8")
        return {
            "success": True,
            "filename": f"{guid}.pdf",
            "content_type": "application/pdf",
            "content": b64_content,
        }

    async def close(self) -> None:
        await self._client.aclose()
