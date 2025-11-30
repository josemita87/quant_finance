import asyncio
import threading
import time
from typing import Any, Dict, List, Optional, Union

import uvicorn
from pydantic import SecretStr
from mcp.server.fastmcp import FastMCP
from pyngrok import ngrok

from patent_mcp_server.server.api.ppub_uspto import PpubsClient
from patent_mcp_server.server.api.uspto import ApiUsptoClient


class USPTOMCPServer:
    """USPTO MCP Server with ngrok support — all-in-one."""

    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 5003
    DEFAULT_NAME = "uspto_patent_tools"

    def __init__(
        self,
        ngrok_auth_token: Optional[SecretStr],
        uspto_api_key: Optional[SecretStr],
        host: str = None,
        port: int = None,
        name: str = None,
    ):
        """
        Args:
            ngrok_auth_token: Optional ngrok auth token (SecretStr)
            uspto_api_key: Optional USPTO API key (SecretStr)
            host: HTTP host for MCP (defaults to 127.0.0.1)
            port: HTTP port for MCP (defaults to 5003)
            name: MCP server name (defaults to 'uspto_patent_tools')
        """
        self.host = host or self.DEFAULT_HOST
        self.port = port or self.DEFAULT_PORT
        self.name = name or self.DEFAULT_NAME

        # Clients
        self.ppubs_client = PpubsClient()
        self.api_client = ApiUsptoClient(api_key=uspto_api_key)

        # MCP
        self.mcp = FastMCP(self.name, host=self.host, port=self.port)

        # Runtime
        self._stop_event = threading.Event()
        self._mcp_thread: Optional[threading.Thread] = None
        self._ngrok_tunnel = None
        self._ngrok_auth_token = ngrok_auth_token
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._public_base_url: Optional[str] = None

        # Register all tools on this instance
        self._register_tools()

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    @property
    def public_url(self) -> Optional[str]:
        """Return the externally reachable MCP endpoint."""
        base = self._public_base_url or (
            f"http://{self.host}:{self.port}" if self.is_running() else None
        )
        if not base:
            return None
        path = self.mcp.settings.streamable_http_path
        return f"{base.rstrip('/')}{path}"

    @property
    def is_tunnel_active(self) -> bool:
        return self._ngrok_tunnel is not None

    def is_running(self) -> bool:
        return (
            self._mcp_thread is not None
            and self._mcp_thread.is_alive()
            and not self._stop_event.is_set()
        )

    async def _serve_mcp(self) -> None:
        """Run the MCP server with explicit control over uvicorn."""
        app = self.mcp.streamable_http_app()
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level=self.mcp.settings.log_level.lower(),
        )
        self._uvicorn_server = uvicorn.Server(config)
        await self._uvicorn_server.serve()

    def _run_mcp_server(self) -> None:
        """Blocking run of MCP (executed in a thread)."""
        asyncio.run(self._serve_mcp())

    def _setup_ngrok(self) -> Optional[str]:
        """Create ngrok HTTP tunnel."""
        if self._ngrok_auth_token:
            ngrok.set_auth_token(self._ngrok_auth_token.get_secret_value())

        self._ngrok_tunnel = ngrok.connect(addr=f"{self.host}:{self.port}", proto="http")
        return self._ngrok_tunnel.public_url

    def cleanup_ngrok(self):
        """Tear down the ngrok tunnel."""
        try:
            if self._ngrok_tunnel:
                ngrok.disconnect(self._ngrok_tunnel.public_url)
        finally:
            self._ngrok_tunnel = None

    def run(self) -> str:
        """
        Start MCP in a background thread and open ngrok tunnel.

        Returns:
            The externally reachable MCP endpoint (including stream path).
        """
        # Start MCP server
        self._mcp_thread = threading.Thread(
            target=self._run_mcp_server, daemon=True
        )
        self._mcp_thread.start()

        # Wait for uvicorn to come online
        start = time.time()
        while True:
            if self._uvicorn_server and getattr(self._uvicorn_server, "started", False):
                break
            if not self._mcp_thread.is_alive():
                raise RuntimeError("MCP server thread exited before startup completed")
            if time.time() - start > 10:
                raise RuntimeError("Timed out waiting for MCP server startup")
            time.sleep(0.05)

        # Expose via ngrok
        public_base = self._setup_ngrok()
        if not public_base:
            public_base = f"http://{self.host}:{self.port}"
        self._public_base_url = public_base
        return self.public_url

    def stop(self):
        """Stop server and clean resources."""
        self._stop_event.set()

        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
            self._uvicorn_server.force_exit = True

        # Cleanup ngrok
        if self._ngrok_tunnel:
            self.cleanup_ngrok()

        # Join thread (best-effort)
        if self._mcp_thread and self._mcp_thread.is_alive():
            self._mcp_thread.join(timeout=5)
            if self._mcp_thread.is_alive():
                return
        self._mcp_thread = None
        self._uvicorn_server = None
        self._public_base_url = None

    # ---------------------------------------------------------------------
    # Tool registration
    # ---------------------------------------------------------------------

    def _register_tools(self) -> None:
        """Register all tools (bound methods) with FastMCP."""
        # ppubs tools
        self.mcp.tool()(self.ppubs_search_patents)
        self.mcp.tool()(self.ppubs_search_applications)
        self.mcp.tool()(self.ppubs_get_full_document)
        self.mcp.tool()(self.ppubs_get_patent_by_number)
        self.mcp.tool()(self.ppubs_download_patent_pdf)

        # api.uspto.gov tools (metadata/search)
        self.mcp.tool()(self.get_app)
        self.mcp.tool()(self.search_applications)
        self.mcp.tool()(self.search_applications_post)
        self.mcp.tool()(self.download_applications)
        self.mcp.tool()(self.download_applications_post)
        self.mcp.tool()(self.get_app_metadata)
        self.mcp.tool()(self.get_app_adjustment)
        self.mcp.tool()(self.get_app_assignment)
        self.mcp.tool()(self.get_app_attorney)
        self.mcp.tool()(self.get_app_continuity)
        self.mcp.tool()(self.get_app_foreign_priority)
        self.mcp.tool()(self.get_app_transactions)
        self.mcp.tool()(self.get_app_documents)
        self.mcp.tool()(self.get_app_associated_documents)
        self.mcp.tool()(self.get_status_codes)
        self.mcp.tool()(self.get_status_codes_post)
        self.mcp.tool()(self.search_datasets)
        self.mcp.tool()(self.get_dataset_product)

    # ---------------------------------------------------------------------
    # Tools for ppubs.uspto.gov (full text + PDFs)
    # ---------------------------------------------------------------------

    async def ppubs_search_patents(
        self,
        query: str,
        start: Optional[int] = 0,
        limit: Optional[int] = 100,
        sort: Optional[str] = "date_publ desc",
        default_operator: Optional[str] = "OR",
        expand_plurals: Optional[bool] = True,
        british_equivalents: Optional[bool] = True,
    ) -> Dict[str, Any]:
        """Search granted patents in ppubs.uspto.gov (full text)."""
        return await self.ppubs_client.run_query(
            query=query,
            start=start,
            limit=limit,
            sort=sort,
            default_operator=default_operator,
            sources=["USPAT"],
            expand_plurals=expand_plurals,
            british_equivalents=british_equivalents,
        )

    async def ppubs_search_applications(
        self,
        query: str,
        start: Optional[int] = 0,
        limit: Optional[int] = 100,
        sort: Optional[str] = "date_publ desc",
        default_operator: Optional[str] = "OR",
        expand_plurals: Optional[bool] = True,
        british_equivalents: Optional[bool] = True,
    ) -> Dict[str, Any]:
        """Search published applications in ppubs.uspto.gov (full text)."""
        return await self.ppubs_client.run_query(
            query=query,
            start=start,
            limit=limit,
            sort=sort,
            default_operator=default_operator,
            sources=["US-PGPUB"],
            expand_plurals=expand_plurals,
            british_equivalents=british_equivalents,
        )

    async def ppubs_get_full_document(self, guid: str, source_type: str) -> Dict[str, Any]:
        """Get full patent document (claims, description, sections) by GUID."""
        return await self.ppubs_client.get_document(guid, source_type)

    async def ppubs_get_patent_by_number(
        self, patent_number: Union[str, int]
    ) -> Dict[str, Any]:
        """Get a granted patent's full text by number from ppubs.uspto.gov."""
        return await self.ppubs_client.get_document_by_number(str(patent_number))

    async def ppubs_download_patent_pdf(
        self, patent_number: Union[str, int]
    ) -> Dict[str, Any]:
        """Download a granted patent as PDF from ppubs.uspto.gov."""
        return await self.ppubs_client.download_patent_pdf(str(patent_number))

    # ---------------------------------------------------------------------
    # Tools for api.uspto.gov (metadata/search)
    # ---------------------------------------------------------------------

    async def get_app(self, app_num: str) -> Dict[str, Any]:
        """Get patent application data."""
        return await self.api_client.get_application(app_num)

    async def search_applications(
        self,
        q: Optional[str] = None,
        sort: Optional[str] = None,
        offset: Optional[int] = 0,
        limit: Optional[int] = 25,
        facets: Optional[str] = None,
        fields: Optional[str] = None,
        filters: Optional[str] = None,
        range_filters: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search patent applications by query parameters."""
        params = {
            "q": q,
            "sort": sort,
            "offset": offset,
            "limit": limit,
            "facets": facets,
            "fields": fields,
            "filters": filters,
            "rangeFilters": range_filters,
        }
        return await self.api_client.search_applications(**params)

    async def search_applications_post(
        self,
        q: Optional[str] = None,
        filters: Optional[List[Dict[str, Any]]] = None,
        range_filters: Optional[List[Dict[str, Any]]] = None,
        sort: Optional[List[Dict[str, Any]]] = None,
        fields: Optional[List[str]] = None,
        offset: Optional[int] = 0,
        limit: Optional[int] = 25,
        facets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search patent applications with JSON payload."""
        payload: Dict[str, Any] = {
            "q": q,
            "filters": filters,
            "rangeFilters": range_filters,
            "sort": sort,
            "fields": fields,
            "facets": facets,
            "pagination": {"offset": offset, "limit": limit},
        }
        return await self.api_client.search_applications_post(**payload)

    async def download_applications(
        self,
        q: Optional[str] = None,
        sort: Optional[str] = None,
        offset: Optional[int] = 0,
        limit: Optional[int] = 25,
        fields: Optional[str] = None,
        filters: Optional[str] = None,
        range_filters: Optional[str] = None,
        format: Optional[str] = "json",
    ) -> Dict[str, Any]:
        """Download patent applications by query parameters."""
        params = {
            "q": q,
            "sort": sort,
            "offset": offset,
            "limit": limit,
            "fields": fields,
            "filters": filters,
            "rangeFilters": range_filters,
            "format": format,
        }
        return await self.api_client.download_applications(**params)

    async def download_applications_post(
        self,
        q: Optional[str] = None,
        filters: Optional[List[Dict[str, Any]]] = None,
        range_filters: Optional[List[Dict[str, Any]]] = None,
        sort: Optional[List[Dict[str, Any]]] = None,
        fields: Optional[List[str]] = None,
        offset: Optional[int] = 0,
        limit: Optional[int] = 25,
        format: Optional[str] = "json",
    ) -> Dict[str, Any]:
        """Download patent applications with JSON payload."""
        payload: Dict[str, Any] = {
            "q": q,
            "filters": filters,
            "rangeFilters": range_filters,
            "sort": sort,
            "fields": fields,
            "format": format,
            "pagination": {"offset": offset, "limit": limit},
        }
        return await self.api_client.download_applications_post(**payload)

    async def get_app_metadata(self, app_num: str) -> Dict[str, Any]:
        """Get patent application metadata."""
        return await self.api_client.get_application_resource(app_num, "meta-data")

    async def get_app_adjustment(self, app_num: str) -> Dict[str, Any]:
        """Get patent term adjustment."""
        return await self.api_client.get_application_resource(app_num, "adjustment")

    async def get_app_assignment(self, app_num: str) -> Dict[str, Any]:
        """Get application assignment data."""
        return await self.api_client.get_application_resource(app_num, "assignment")

    async def get_app_attorney(self, app_num: str) -> Dict[str, Any]:
        """Get attorney/agent data."""
        return await self.api_client.get_application_resource(app_num, "attorney")

    async def get_app_continuity(self, app_num: str) -> Dict[str, Any]:
        """Get continuity data."""
        return await self.api_client.get_application_resource(app_num, "continuity")

    async def get_app_foreign_priority(self, app_num: str) -> Dict[str, Any]:
        """Get foreign priority data."""
        return await self.api_client.get_application_resource(app_num, "foreign-priority")

    async def get_app_transactions(self, app_num: str) -> Dict[str, Any]:
        """Get transaction data."""
        return await self.api_client.get_application_resource(app_num, "transactions")

    async def get_app_documents(self, app_num: str) -> Dict[str, Any]:
        """Get document details for an application."""
        return await self.api_client.get_application_resource(app_num, "documents")

    async def get_app_associated_documents(self, app_num: str) -> Dict[str, Any]:
        """Get associated documents metadata for an application."""
        return await self.api_client.get_application_resource(app_num, "associated-documents")

    async def get_status_codes(
        self,
        q: Optional[str] = None,
        offset: Optional[int] = 0,
        limit: Optional[int] = 25,
    ) -> Dict[str, Any]:
        """Search patent application status codes."""
        params = {"q": q, "offset": offset, "limit": limit}
        return await self.api_client.get_status_codes(**params)

    async def get_status_codes_post(
        self,
        q: Optional[str] = None,
        offset: Optional[int] = 0,
        limit: Optional[int] = 25,
    ) -> Dict[str, Any]:
        """Search patent application status codes (POST)."""
        payload = {"q": q, "pagination": {"offset": offset, "limit": limit}}
        return await self.api_client.get_status_codes_post(**payload)

    async def search_datasets(
        self,
        q: Optional[str] = None,
        product_title: Optional[str] = None,
        product_description: Optional[str] = None,
        product_short_name: Optional[str] = None,
        offset: Optional[int] = 0,
        limit: Optional[int] = 10,
        facets: Optional[str] = None,
        include_files: Optional[bool] = True,
        latest: Optional[bool] = False,
        labels: Optional[str] = None,
        categories: Optional[str] = None,
        datasets: Optional[str] = None,
        file_types: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search bulk datasets products."""
        params = {
            "q": q,
            "productTitle": product_title,
            "productDescription": product_description,
            "productShortName": product_short_name,
            "offset": offset,
            "limit": limit,
            "facets": facets,
            "includeFiles": include_files,
            "latest": latest,
            "labels": labels,
            "categories": categories,
            "datasets": datasets,
            "fileTypes": file_types,
        }
        return await self.api_client.search_datasets(**params)

    async def get_dataset_product(
        self,
        product_id: str,
        file_data_from_date: Optional[str] = None,
        file_data_to_date: Optional[str] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        include_files: Optional[bool] = None,
        latest: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Get a specific dataset product and optionally filter files."""
        params = {
            "fileDataFromDate": file_data_from_date,
            "fileDataToDate": file_data_to_date,
            "offset": offset,
            "limit": limit,
            "includeFiles": include_files,
            "latest": latest,
        }
        return await self.api_client.get_dataset_product(product_id, **params)


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: read secrets from env or config as needed
    import os
    from pydantic import SecretStr

    NGROK_TOKEN = os.getenv("NGROK_AUTHTOKEN")
    USPTO_API_KEY = os.getenv("USPTO_API_KEY")

    server = USPTOMCPServer(
        ngrok_auth_token=SecretStr(NGROK_TOKEN) if NGROK_TOKEN else None,
        uspto_api_key=SecretStr(USPTO_API_KEY) if USPTO_API_KEY else None,
    )

    try:
        url = server.run()
        print(f"MCP server '{server.name}' running at {server.host}:{server.port}")
        print(f"Public URL: {url}")
        # Keep main thread alive (Ctrl+C to stop)
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.stop()
        print("Stopped.")
