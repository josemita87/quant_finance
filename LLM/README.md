# USPTO Patent MCP Server with Anthropic Integration

A Model Context Protocol (MCP) server that provides access to USPTO patent data with integrated Anthropic API support for LLM interactions.

## Features

- **USPTO Patent Search**: Full-text patent search via ppubs.uspto.gov
- **Patent Metadata**: Application data, assignments, transactions via api.uspto.gov
- **PDF Downloads**: Download patent documents as PDFs
- **Anthropic Integration**: Seamless LLM interactions with Claude
- **Ngrok Support**: Public tunnel access for remote connections
- **Secure Credentials**: SecretStr handling for all sensitive data

## Quick Start

### 1. Setup Credentials

Copy the credentials template and fill in your API keys:

```bash
cp .credentials.env.example .credentials.env
```

Edit `.credentials.env` with your actual credentials:

```env
# Required
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional
USPTO_API_KEY=your_uspto_api_key_here
NGROK_AUTH_TOKEN=your_ngrok_auth_token_here

# Configuration
MCP_USE_NGROK=false
MCP_HOST=localhost
MCP_PORT=8000
ANTHROPIC_VERSION=2023-06-01
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Run the Server

```bash
uv run patent-mcp-server
```

## Usage

### Basic Usage

The server automatically:
1. Loads credentials from `.credentials.env`
2. Initializes the USPTO MCP server
3. Initializes the Anthropic API adapter
4. Starts the MCP server
5. Provides the server URL for client access

### With Ngrok Tunnel

To expose the server publicly via ngrok:

1. Set `MCP_USE_NGROK=true` in `.credentials.env`
2. Add your `NGROK_AUTH_TOKEN` for authenticated tunnels
3. Run the server - it will automatically create a public tunnel

### API Integration

The server provides 23 MCP tools for patent search and analysis:

**Patent Search Tools:**
- `ppubs_search_patents` - Search granted patents
- `ppubs_search_applications` - Search patent applications
- `ppubs_get_patent_by_number` - Get patent by number
- `ppubs_download_patent_pdf` - Download patent PDF

**Metadata Tools:**
- `get_app` - Get application data
- `search_applications` - Search applications
- `get_app_metadata` - Get application metadata
- `get_app_assignment` - Get assignment data
- And 15+ more metadata tools

## Architecture

```
src/patent_mcp_server/
├── main.py                 # Main entry point
├── server/main.py          # USPTO MCP Server
├── llm/anthropic/api.py    # Anthropic API integration
└── server/api/             # USPTO API clients
    ├── ppub_uspto.py       # ppubs.uspto.gov client
    └── uspto.py            # api.uspto.gov client
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | Anthropic API key |
| `USPTO_API_KEY` | No | - | USPTO API key for enhanced rate limits |
| `NGROK_AUTH_TOKEN` | No | - | Ngrok auth token for authenticated tunnels |
| `MCP_USE_NGROK` | No | false | Enable ngrok tunnel |
| `MCP_HOST` | No | localhost | Server host |
| `MCP_PORT` | No | 8000 | Server port |
| `ANTHROPIC_VERSION` | No | 2023-06-01 | Anthropic API version |

### Server Features

- **Automatic Startup**: MCP server starts automatically when needed
- **URL Exposure**: Get server URL for client connections
- **Tunnel Management**: Automatic ngrok tunnel setup and cleanup
- **Credential Security**: All sensitive data handled with SecretStr
- **Error Handling**: Comprehensive error handling and logging

## Development

### Running Tests

```bash
uv run pytest
```

### Code Structure

- `src/patent_mcp_server/main.py` - Main application entry point
- `src/patent_mcp_server/server/main.py` - USPTO MCP Server implementation
- `src/patent_mcp_server/llm/anthropic/api.py` - Anthropic API integration
- `src/patent_mcp_server/server/api/` - USPTO API client implementations

## License

This project is licensed under the MIT License.
