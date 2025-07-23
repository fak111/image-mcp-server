import asyncio
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    ListToolsRequest,
    TextContent,
)
from .tools import MCPTools
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TermiVisServer:
    def __init__(self):
        self.server = Server("termivls")
        self.tools = MCPTools()
        self.setup_handlers()

    def setup_handlers(self):
        """Setup MCP server handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools(request: ListToolsRequest) -> list:
            """Handle list tools request."""
            return self.tools.get_tools()

        @self.server.call_tool()
        async def handle_call_tool(request: CallToolRequest) -> list:
            """Handle tool call request."""
            logger.info(f"Calling tool: {request.params.name} with args: {request.params.arguments}")
            
            try:
                result = await self.tools.call_tool(
                    request.params.name,
                    request.params.arguments or {}
                )
                return result
            except Exception as e:
                logger.error(f"Error calling tool {request.params.name}: {str(e)}")
                return [TextContent(
                    type="text",
                    text=f"Error executing tool: {str(e)}"
                )]

    async def run(self):
        """Run the MCP server."""
        logger.info("Starting TermiVis MCP Server")
        logger.info(f"Using model: {settings.default_model}")
        logger.info(f"Max images per request: {settings.max_images_per_request}")
        
        try:
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
            raise


async def main():
    """Main entry point."""
    try:
        # Validate configuration
        if not settings.internvl_api_key:
            logger.error("INTERNVL_API_KEY environment variable is required")
            return 1
        
        server = TermiVisServer()
        await server.run()
        return 0
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())