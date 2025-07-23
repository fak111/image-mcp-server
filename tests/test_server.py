import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from mcp.types import CallToolRequest, ListToolsRequest, TextContent
from src.image_mcp.server import TermiVisServer

# Mark all tests in this module as unstable due to MCP framework complexity
pytestmark = pytest.mark.unstable


class TestTermiVisServer:
    @pytest.fixture
    def server(self, mock_api_key):
        return TermiVisServer()

    def test_server_initialization(self, server):
        """Test server initialization."""
        assert server.server is not None
        assert server.tools is not None

    @pytest.mark.asyncio
    async def test_handle_list_tools(self, server):
        """Test list tools handler."""
        request = ListToolsRequest(method="tools/list", params=None)

        with patch.object(server.tools, "get_tools") as mock_get_tools:
            mock_tools = [MagicMock(name="mock_tool")]
            mock_get_tools.return_value = mock_tools

            # Get the handler function
            handlers = server.server._tool_list_handlers
            assert len(handlers) == 1
            handler = handlers[0]

            result = await handler(request)
            assert result == mock_tools
            mock_get_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_call_tool_success(self, server):
        """Test successful tool call."""
        request = CallToolRequest(
            method="tools/call",
            params=MagicMock(
                name="analyze_image",
                arguments={"images": ["test.jpg"], "prompt": "Test prompt"},
            ),
        )

        mock_result = [TextContent(type="text", text="Success result")]

        with patch.object(
            server.tools, "call_tool", return_value=mock_result
        ) as mock_call_tool:
            # Get the handler function
            handlers = server.server._tool_call_handlers
            assert len(handlers) == 1
            handler = handlers[0]

            result = await handler(request)

            assert result == mock_result
            mock_call_tool.assert_called_once_with(
                "analyze_image", {"images": ["test.jpg"], "prompt": "Test prompt"}
            )

    @pytest.mark.asyncio
    async def test_handle_call_tool_error(self, server):
        """Test tool call with error."""
        request = CallToolRequest(
            method="tools/call",
            params=MagicMock(
                name="analyze_image",
                arguments={"images": ["test.jpg"], "prompt": "Test prompt"},
            ),
        )

        with patch.object(
            server.tools, "call_tool", side_effect=Exception("Test error")
        ) as mock_call_tool:
            # Get the handler function
            handlers = server.server._tool_call_handlers
            assert len(handlers) == 1
            handler = handlers[0]

            result = await handler(request)

            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert "Error executing tool: Test error" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_call_tool_no_arguments(self, server):
        """Test tool call with no arguments."""
        request = CallToolRequest(
            method="tools/call", params=MagicMock(name="analyze_image", arguments=None)
        )

        mock_result = [TextContent(type="text", text="Success with empty args")]

        with patch.object(
            server.tools, "call_tool", return_value=mock_result
        ) as mock_call_tool:
            # Get the handler function
            handlers = server.server._tool_call_handlers
            assert len(handlers) == 1
            handler = handlers[0]

            result = await handler(request)

            assert result == mock_result
            mock_call_tool.assert_called_once_with("analyze_image", {})

    @pytest.mark.asyncio
    async def test_run_server(self, server):
        """Test running the server."""
        mock_read_stream = MagicMock()
        mock_write_stream = MagicMock()

        with (
            patch("src.image_mcp.server.stdio_server") as mock_stdio,
            patch.object(server.server, "run") as mock_run,
            patch.object(
                server.server, "create_initialization_options", return_value={}
            ) as mock_init,
        ):
            # Setup the async context manager
            mock_stdio.return_value.__aenter__ = AsyncMock(
                return_value=(mock_read_stream, mock_write_stream)
            )
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_run.return_value = None

            await server.run()

            mock_stdio.assert_called_once()
            mock_run.assert_called_once_with(mock_read_stream, mock_write_stream, {})

    @pytest.mark.asyncio
    async def test_run_server_error(self, server):
        """Test server run with error."""
        with patch("src.image_mcp.server.stdio_server") as mock_stdio:
            mock_stdio.side_effect = Exception("Server startup error")

            with pytest.raises(Exception, match="Server startup error"):
                await server.run()


class TestMainFunction:
    @pytest.mark.asyncio
    async def test_main_success(self, mock_api_key):
        """Test successful main function execution."""
        with patch("src.image_mcp.server.TermiVisServer") as mock_server_class:
            mock_server = MagicMock()
            mock_server.run = AsyncMock()
            mock_server_class.return_value = mock_server

            from src.image_mcp.server import main

            result = await main()

            assert result == 0
            mock_server.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_no_api_key(self, monkeypatch):
        """Test main function without API key."""
        monkeypatch.delenv("INTERNVL_API_KEY", raising=False)

        from src.image_mcp.server import main

        result = await main()

        assert result == 1

    @pytest.mark.asyncio
    async def test_main_keyboard_interrupt(self, mock_api_key):
        """Test main function with keyboard interrupt."""
        with patch("src.image_mcp.server.TermiVisServer") as mock_server_class:
            mock_server = MagicMock()
            mock_server.run = AsyncMock(side_effect=KeyboardInterrupt())
            mock_server_class.return_value = mock_server

            from src.image_mcp.server import main

            result = await main()

            assert result == 0

    @pytest.mark.asyncio
    async def test_main_exception(self, mock_api_key):
        """Test main function with exception."""
        with patch("src.image_mcp.server.TermiVisServer") as mock_server_class:
            mock_server = MagicMock()
            mock_server.run = AsyncMock(side_effect=Exception("Fatal error"))
            mock_server_class.return_value = mock_server

            from src.image_mcp.server import main

            result = await main()

            assert result == 1
