import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from mcp.types import TextContent
from src.image_mcp.tools import MCPTools


class TestMCPTools:
    
    @pytest.fixture
    def tools(self, mock_api_key):
        return MCPTools()
    
    def test_get_tools(self, tools):
        """Test getting list of available tools."""
        tool_list = tools.get_tools()
        
        assert len(tool_list) == 5
        tool_names = [tool.name for tool in tool_list]
        
        expected_tools = [
            "analyze_image",
            "describe_image", 
            "extract_text",
            "compare_images",
            "identify_objects"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names
    
    def test_analyze_image_tool_schema(self, tools):
        """Test analyze_image tool schema."""
        tool_list = tools.get_tools()
        analyze_tool = next(tool for tool in tool_list if tool.name == "analyze_image")
        
        schema = analyze_tool.inputSchema
        assert "images" in schema["properties"]
        assert "prompt" in schema["properties"]
        assert schema["properties"]["images"]["minItems"] == 1
        assert schema["properties"]["images"]["maxItems"] == 5
        assert "images" in schema["required"]
        assert "prompt" in schema["required"]
    
    def test_compare_images_tool_schema(self, tools):
        """Test compare_images tool schema."""
        tool_list = tools.get_tools()
        compare_tool = next(tool for tool in tool_list if tool.name == "compare_images")
        
        schema = compare_tool.inputSchema
        assert schema["properties"]["images"]["minItems"] == 2
        assert schema["properties"]["images"]["maxItems"] == 5
    
    @pytest.mark.asyncio
    async def test_call_unknown_tool(self, tools):
        """Test calling an unknown tool."""
        result = await tools.call_tool("unknown_tool", {})
        
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Unknown tool: unknown_tool" in result[0].text
    
    @pytest.mark.asyncio
    async def test_analyze_image_success(self, tools, sample_image_path):
        """Test successful analyze_image call."""
        mock_processed_images = ["data:image/png;base64,mock_data"]
        mock_result = ["This is a test analysis"]
        
        with patch.object(tools.image_handler, 'process_images', return_value=mock_processed_images), \
             patch.object(tools.api_client, 'analyze_image', return_value=iter(mock_result)):
            
            arguments = {
                "images": [sample_image_path],
                "prompt": "What do you see?"
            }
            
            result = await tools.call_tool("analyze_image", arguments)
            
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert result[0].text == "This is a test analysis"
    
    @pytest.mark.asyncio
    async def test_analyze_image_error(self, tools):
        """Test analyze_image with error."""
        with patch.object(tools.image_handler, 'process_images', side_effect=Exception("Test error")):
            
            arguments = {
                "images": ["test.jpg"],
                "prompt": "What do you see?"
            }
            
            result = await tools.call_tool("analyze_image", arguments)
            
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert "Error: Test error" in result[0].text
    
    @pytest.mark.asyncio
    async def test_describe_image_success(self, tools, sample_image_path):
        """Test successful describe_image call."""
        mock_processed_images = ["data:image/png;base64,mock_data"]
        mock_result = ["A detailed image description"]
        
        with patch.object(tools.image_handler, 'process_images', return_value=mock_processed_images), \
             patch.object(tools.api_client, 'describe_image', return_value=iter(mock_result)):
            
            arguments = {
                "images": [sample_image_path],
                "detail_level": "detailed"
            }
            
            result = await tools.call_tool("describe_image", arguments)
            
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert result[0].text == "A detailed image description"
    
    @pytest.mark.asyncio
    async def test_extract_text_success(self, tools, sample_image_path):
        """Test successful extract_text call."""
        mock_processed_images = ["data:image/png;base64,mock_data"]
        mock_result = ["Extracted text content"]
        
        with patch.object(tools.image_handler, 'process_images', return_value=mock_processed_images), \
             patch.object(tools.api_client, 'extract_text', return_value=iter(mock_result)):
            
            arguments = {
                "images": [sample_image_path]
            }
            
            result = await tools.call_tool("extract_text", arguments)
            
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert result[0].text == "Extracted text content"
    
    @pytest.mark.asyncio
    async def test_compare_images_success(self, tools, sample_image_path):
        """Test successful compare_images call."""
        mock_processed_images = ["data:image/png;base64,mock_data1", "data:image/png;base64,mock_data2"]
        mock_result = ["Images comparison result"]
        
        with patch.object(tools.image_handler, 'process_images', return_value=mock_processed_images), \
             patch.object(tools.api_client, 'compare_images', return_value=iter(mock_result)):
            
            arguments = {
                "images": [sample_image_path, sample_image_path]
            }
            
            result = await tools.call_tool("compare_images", arguments)
            
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert result[0].text == "Images comparison result"
    
    @pytest.mark.asyncio
    async def test_compare_images_insufficient(self, tools):
        """Test compare_images with insufficient images."""
        arguments = {
            "images": ["single_image.jpg"]
        }
        
        result = await tools.call_tool("compare_images", arguments)
        
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "At least 2 images required" in result[0].text
    
    @pytest.mark.asyncio
    async def test_identify_objects_success(self, tools, sample_image_path):
        """Test successful identify_objects call."""
        mock_processed_images = ["data:image/png;base64,mock_data"]
        mock_result = ["Objects found: rectangle, circle"]
        
        with patch.object(tools.image_handler, 'process_images', return_value=mock_processed_images), \
             patch.object(tools.api_client, 'identify_objects', return_value=iter(mock_result)):
            
            arguments = {
                "images": [sample_image_path],
                "include_locations": True
            }
            
            result = await tools.call_tool("identify_objects", arguments)
            
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert result[0].text == "Objects found: rectangle, circle"
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, tools, sample_image_path):
        """Test API error handling."""
        from src.image_mcp.api_client import APIError
        
        mock_processed_images = ["data:image/png;base64,mock_data"]
        
        with patch.object(tools.image_handler, 'process_images', return_value=mock_processed_images), \
             patch.object(tools.api_client, 'analyze_image', side_effect=APIError("API failed")):
            
            arguments = {
                "images": [sample_image_path],
                "prompt": "Test prompt"
            }
            
            result = await tools.call_tool("analyze_image", arguments)
            
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert "API Error: API failed" in result[0].text