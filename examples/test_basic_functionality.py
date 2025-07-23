#!/usr/bin/env python3
"""
Basic functionality test for TermiVis
This script tests core functionality without requiring actual API calls.
"""

import sys
import os
import tempfile
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from image_mcp.image_handler import ImageHandler
from image_mcp.tools import MCPTools


def create_test_image():
    """Create a simple test image."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        image = Image.new('RGB', (200, 100), color='red')
        # Add some simple pattern
        for x in range(50, 150):
            for y in range(25, 75):
                if (x + y) % 20 < 10:
                    image.putpixel((x, y), (0, 0, 255))  # Blue pixels
        image.save(f.name, 'PNG')
        return f.name


async def test_image_processing():
    """Test image processing functionality."""
    print("🧪 Testing image processing...")
    
    handler = ImageHandler()
    test_image_path = create_test_image()
    
    try:
        # Test single image processing
        result = await handler._process_single_image(test_image_path)
        assert result.startswith("data:image/png;base64,")
        print("✅ Single image processing works")
        
        # Test multiple image processing
        results = await handler.process_images([test_image_path])
        assert len(results) == 1
        assert results[0].startswith("data:image/png;base64,")
        print("✅ Multiple image processing works")
        
        # Test MIME type detection
        mime_type = handler._get_mime_type("PNG")
        assert mime_type == "image/png"
        print("✅ MIME type detection works")
        
    finally:
        os.unlink(test_image_path)


def test_tools_registration():
    """Test that all tools are properly registered."""
    print("🧪 Testing tools registration...")
    
    tools = MCPTools()
    tool_list = tools.get_tools()
    
    expected_tools = {
        "understand_visual"
    }
    
    actual_tools = {tool.name for tool in tool_list}
    
    assert expected_tools == actual_tools, f"Expected {expected_tools}, got {actual_tools}"
    print("✅ All tools are registered correctly")
    
    # Test tool schemas
    for tool in tool_list:
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert hasattr(tool, 'inputSchema')
        assert 'properties' in tool.inputSchema
    
    print("✅ All tool schemas are valid")


def test_url_detection():
    """Test URL detection functionality."""
    print("🧪 Testing URL detection...")
    
    handler = ImageHandler()
    
    # Test valid URLs
    assert handler._is_url("https://example.com/image.jpg") == True
    assert handler._is_url("http://example.com/image.png") == True
    
    # Test invalid URLs
    assert handler._is_url("/local/path/image.jpg") == False
    assert handler._is_url("image.png") == False
    assert handler._is_url("not_a_url") == False
    
    print("✅ URL detection works correctly")


async def main():
    """Run all basic functionality tests."""
    print("🚀 Starting TermiVis Basic Functionality Tests\n")
    
    try:
        # Test URL detection (synchronous)
        test_url_detection()
        
        # Test tools registration (synchronous)
        test_tools_registration()
        
        # Test image processing (asynchronous)
        await test_image_processing()
        
        print("\n🎉 All basic functionality tests passed!")
        print("✅ TermiVis is ready for use!")
        
        # Show next steps
        print("\n📋 Next Steps:")
        print("1. Configure your .env file with INTERNVL_API_KEY")
        print("2. Run: uv run python -m src.image_mcp.server")
        print("3. Configure your MCP client to use this server")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)