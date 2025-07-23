import pytest
import os
import tempfile
from PIL import Image
from unittest.mock import patch, AsyncMock
from src.image_mcp.tools import MCPTools


class TestIntegration:
    """Integration tests that test the full workflow."""

    @pytest.fixture
    def tools(self, mock_api_key):
        return MCPTools()

    @pytest.fixture
    def test_image_with_text(self):
        """Create a test image with text for OCR testing."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create an image with some text
            image = Image.new("RGB", (200, 100), color="white")
            # Note: This is a simple colored image, not actual text
            # In real scenarios, you'd use PIL.ImageDraw to add text
            image.save(f.name, "PNG")
            yield f.name
        os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_full_image_analysis_workflow(
        self, tools, sample_image_path, make_async_iterator
    ):
        """Test complete image analysis workflow."""
        # Mock the API response
        mock_api_response = [
            "This image shows a red colored rectangular shape. ",
            "The image appears to be a simple geometric form ",
            "with solid coloring and clean edges.",
        ]

        with patch.object(
            tools.api_client,
            "analyze_image",
            return_value=make_async_iterator(mock_api_response),
        ):
            arguments = {
                "images": [sample_image_path],
                "context": "Describe what you see in this image in detail.",
                "focus": "general analysis",
            }

            result = await tools.call_tool("understand_visual", arguments)

            assert len(result) == 1
            assert "red colored rectangular shape" in result[0].text
            assert "geometric form" in result[0].text

    @pytest.mark.asyncio
    async def test_full_image_description_workflow(
        self, tools, sample_image_path, make_async_iterator
    ):
        """Test complete image description workflow."""
        mock_api_response = [
            "The image contains a simple red rectangle ",
            "positioned against a transparent background. ",
            "The shape is uniform in color and has sharp, clean edges.",
        ]

        with patch.object(
            tools.api_client,
            "analyze_image",
            return_value=make_async_iterator(mock_api_response),
        ):
            arguments = {
                "images": [sample_image_path],
                "context": "Describe this image in detail",
                "focus": "detailed description",
            }

            result = await tools.call_tool("understand_visual", arguments)

            assert len(result) == 1
            full_text = result[0].text
            assert "red rectangle" in full_text
            assert "clean edges" in full_text

    @pytest.mark.asyncio
    async def test_full_text_extraction_workflow(
        self, tools, test_image_with_text, make_async_iterator
    ):
        """Test complete text extraction workflow."""
        mock_api_response = ["No visible text found in this image."]

        with patch.object(
            tools.api_client,
            "analyze_image",
            return_value=make_async_iterator(mock_api_response),
        ):
            arguments = {
                "images": [test_image_with_text],
                "context": "Extract any text from this image",
                "focus": "text extraction",
            }

            result = await tools.call_tool("understand_visual", arguments)

            assert len(result) == 1
            assert "No visible text found" in result[0].text

    @pytest.mark.asyncio
    async def test_full_image_comparison_workflow(
        self, tools, sample_image_path, make_async_iterator
    ):
        """Test complete image comparison workflow."""
        # Create a second test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image = Image.new("RGB", (100, 100), color="blue")
            image.save(f.name, "PNG")
            second_image_path = f.name

        try:
            mock_api_response = [
                "Comparing the two images: ",
                "The first image shows a red rectangle, ",
                "while the second image shows a blue rectangle. ",
                "Both images have the same size and shape, ",
                "but differ in color.",
            ]

            with patch.object(
                tools.api_client,
                "analyze_image",
                return_value=make_async_iterator(mock_api_response),
            ):
                arguments = {
                    "images": [sample_image_path, second_image_path],
                    "context": "Compare these two images and highlight the differences",
                    "focus": "colors",
                }

                result = await tools.call_tool("understand_visual", arguments)

                assert len(result) == 1
                full_text = result[0].text
                assert "red rectangle" in full_text
                assert "blue rectangle" in full_text
                assert "differ in color" in full_text

        finally:
            os.unlink(second_image_path)

    @pytest.mark.asyncio
    async def test_full_object_identification_workflow(
        self, tools, sample_image_path, make_async_iterator
    ):
        """Test complete object identification workflow."""
        mock_api_response = [
            "Objects identified in the image:\n",
            "1. Rectangular shape - located in the center\n",
            "2. Solid color fill - covering the entire rectangle\n",
            "3. Clean geometric edges - defining the shape boundaries",
        ]

        with patch.object(
            tools.api_client,
            "analyze_image",
            return_value=make_async_iterator(mock_api_response),
        ):
            arguments = {
                "images": [sample_image_path],
                "context": "Identify and locate objects in this image",
                "focus": "object identification",
            }

            result = await tools.call_tool("understand_visual", arguments)

            assert len(result) == 1
            full_text = result[0].text
            assert "Rectangular shape" in full_text
            assert "located in the center" in full_text
            assert "geometric edges" in full_text

    @pytest.mark.asyncio
    async def test_multiple_images_workflow(
        self, tools, sample_image_path, make_async_iterator
    ):
        """Test workflow with multiple images."""
        # Create additional test images
        test_images = [sample_image_path]
        temp_files = []

        try:
            # Create 3 more test images
            for i, color in enumerate(["green", "yellow", "purple"]):
                with tempfile.NamedTemporaryFile(
                    suffix=f"_{color}.png", delete=False
                ) as f:
                    image = Image.new("RGB", (100, 100), color=color)
                    image.save(f.name, "PNG")
                    test_images.append(f.name)
                    temp_files.append(f.name)

            mock_api_response = [
                "Analysis of multiple images:\n",
                "Image 1: Red rectangle\n",
                "Image 2: Green rectangle\n",
                "Image 3: Yellow rectangle\n",
                "Image 4: Purple rectangle\n",
                "All images share the same rectangular shape but differ in color.",
            ]

            with patch.object(
                tools.api_client,
                "analyze_image",
                return_value=make_async_iterator(mock_api_response),
            ):
                arguments = {
                    "images": test_images,
                    "context": "Analyze and compare all these images",
                    "focus": "comparison",
                }

                result = await tools.call_tool("understand_visual", arguments)

                assert len(result) == 1
                full_text = result[0].text
                assert "Red rectangle" in full_text
                assert "Green rectangle" in full_text
                assert "Yellow rectangle" in full_text
                assert "Purple rectangle" in full_text
                assert "same rectangular shape" in full_text

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                os.unlink(temp_file)

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, tools):
        """Test error handling and recovery in workflows."""
        # Test with invalid image path
        arguments = {
            "images": ["/nonexistent/path/image.jpg"],
            "context": "Describe this image",
            "focus": "general",
        }

        result = await tools.call_tool("understand_visual", arguments)

        assert len(result) == 1
        assert "Error:" in result[0].text
        # Should handle the file not found error gracefully

    @pytest.mark.asyncio
    async def test_large_image_compression_workflow(
        self, tools, large_image_path, make_async_iterator
    ):
        """Test workflow with large image that needs compression."""
        mock_api_response = [
            "This is a large green image that has been automatically ",
            "compressed for efficient processing. The image maintains ",
            "its visual quality while being optimized for analysis.",
        ]

        with patch.object(
            tools.api_client,
            "analyze_image",
            return_value=make_async_iterator(mock_api_response),
        ):
            arguments = {
                "images": [large_image_path],
                "context": "Analyze this large image",
                "focus": "compression handling",
            }

            result = await tools.call_tool("understand_visual", arguments)

            assert len(result) == 1
            full_text = result[0].text
            assert "large green image" in full_text
            assert "compressed" in full_text
            assert "optimized" in full_text
