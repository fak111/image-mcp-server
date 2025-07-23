from typing import List, Dict, Any, Optional
from mcp.types import Tool, TextContent
from .image_handler import ImageHandler
from .api_client import InternVLClient, APIError


class MCPTools:
    def __init__(self):
        self.image_handler = ImageHandler()
        self.api_client = InternVLClient()

    def get_tools(self) -> List[Tool]:
        """Return list of all available MCP tools."""
        return [
            Tool(
                name="understand_visual",
                description="Intelligently analyze images in the context of software development. Automatically adapts analysis based on user intent - whether it's debugging errors, reviewing UI designs, comparing versions, extracting code/text, or understanding technical diagrams.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "images": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of image sources (file paths, URLs, or 'clipboard')",
                            "minItems": 1,
                            "maxItems": 5
                        },
                        "context": {
                            "type": "string",
                            "description": "What you're trying to accomplish or understand from these images. Be natural - describe your development need or question."
                        },
                        "focus": {
                            "type": "string",
                            "description": "Optional: specific aspect to emphasize (code, ui, errors, design, comparison, etc.)",
                            "examples": ["code analysis", "ui review", "error debugging", "design implementation", "version comparison", "text extraction"]
                        }
                    },
                    "required": ["images", "context"]
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Execute a tool with given arguments."""
        try:
            if name == "understand_visual":
                return await self._understand_visual(**arguments)
            else:
                return [TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error executing {name}: {str(e)}"
            )]

    async def _understand_visual(
        self,
        images: List[str],
        context: str,
        focus: Optional[str] = None,
        **kwargs
    ) -> List[TextContent]:
        """Intelligently analyze images based on development context."""
        try:
            # Process images
            processed_images = await self.image_handler.process_images(images)
            
            # Generate intelligent prompt based on context and focus
            prompt = self._create_smart_prompt(context, focus, len(images))
            
            # Stream response
            result_chunks = []
            async for chunk in self.api_client.analyze_image(
                processed_images, 
                prompt
            ):
                result_chunks.append(chunk)
            
            result = ''.join(result_chunks)
            return [TextContent(type="text", text=result)]
            
        except APIError as e:
            return [TextContent(type="text", text=f"API Error: {str(e)}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    def _create_smart_prompt(self, context: str, focus: Optional[str], image_count: int) -> str:
        """Create an intelligent prompt based on user context and focus."""
        
        # Base prompt that emphasizes development context
        base_prompt = f"""As a software development assistant, analyze this image to help with: {context}

Please provide a practical, actionable response that directly addresses the developer's need."""

        # Add focus-specific instructions
        if focus:
            focus_lower = focus.lower()
            if "code" in focus_lower:
                base_prompt += "\n\nFocus on: Code structure, syntax, logic, potential issues, and implementation details."
            elif "ui" in focus_lower or "design" in focus_lower:
                base_prompt += "\n\nFocus on: User interface elements, layout, design patterns, accessibility, and implementation suggestions."
            elif "error" in focus_lower or "debug" in focus_lower:
                base_prompt += "\n\nFocus on: Error messages, stack traces, debugging information, and potential solutions."
            elif "comparison" in focus_lower:
                base_prompt += "\n\nFocus on: Differences, similarities, changes, and their implications for development."
            elif "text" in focus_lower:
                base_prompt += "\n\nFocus on: Extracting and analyzing all visible text content accurately."
        
        # Add multi-image handling
        if image_count > 1:
            base_prompt += f"\n\nNote: Analyze all {image_count} images in relation to each other and the stated context."
        
        # Add development-specific context clues
        context_lower = context.lower()
        if any(keyword in context_lower for keyword in ["error", "bug", "issue", "problem", "fix"]):
            base_prompt += "\n\nThis appears to be a debugging scenario. Please identify the issue and suggest solutions."
        elif any(keyword in context_lower for keyword in ["ui", "interface", "design", "layout"]):
            base_prompt += "\n\nThis appears to be UI/design related. Consider usability, implementation feasibility, and best practices."
        elif any(keyword in context_lower for keyword in ["code", "implement", "function", "class"]):
            base_prompt += "\n\nThis appears to be code-related. Focus on code quality, patterns, and implementation guidance."
        elif any(keyword in context_lower for keyword in ["compare", "difference", "version", "change"]):
            base_prompt += "\n\nThis appears to be a comparison task. Highlight key differences and their implications."
        
        return base_prompt