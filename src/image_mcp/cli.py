#!/usr/bin/env python3
"""
TermiVis CLI - One-click setup for image analysis in terminal
"""

import os
import sys
import json
import subprocess
import typer
from pathlib import Path
from typing import Optional
import platform

from .config import Settings
from . import __version__

app = typer.Typer(
    name="termivls",
    help="TermiVis: Visual intelligence for your terminal",
    no_args_is_help=True
)


def get_claude_config_path() -> Path:
    """Get Claude Code configuration directory path."""
    system = platform.system().lower()
    if system == "darwin":  # macOS
        return Path.home() / ".config" / "claude-code"
    elif system == "linux":
        return Path.home() / ".config" / "claude-code"
    elif system == "windows":
        return Path.home() / "AppData" / "Roaming" / "claude-code"
    else:
        return Path.home() / ".config" / "claude-code"


def ensure_claude_config_dir() -> Path:
    """Ensure Claude Code configuration directory exists."""
    config_path = get_claude_config_path()
    config_path.mkdir(parents=True, exist_ok=True)
    return config_path


def get_termivls_executable() -> str:
    """Get the path to the TermiVis server executable."""
    # Try to find the current Python executable with the module
    python_exe = sys.executable
    return f"{python_exe} -m image_mcp.server"


def is_service_running() -> bool:
    """Check if TermiVis service is already running."""
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return "image_mcp.server" in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def validate_api_key(api_key: str) -> bool:
    """Validate API key format (basic check)."""
    if not api_key or len(api_key.strip()) < 10:
        return False
    return True


@app.command()
def setup(
    api_key: str = typer.Option(..., "--api-key", "-k", help="Your InternVL API key"),
    server_name: str = typer.Option("termivls", "--name", "-n", help="MCP server name"),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite existing configuration")
):
    """Set up TermiVis with Claude Code - one command setup!"""
    
    typer.echo("🚀 Setting up TermiVis for Claude Code...")
    
    # Validate API key
    if not validate_api_key(api_key):
        typer.echo("❌ Invalid API key format. Please provide a valid InternVL API key.", err=True)
        raise typer.Exit(1)
    
    # Set up environment
    env_path = Path.cwd() / ".env"
    if env_path.exists() and not force:
        typer.echo(f"⚠️  .env file already exists. Use --force to overwrite.")
    else:
        with open(env_path, "w") as f:
            f.write(f"INTERNVL_API_KEY={api_key}\n")
        typer.echo("✅ Created .env file with API key")
    
    # Ensure Claude Code config directory
    config_dir = ensure_claude_config_dir()
    mcp_config_file = config_dir / "mcp_servers.json"
    
    # Read existing MCP configuration
    mcp_config = {"mcpServers": {}}
    if mcp_config_file.exists():
        try:
            with open(mcp_config_file, "r") as f:
                mcp_config = json.load(f)
        except json.JSONDecodeError:
            typer.echo("⚠️  Existing MCP config is corrupted, creating new one...")
    
    # Check if server already exists
    if server_name in mcp_config.get("mcpServers", {}) and not force:
        typer.echo(f"⚠️  MCP server '{server_name}' already exists. Use --force to overwrite.")
    else:
        # Add TermiVis server configuration
        executable_cmd = get_termivls_executable()
        mcp_config["mcpServers"][server_name] = {
            "command": executable_cmd.split()[0],
            "args": executable_cmd.split()[1:],
            "cwd": str(Path.cwd()),
            "env": {"INTERNVL_API_KEY": api_key}
        }
        
        # Write updated configuration
        with open(mcp_config_file, "w") as f:
            json.dump(mcp_config, f, indent=2)
        
        typer.echo(f"✅ Added '{server_name}' to Claude Code MCP configuration")
    
    typer.echo("\n🎉 Setup complete!")
    typer.echo("\n📋 Next steps:")
    typer.echo("1. Restart Claude Code to load the new MCP server")
    typer.echo(f"2. The server '{server_name}' is now available in Claude Code")
    typer.echo("3. Try: 'What's in this image?' with any image")
    typer.echo("\n💡 Use 'termivls status' to check server health")


@app.command()
def run():
    """Run TermiVis server directly."""
    typer.echo("🚀 Starting TermiVis server...")
    
    # Check if API key is configured
    try:
        settings = Settings()
        if not settings.internvl_api_key:
            typer.echo("❌ No API key found. Run 'termivls setup --api-key YOUR_KEY' first.", err=True)
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Configuration error: {str(e)}", err=True)
        raise typer.Exit(1)
    
    # Import and run server
    try:
        from .server import main
        import asyncio
        
        typer.echo("✅ Configuration validated")
        typer.echo("🔄 Starting MCP server...")
        
        exit_code = asyncio.run(main())
        if exit_code != 0:
            raise typer.Exit(exit_code)
            
    except KeyboardInterrupt:
        typer.echo("\n⏹️  Server stopped by user")
    except Exception as e:
        typer.echo(f"❌ Server error: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def status():
    """Check TermiVis service status and configuration."""
    typer.echo("🔍 Checking TermiVis status...\n")
    
    # Check API key configuration
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        typer.echo("✅ .env file found")
        try:
            settings = Settings()
            if settings.internvl_api_key:
                key_preview = settings.internvl_api_key[:10] + "..." + settings.internvl_api_key[-4:]
                typer.echo(f"✅ API key configured: {key_preview}")
            else:
                typer.echo("❌ No API key in configuration")
        except Exception as e:
            typer.echo(f"❌ Configuration error: {str(e)}")
    else:
        typer.echo("❌ No .env file found")
    
    # Check Claude Code MCP configuration
    config_dir = get_claude_config_path()
    mcp_config_file = config_dir / "mcp_servers.json"
    
    if mcp_config_file.exists():
        try:
            with open(mcp_config_file, "r") as f:
                mcp_config = json.load(f)
            
            termivls_servers = [
                name for name, config in mcp_config.get("mcpServers", {}).items()
                if "image_mcp.server" in str(config)
            ]
            
            if termivls_servers:
                typer.echo(f"✅ Claude Code MCP configured: {', '.join(termivls_servers)}")
            else:
                typer.echo("❌ No TermiVis server found in Claude Code MCP config")
                
        except Exception as e:
            typer.echo(f"❌ MCP config error: {str(e)}")
    else:
        typer.echo("❌ No Claude Code MCP configuration found")
    
    # Check if service is running
    if is_service_running():
        typer.echo("✅ TermiVis server is running")
    else:
        typer.echo("⏹️  TermiVis server is not running")
    
    # Show available tools
    try:
        from .tools import MCPTools
        tools = MCPTools()
        tool_list = tools.get_tools()
        typer.echo(f"\n🛠️  Available tools: {len(tool_list)}")
        for tool in tool_list:
            typer.echo(f"   • {tool.name}: {tool.description[:60]}...")
    except Exception as e:
        typer.echo(f"❌ Could not load tools: {str(e)}")


@app.command()
def version():
    """Show TermiVis version information."""
    typer.echo(f"TermiVis v{__version__}")
    typer.echo("Visual intelligence for your terminal")
    typer.echo("https://github.com/your-username/image-mcp-server")


@app.command()
def uninstall(
    server_name: str = typer.Option("termivls", "--name", "-n", help="MCP server name to remove"),
    remove_env: bool = typer.Option(False, "--remove-env", help="Also remove .env file")
):
    """Remove TermiVis from Claude Code configuration."""
    typer.echo("🗑️  Uninstalling TermiVis...")
    
    # Remove from Claude Code MCP configuration
    config_dir = get_claude_config_path()
    mcp_config_file = config_dir / "mcp_servers.json"
    
    if mcp_config_file.exists():
        try:
            with open(mcp_config_file, "r") as f:
                mcp_config = json.load(f)
            
            if server_name in mcp_config.get("mcpServers", {}):
                del mcp_config["mcpServers"][server_name]
                
                with open(mcp_config_file, "w") as f:
                    json.dump(mcp_config, f, indent=2)
                
                typer.echo(f"✅ Removed '{server_name}' from Claude Code MCP configuration")
            else:
                typer.echo(f"⚠️  Server '{server_name}' not found in MCP configuration")
                
        except Exception as e:
            typer.echo(f"❌ Error updating MCP config: {str(e)}")
    
    # Optionally remove .env file
    if remove_env:
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            env_path.unlink()
            typer.echo("✅ Removed .env file")
    
    typer.echo("✅ Uninstall complete!")
    typer.echo("💡 You may need to restart Claude Code to complete the removal")


def main():
    """Main CLI entry point."""
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        typer.echo(f"❌ Unexpected error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()