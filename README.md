# TermiVis 🚀

**Visual intelligence for your terminal** - One-click setup for image analysis in Claude Code

[![PyPI version](https://badge.fury.io/py/termivls.svg)](https://badge.fury.io/py/termivls)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 🎯 One-Click Installation

Get started with TermiVis in under 30 seconds:

### Method 1: pipx (Recommended)
```bash
# Install TermiVis
pipx install termivls

# Setup with your API key
termivls setup --api-key YOUR_INTERNVL_API_KEY
```

### Method 2: One-Line Installation
```bash
curl -sSL https://get.termivls.com | bash -s -- --api-key YOUR_INTERNVL_API_KEY
```

### Method 3: Docker
```bash
docker run -e INTERNVL_API_KEY=YOUR_KEY -v ~/.config/claude-code:/app/.claude-config termivls/termivls
```

**That's it!** 🎉 TermiVis is now integrated with Claude Code.

## ✨ What You Get

- **🧠 Smart Image Analysis** - AI-powered understanding of screenshots, diagrams, code, and designs
- **🔄 Natural Integration** - Works seamlessly within Claude Code - no separate apps needed
- **⚡ Zero Configuration** - One command setup, automatic Claude Code integration
- **🛠️ Developer-Focused** - Optimized for debugging, UI review, code analysis, and design workflows

## 🚀 Instant Usage

After installation, just use Claude Code naturally:

```bash
# In Claude Code, simply ask:
"What's wrong with this error screenshot?" [attach image]
"How can I improve this UI design?" [attach design]
"What does this code do?" [attach code screenshot]
"Compare these two versions" [attach images]
```

No need to remember tool names or parameters - TermiVis automatically understands your intent!

## 📋 Available Commands

| Command | Description |
|---------|-------------|
| `termivls setup` | One-time setup with API key |
| `termivls status` | Check service health and configuration |
| `termivls run` | Run server manually (debugging) |
| `termivls uninstall` | Remove from Claude Code |
| `termivls --help` | Show all commands |

## 🔧 Advanced Installation

### From Source
```bash
git clone https://github.com/your-username/termivls
cd termivls
pip install -e .
termivls setup --api-key YOUR_KEY
```

### Development Setup
```bash
git clone https://github.com/your-username/termivls
cd termivls
uv sync
source .venv/bin/activate
termivls setup --api-key YOUR_KEY
```

## 📊 System Requirements

- **Python**: 3.10 or higher
- **OS**: macOS, Linux, Windows (WSL)
- **Claude Code**: Latest version
- **API Key**: InternVL API access

## 🛠️ How It Works

1. **Image Processing**: Handles PNG, JPG, WEBP, GIF, BMP from files, URLs, or clipboard
2. **Smart Analysis**: Uses InternVL AI to understand images in development context
3. **MCP Integration**: Registers as a standard MCP server in Claude Code
4. **Natural Interface**: No technical commands - just describe what you need

## 🔍 Troubleshooting

### Quick Health Check
```bash
termivls status
```

### Common Issues

**"Command not found"**
```bash
# Restart terminal or:
source ~/.bashrc  # Linux/macOS
# Or add to PATH: ~/.local/bin
```

**"API key invalid"**
```bash
# Re-setup with correct key:
termivls setup --api-key YOUR_CORRECT_KEY --force
```

**"Server not responding"**
```bash
# Check logs and restart:
termivls run
```

### Getting Help
- **Status Check**: `termivls status`
- **Logs**: `termivls run` (shows real-time logs)
- **GitHub Issues**: [Report bugs](https://github.com/your-username/termivls/issues)

## 🎨 Use Cases

### 🐛 Debug Errors
```
"This Python error is confusing, what's the actual problem?"
[attach error screenshot]
```

### 🎨 Review UI/Design
```
"What's wrong with this interface from a UX perspective?"
[attach UI mockup]
```

### 💻 Analyze Code
```
"Can you explain this algorithm and suggest improvements?"
[attach code screenshot]
```

### 🔄 Compare Versions
```
"What changed between these two versions?"
[attach before/after images]
```

## 🚢 Deployment Options

### Personal Use
```bash
pipx install termivls
termivls setup --api-key YOUR_KEY
```

### Team/Organization
```bash
# Docker Compose
curl -O https://raw.githubusercontent.com/your-username/termivls/main/docker-compose.yml
INTERNVL_API_KEY=your_key docker-compose up -d
```

### CI/CD Integration
```yaml
- name: Install TermiVis
  run: pipx install termivls
- name: Setup TermiVis
  run: termivls setup --api-key ${{ secrets.INTERNVL_API_KEY }}
```

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Setup development environment
git clone https://github.com/your-username/termivls
cd termivls
uv sync
source .venv/bin/activate

# Run tests
pytest tests/
python examples/test_basic_functionality.py

# Submit PR
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **InternVL Team** - For the powerful vision-language model
- **Anthropic** - For Claude Code and MCP protocol
- **Python Community** - For the amazing ecosystem

---

**Made with ❤️ for developers who love efficient workflows**

[🌟 Star us on GitHub](https://github.com/your-username/termivls) • [📚 Documentation](https://github.com/your-username/termivls/wiki) • [🐛 Report Issues](https://github.com/your-username/termivls/issues)