#!/bin/bash
# TermiVis One-Click Installation Script
# Usage: curl -sSL https://get.termivls.com | bash -s -- --api-key YOUR_API_KEY

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
API_KEY=""
INSTALL_METHOD="pipx"
FORCE=false
QUIET=false

# Helper functions
log_info() {
    if [ "$QUIET" = false ]; then
        echo -e "${BLUE}ℹ️  $1${NC}"
    fi
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}" >&2
}

show_usage() {
    cat << EOF
TermiVis One-Click Installation Script

Usage: $0 [OPTIONS]

Options:
    --api-key KEY       Your InternVL API key (required)
    --method METHOD     Installation method: pipx, pip, or docker (default: pipx)
    --force            Force overwrite existing installation
    --quiet            Suppress non-essential output
    --help             Show this help message

Examples:
    # Install with pipx (recommended)
    $0 --api-key YOUR_API_KEY
    
    # Install with pip
    $0 --api-key YOUR_API_KEY --method pip
    
    # Install with Docker
    $0 --api-key YOUR_API_KEY --method docker

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --method)
            INSTALL_METHOD="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate API key
if [ -z "$API_KEY" ]; then
    log_error "API key is required"
    show_usage
    exit 1
fi

if [ ${#API_KEY} -lt 10 ]; then
    log_error "API key seems too short. Please check your InternVL API key."
    exit 1
fi

# Detect OS
OS=$(uname -s)
case $OS in
    Darwin)
        OS_NAME="macOS"
        ;;
    Linux)
        OS_NAME="Linux"
        ;;
    MINGW*|CYGWIN*|MSYS*)
        OS_NAME="Windows"
        ;;
    *)
        log_error "Unsupported operating system: $OS"
        exit 1
        ;;
esac

log_info "Detected OS: $OS_NAME"

# Check Python version
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is required but not installed"
    log_info "Please install Python 3.10 or higher from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
log_info "Python version: $PYTHON_VERSION"

# Check if Python version is >= 3.10
if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)'; then
    log_info "Python version check passed"
else
    log_error "Python 3.10 or higher is required, found $PYTHON_VERSION"
    exit 1
fi

# Installation methods
install_with_pipx() {
    log_info "Installing TermiVis with pipx..."
    
    # Check if pipx is installed
    if ! command -v pipx &> /dev/null; then
        log_info "pipx not found, installing pipx..."
        python3 -m pip install --user pipx
        python3 -m pipx ensurepath
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    # Install TermiVis
    if [ "$FORCE" = true ]; then
        pipx install --force termivls
    else
        pipx install termivls
    fi
    
    log_success "TermiVis installed with pipx"
}

install_with_pip() {
    log_info "Installing TermiVis with pip..."
    
    # Install TermiVis
    if [ "$FORCE" = true ]; then
        pip3 install --user --force-reinstall termivls
    else
        pip3 install --user termivls
    fi
    
    # Add user bin to PATH if needed
    USER_BIN="$HOME/.local/bin"
    if [[ ":$PATH:" != *":$USER_BIN:"* ]]; then
        log_warning "Adding $USER_BIN to PATH"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        export PATH="$USER_BIN:$PATH"
    fi
    
    log_success "TermiVis installed with pip"
}

install_with_docker() {
    log_info "Setting up TermiVis with Docker..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        log_info "Please install Docker from https://www.docker.com/"
        exit 1
    fi
    
    # Create docker-compose configuration
    mkdir -p ~/termivls
    cat > ~/termivls/docker-compose.yml << EOF
version: '3.8'

services:
  termivls:
    image: termivls/termivls:latest
    container_name: termivls-server
    environment:
      - INTERNVL_API_KEY=$API_KEY
    volumes:
      - ~/.config/claude-code:/app/.claude-config:rw
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "termivls", "status"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
EOF
    
    # Start the container
    cd ~/termivls
    docker-compose pull
    docker-compose up -d
    
    # Create a wrapper script
    cat > ~/.local/bin/termivls << 'EOF'
#!/bin/bash
docker-compose -f ~/termivls/docker-compose.yml exec termivls termivls "$@"
EOF
    chmod +x ~/.local/bin/termivls
    
    log_success "TermiVis Docker setup complete"
}

# Main installation
log_info "Starting TermiVis installation..."
log_info "Installation method: $INSTALL_METHOD"

case $INSTALL_METHOD in
    pipx)
        install_with_pipx
        ;;
    pip)
        install_with_pip
        ;;
    docker)
        install_with_docker
        ;;
    *)
        log_error "Unknown installation method: $INSTALL_METHOD"
        exit 1
        ;;
esac

# Verify installation
log_info "Verifying installation..."
if command -v termivls &> /dev/null; then
    TERMIVLS_VERSION=$(termivls version 2>/dev/null | head -n1 || echo "Unknown")
    log_success "TermiVis is available: $TERMIVLS_VERSION"
else
    log_error "TermiVis command not found in PATH"
    log_info "You may need to restart your terminal or run: source ~/.bashrc"
    exit 1
fi

# Setup TermiVis
log_info "Setting up TermiVis with Claude Code..."
if termivls setup --api-key "$API_KEY" $([ "$FORCE" = true ] && echo "--force"); then
    log_success "TermiVis setup complete!"
else
    log_error "Setup failed"
    exit 1
fi

# Final instructions
echo
log_success "🎉 TermiVis installation completed successfully!"
echo
echo "📋 Next steps:"
echo "1. Restart Claude Code to load the new MCP server"
echo "2. Try asking: 'What's in this image?' with any image"
echo "3. Use 'termivls status' to check server health"
echo
echo "💡 Useful commands:"
echo "  termivls status    - Check service status"
echo "  termivls run       - Run server manually"
echo "  termivls --help    - Show all commands"
echo
echo "🆘 Need help? Visit: https://github.com/your-username/termivls"
echo