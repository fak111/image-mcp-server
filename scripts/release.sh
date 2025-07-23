#!/bin/bash

# TermiVis Release Script
# Usage: ./scripts/release.sh [version]
# Example: ./scripts/release.sh 1.0.1

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 TermiVis Release Script${NC}"

# Check if version is provided
if [ -z "$1" ]; then
    echo -e "${RED}❌ Error: Version required${NC}"
    echo "Usage: ./scripts/release.sh [version]"
    echo "Example: ./scripts/release.sh 1.0.1"
    exit 1
fi

VERSION=$1
TAG="v$VERSION"

echo -e "${YELLOW}📋 Release Checklist:${NC}"
echo "  Version: $VERSION"
echo "  Tag: $TAG"
echo ""

# Validate version format
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo -e "${RED}❌ Error: Invalid version format. Use semantic versioning (e.g., 1.0.1)${NC}"
    exit 1
fi

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${YELLOW}⚠️  Warning: You're not on main branch (current: $CURRENT_BRANCH)${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}⚠️  Warning: You have uncommitted changes${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update version in pyproject.toml
echo -e "${YELLOW}📝 Updating version in pyproject.toml...${NC}"
sed -i.bak "s/version = \".*\"/version = \"$VERSION\"/" pyproject.toml

# Show the change
echo "Version updated:"
grep "version = " pyproject.toml

# Commit version change
echo -e "${YELLOW}💾 Committing version change...${NC}"
git add pyproject.toml
git commit -m "Bump version to $VERSION"

# Create and push tag
echo -e "${YELLOW}🏷️  Creating tag $TAG...${NC}"
git tag -a "$TAG" -m "Release $VERSION"

echo -e "${YELLOW}📤 Pushing changes and tag...${NC}"
git push origin main
git push origin "$TAG"

echo -e "${GREEN}✅ Success! Release process initiated.${NC}"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. 🔍 Monitor GitHub Actions: https://github.com/zhangbeibei/image-mcp-server/actions"
echo "2. 📦 Check PyPI: https://pypi.org/project/termivls/"
echo "3. 🧪 Test installation: pipx install termivls"
echo ""
echo -e "${YELLOW}💡 Tip: You can also create a GitHub Release manually at:${NC}"
echo "   https://github.com/zhangbeibei/image-mcp-server/releases/new"
