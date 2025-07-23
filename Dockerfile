# Multi-stage build for optimal image size
FROM python:3.11-slim as builder

# Install uv for fast dependency resolution
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies in virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache-dir -r <(uv export --format requirements-txt)

# Production stage
FROM python:3.11-slim

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y \
    libpng-dev \
    libjpeg-dev \
    libwebp-dev \
    libgif-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ src/
COPY README.md LICENSE* ./

# Install the application
RUN pip install -e .

# Create non-root user for security
RUN groupadd -r termivls && useradd -r -g termivls termivls
RUN chown -R termivls:termivls /app
USER termivls

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD termivls status || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port (if needed for HTTP mode)
EXPOSE 8080

# Default command
CMD ["termivls", "run"]