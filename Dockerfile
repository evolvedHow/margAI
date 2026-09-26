# margAI Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY examples/ ./examples/
COPY margAI.toml ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Expose port (must match [gateway] port in margAI.toml)
EXPOSE 8002

# Run margAI
CMD [".venv/bin/python", "-m", "margAI"]