# CliMB Dockerfile
# Build: podman build -t climb .
# Run:   podman run -p 8501:8501 --env-file .env climb

FROM python:3.10-slim AS base

# System dependencies required by scientific Python packages + git for setuptools-scm.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy git metadata for setuptools-scm version detection.
COPY .git .git

# Install Python dependencies first (layer caching).
COPY setup.cfg setup.py pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir .

# Copy the rest of the application.
COPY entry/ entry/
COPY config_examples/ config_examples/

# Default environment variables (overridable at runtime).
ENV LLM_PROVIDER=openai
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "entry/st/app.py"]
