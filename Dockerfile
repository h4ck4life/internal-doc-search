FROM python:3.11-slim

WORKDIR /app

# Install system deps for Crawl4AI (Playwright)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install CPU-only PyTorch FIRST ────────────────────────────────
# sentence-transformers depends on torch; pre-installing the CPU wheel
# (~180 MB) prevents pip from downloading the CUDA wheel (~426 MB).
# Use BuildKit cache mount so pip packages survive rebuilds.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --cache-dir /root/.cache/pip \
    torch --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining Python deps ─────────────────────────────────
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --cache-dir /root/.cache/pip -r requirements.txt

# ── Install Playwright Chromium (cached between builds) ───────────
RUN --mount=type=cache,target=/root/.cache/ms-playwright \
    python -m playwright install --with-deps chromium

COPY api.py ingest.py store.py chunker.py mcp_server.py ./
COPY static/ static/
COPY urls.txt ./

RUN mkdir -p /app/data /app/.cache

# Persist HF model downloads across rebuilds via volume mount
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
