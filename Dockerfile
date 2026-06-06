FROM python:3.11-slim

WORKDIR /app

# Install system deps for Crawl4AI (Playwright)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers for Crawl4AI
RUN python -m playwright install --with-deps chromium

COPY api.py ingest.py store.py chunker.py mcp_server.py ./
COPY static/ static/
COPY urls.txt ./

RUN mkdir -p /app/data /app/.cache

# Persist HF model downloads across rebuilds via volume mount
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
