# syntax=docker/dockerfile:1.6
# Multi-stage build: install Playwright Chromium ONCE in a "browsers" stage
# and COPY it into the runtime image. The browser install is the most
# expensive step in the build (~100 s for a 109 MB download), so we keep
# it in its own stage: as long as the base image and requirements.txt
# don't change, the `browsers` stage's layers are CACHED and the
# `COPY --from=browsers` step is a fast copy. Application-code edits
# skip the download entirely.

FROM python:3.11-slim AS base

WORKDIR /app

# Install system deps for Crawl4AI (Playwright) and the build tools we'll
# need in the browsers stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install CPU-only PyTorch FIRST ────────────────────────────────
# sentence-transformers depends on torch; pre-installing the CPU wheel
# (~180 MB) prevents pip from downloading the CUDA wheel (~426 MB).
# Use BuildKit cache mount so pip packages survive rebuilds with buildx.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --cache-dir /root/.cache/pip \
    torch --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining Python deps ─────────────────────────────────
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --cache-dir /root/.cache/pip -r requirements.txt

# ── Stage: install Playwright Chromium ONCE ──────────────────────
# Install to /app/.cache/ms-playwright (the real stage path) so the
# runtime stage can COPY it via `--from=browsers`. We don't use a
# BuildKit cache mount here because cache mounts overlay a tempfs during
# the RUN — the files never become part of the stage's committed layer,
# so `COPY --from` can't see them. Instead, the cost of redownloading is
# amortized by the layer cache: as long as requirements.txt and the
# base image haven't changed, the install RUN is a no-op CACHED step.
FROM base AS browsers
# Install the OS libraries Chromium needs (libnss3, libatk1.0, libgbm,
# etc.) so `playwright install` doesn't need `--with-deps` (which would
# call apt-get itself, complicating the layer graph).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2 libatspi2.0-0 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.cache/ms-playwright
RUN python -m playwright install chromium

# ── Stage: download HuggingFace models ONCE ─────────────────────
FROM base AS models
ENV HF_HOME=/app/.cache/huggingface
# Download bi-encoder + cross-encoder (~420 MB) so users don't wait on first start.
# Same pattern as browsers stage — cached layer survives code-only rebuilds.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('multi-qa-mpnet-base-cos-v1'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# ── Runtime stage ────────────────────────────────────────────────
FROM base

# Install Chromium's runtime libraries (libnss3, libgbm, libatk, …) so
# the Playwright browser can launch at runtime. We install these in the
# final image rather than copying from the browsers stage because they
# live in /usr/lib, which the base image already provides — only the
# browser binary at /app/.cache/ms-playwright is build-time-only state.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2 libatspi2.0-0 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Copy the Playwright browsers from the previous stage directly into the
# runtime image at the path PLAYWRIGHT_BROWSERS_PATH points to. The image
# is the source of truth — no host bind mount shadows this.
COPY --from=browsers /app/.cache/ms-playwright /app/.cache/ms-playwright

# Same pattern for HuggingFace models — baked at build time so users
# don't wait 2-3 minutes for 400MB+ downloads on first start.
COPY --from=models /app/.cache/huggingface /app/.cache/huggingface

# Copy all top-level Python files and the URLs list. New modules are
# picked up automatically — no need to edit this list per file.
COPY *.py ./
COPY static/ static/
COPY urls.txt ./

RUN mkdir -p /app/data

# Playwright Chromium and HuggingFace models are baked directly into this
# image at the paths below — no host bind mount is needed for /app/.cache.
# Only the SQLite database is persisted (see docker-compose.yml).
ENV HF_HOME=/app/.cache/huggingface
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.cache/ms-playwright

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
