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

# Keep Python/pip from writing avoidable files into image layers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Install the OS libraries Chromium needs once in the shared base. Both
# the browser-install stage and the runtime stage inherit these files.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2 libatspi2.0-0 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# ── Install PyTorch FIRST ─────────────────────────────────────────
# sentence-transformers depends on torch; pre-install it so pip does not
# choose a larger/default wheel later. Override TORCH_INDEX_URL with a CUDA
# wheel index (for example cu124) when building the optional GPU image.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# Use BuildKit cache mount so pip packages survive rebuilds with buildx.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-compile --cache-dir /root/.cache/pip \
    torch --index-url ${TORCH_INDEX_URL}

# ── Install remaining Python deps ─────────────────────────────────
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-compile --cache-dir /root/.cache/pip -r requirements.txt

# ── Stage: install Playwright Chromium ONCE ──────────────────────
# Install to /app/.cache/ms-playwright (the real stage path) so the
# runtime stage can COPY it via `--from=browsers`. We don't use a
# BuildKit cache mount here because cache mounts overlay a tempfs during
# the RUN — the files never become part of the stage's committed layer,
# so `COPY --from` can't see them. Instead, the cost of redownloading is
# amortized by the layer cache: as long as requirements.txt and the
# base image haven't changed, the install RUN is a no-op CACHED step.
FROM base AS browsers
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
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
