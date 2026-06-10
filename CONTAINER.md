# Container Guide

## Published Image

The image is published to GitHub Container Registry (GHCR):

```
ghcr.io/h4ck4life/internal-doc-search
```

**Tags**: `v1.0.0`, `latest`

## Quick Start (Pre-built Image)

```bash
# Pull and run — no build needed
docker compose -f docker-compose.published.yml up -d
```

Or with the standard compose file (replace `build: .` with `image: ghcr.io/h4ck4life/internal-doc-search:latest`).

## What's Baked In

| Component | Baked at Build | Runtime Download |
|-----------|---------------|------------------|
| Playwright Chromium | ✅ (~110 MB) | — |
| HuggingFace Models | ✅ (~420 MB) | — |
| Python deps | ✅ | — |

First startup is instant — no downloads needed.

## Publishing a New Version

The image is multi-arch (`linux/amd64` + `linux/arm64`). The `linux/arm64`
manifest is built locally and pushed manually; the `linux/amd64` manifest
is built and merged via GitHub Actions (`.github/workflows/build-amd64.yml`)
because cross-building amd64 via QEMU on Apple Silicon stalls under
emulation.

### One-time setup

```bash
# Refresh auth with packages scope (one-time)
gh auth refresh -h github.com -s write:packages
```

### 1. Build & push arm64 locally

```bash
# Build the multi-stage image on the host (native arm64 on Apple Silicon).
docker build \
  -t ghcr.io/h4ck4life/internal-doc-search:arm64-tmp \
  -t ghcr.io/h4ck4life/internal-doc-search:latest \
  .

# Login and push the arm64 image and the rolling :latest tag.
gh auth token | docker login ghcr.io -u h4ck4life --password-stdin
docker push ghcr.io/h4ck4life/internal-doc-search:arm64-tmp
docker push ghcr.io/h4ck4life/internal-doc-search:latest
```

### 2. Trigger the amd64 build & multi-arch merge

The workflow runs on a free GitHub-hosted `ubuntu-latest` runner, so the
amd64 build is native (no QEMU) and reliable. It reads the arm64
manifest digest from `:latest`, builds amd64, and pushes a multi-arch
index that updates `:latest` (and `:vX.Y.Z` if triggered by a tag).

```bash
# Manual run (updates :latest only)
gh workflow run build-amd64.yml

# Or: cut a release tag, push it, the workflow runs automatically and
# pushes both :latest and :vX.Y.Z
git tag v1.0.1
git push origin v1.0.1
```

### 3. (Optional) Drop the temporary arm64 tag

```bash
gh api -X DELETE /user/packages/container/internal-doc-search/versions/$( \
  gh api /user/packages/container/internal-doc-search/versions \
    --jq '.[] | select(.metadata.container.tags[]? == "arm64-tmp") | .id' \
)
```

### Verify

```bash
# Multi-arch manifest should list both arm64 and amd64.
docker buildx imagetools inspect ghcr.io/h4ck4life/internal-doc-search:latest

# Anonymous pull should succeed (image is public).
docker pull ghcr.io/h4ck4life/internal-doc-search:latest
```

GitHub UI: https://github.com/h4ck4life/internal-doc-search/pkgs/container/internal-doc-search
```

## Image Size

| Layer | Size |
|-------|------|
| Base (Python 3.11-slim) | ~150 MB |
| PyTorch CPU | ~180 MB |
| Python deps | ~200 MB |
| Playwright Chromium | ~110 MB |
| HuggingFace Models | ~420 MB |
| Application code | <1 MB |
| **Total** | **~1.1 GB** |

## Docker Compose for Users

Create `docker-compose.published.yml`:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.15.2
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  api:
    image: ghcr.io/h4ck4life/internal-doc-search:latest
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      # Host home directory is exposed read-only for folder watching.
      # Add watchers using the container path, for example /watched/home/Documents.
      # Set FOLDER_WATCH_HOST_ROOT before docker compose up to expose another root.
      - ${FOLDER_WATCH_HOST_ROOT:-~}:/watched/home:ro
      # Playwright Chromium and HuggingFace models are baked into the
      # image — no host bind mount for /app/.cache is needed.
    environment:
      - QDRANT_URL=http://qdrant:6333
      - HF_HUB_OFFLINE=1
      - TRANSFORMERS_OFFLINE=1
      - CRAWL_WAIT_UNTIL=networkidle
      - CRAWL_DELAY_BEFORE_HTML=2.0
      - CRAWL_PAGE_TIMEOUT_MS=60000
      - CRAWL_WORD_COUNT_THRESHOLD=1
      - CRAWL_USE_CONTENT_FILTER=true
      - CRAWL_PRUNE_THRESHOLD=0.35
      - CRAWL_SCAN_FULL_PAGE=true
      - CRAWL_SCROLL_DELAY=0.2
      - CRAWL_MAX_SCROLL_STEPS=15
      - CRAWL_PROCESS_IFRAMES=true
      - CRAWL_FLATTEN_SHADOW_DOM=true
      - CRAWL_REMOVE_OVERLAYS=true
      - CRAWL_REMOVE_CONSENT_POPUPS=true
      - CRAWL_SIMULATE_USER=false
      - CRAWL_MAGIC=false
      - CRAWL_OVERRIDE_NAVIGATOR=false
      - CRAWL_DEEP_MAX_PAGES=500
      - CRAWL_USER_AGENT_MODE=random
      - FOLDER_WATCH_ALLOWED_ROOTS=/watched/home
      - FOLDER_WATCH_SCAN_INTERVAL=5
      - FOLDER_WATCH_DEBOUNCE_SECONDS=2
      - FOLDER_WATCH_MAX_FILES_PER_SCAN=10
      - FILE_QDRANT_BATCH_SIZE=64
      - QDRANT_TIMEOUT_SECONDS=120
    shm_size: '2gb'

volumes:
  qdrant_data:
```

## Optional GPU API Container with Podman

The default compose stack runs the CPU API image. To expose an NVIDIA GPU to the API container with Podman, start the GPU API service explicitly:

```bash
podman compose up -d qdrant api-gpu
```

If the normal API is already running, stop it first because both services bind `localhost:8000`:

```bash
podman compose stop api
```

The GPU service uses `devices: nvidia.com/gpu=all`; the in-app GPU inference checkbox remains off by default until CUDA is visible inside the running container and the user enables it.
