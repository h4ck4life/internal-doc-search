# Container Guide

## Published Image

The image is published to GitHub Container Registry (GHCR):

```
ghcr.io/h4ck4life/docsearch-api
```

**Tags**: `v1.0.0`, `latest`

## Quick Start (Pre-built Image)

```bash
# Pull and run — no build needed
docker compose -f docker-compose.published.yml up -d
```

Or with the standard compose file (replace `build: .` with `image: ghcr.io/h4ck4life/docsearch-api:latest`).

## What's Baked In

| Component | Baked at Build | Runtime Download |
|-----------|---------------|------------------|
| Playwright Chromium | ✅ (~110 MB) | — |
| HuggingFace Models | ✅ (~420 MB) | — |
| Python deps | ✅ | — |

First startup is instant — no downloads needed.

## Publishing a New Version

### Prerequisites
- GitHub CLI (`gh`) installed and authenticated
- `write:packages` scope on your token

```bash
# Refresh auth with packages scope (one-time)
gh auth refresh -h github.com -s write:packages
```

### Build, Tag & Push

```bash
# 1. Tag the release
git tag v1.0.1
git push origin v1.0.1

# 2. Build with GHCR tags
docker build \
  -t ghcr.io/h4ck4life/docsearch-api:v1.0.1 \
  -t ghcr.io/h4ck4life/docsearch-api:latest \
  .

# 3. Login to GHCR
gh auth token | docker login ghcr.io -u h4ck4life --password-stdin

# 4. Push both tags
docker push ghcr.io/h4ck4life/docsearch-api:v1.0.1
docker push ghcr.io/h4ck4life/docsearch-api:latest
```

### Verify

```
https://github.com/h4ck4life/internal-doc-search/pkgs/container/docsearch-api
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
    image: ghcr.io/h4ck4life/docsearch-api:latest
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./.cache:/app/.cache
    environment:
      - QDRANT_URL=http://qdrant:6333
    shm_size: '2gb'

volumes:
  qdrant_data:
```
