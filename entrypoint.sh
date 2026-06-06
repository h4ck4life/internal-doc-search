#!/bin/sh
# Container entrypoint: ensure the Playwright browser binary is in the
# expected location before starting uvicorn.
#
# Why this exists: docker-compose mounts the host's ./cache directory at
# /app/.cache, which shadows the image-baked ms-playwright/ subdirectory
# with whatever is on the host. On a fresh clone the host's ./cache/ms-
# playwright doesn't exist, so the crawler would fail with
# "Executable doesn't exist at /app/.cache/ms-playwright/...".
#
# The fix: if the volume mount at $PLAYWRIGHT_BROWSERS_PATH is empty but
# the image baked a copy at /app/.cache/ms-playwright.image, copy the
# image copy into the volume location. Idempotent: a non-empty volume
# means we already have a working browser, so we skip the copy.

set -e

BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-/app/.cache/ms-playwright}"
IMAGE_BROWSERS="/app/.cache/ms-playwright.image"

# Move the image-baked copy aside on first start so the volume mount
# doesn't shadow it. We rename rather than copy because the browser is
# ~110 MB and we want to keep the image-side copy untouched.
if [ -d "$IMAGE_BROWSERS" ] && [ ! -d "$BROWSERS_PATH" ]; then
    echo "[entrypoint] Seeding Playwright browser volume from image copy..."
    mkdir -p "$(dirname "$BROWSERS_PATH")"
    cp -a "$IMAGE_BROWSERS" "$BROWSERS_PATH"
    echo "[entrypoint] Browser ready at $BROWSERS_PATH"
fi

exec "$@"
