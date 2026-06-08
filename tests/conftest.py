"""Shared test fixtures."""

import os
import tempfile
import sys
import types
import pytest

from fastapi.testclient import TestClient


def pytest_addoption(parser):
    parser.addoption(
        "--run-qdrant-tests",
        action="store_true",
        default=False,
        help="run tests that require a live Qdrant instance",
    )


def _install_slowapi_stub() -> None:
    """Provide a tiny slowapi stand-in for unit tests.

    The app imports slowapi at module import time. Unit tests don't need the
    real rate limiter behavior, so this keeps API tests independent from that
    optional external package in the active venv.
    """
    if "slowapi" in sys.modules:
        return

    slowapi = types.ModuleType("slowapi")
    util = types.ModuleType("slowapi.util")
    errors = types.ModuleType("slowapi.errors")

    class Limiter:
        def __init__(self, key_func):
            self.key_func = key_func

        def limit(self, _rule):
            def decorator(func):
                return func

            return decorator

    class RateLimitExceeded(Exception):
        pass

    def get_remote_address(request):
        return getattr(getattr(request, "client", None), "host", "testclient")

    async def _rate_limit_exceeded_handler(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)

    slowapi.Limiter = Limiter
    slowapi._rate_limit_exceeded_handler = _rate_limit_exceeded_handler
    util.get_remote_address = get_remote_address
    errors.RateLimitExceeded = RateLimitExceeded

    sys.modules["slowapi"] = slowapi
    sys.modules["slowapi.util"] = util
    sys.modules["slowapi.errors"] = errors


_install_slowapi_stub()


@pytest.fixture
def temp_db():
    """Use a temporary database for store tests."""
    import store

    old_path = store.DB_PATH
    with tempfile.TemporaryDirectory() as tmpdir:
        store.DB_PATH = os.path.join(tmpdir, "test_config.db")
        store.init_db()
        yield store
        store.DB_PATH = old_path


@pytest.fixture
def client():
    """Create a FastAPI TestClient (models not loaded for unit tests)."""
    from api import app
    with TestClient(app) as c:
        yield c
