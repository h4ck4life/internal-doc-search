"""Shared test fixtures."""

import os
import tempfile
import pytest

from fastapi.testclient import TestClient


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
