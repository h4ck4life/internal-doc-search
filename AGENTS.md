# Repository Guidelines

## Project Structure & Module Organization

This is a Python FastAPI service for crawling, indexing, and searching internal documentation. Core modules live at the repository root: `api.py` exposes HTTP endpoints and the web UI, `mcp_server.py` exposes MCP tools, `ingest.py` handles crawling, `chunker.py` splits documents, `search_utils.py` ranks results, and `store.py` manages SQLite state. Runtime configuration and model loading are in `shared.py`.

Tests are in `tests/` and usually mirror the module under test, for example `tests/test_store.py` and `tests/test_api.py`. Static browser assets live in `static/`. OpenSpec notes and specs are under `openspec/`.

## Build, Test, and Development Commands

Use the repository virtual environment when available:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -v
```

Common commands:

```powershell
docker compose up -d qdrant
uvicorn api:app --reload
docker compose up -d
python -m pytest tests/ -v
python -m pytest tests/ -v -m "not qdrant"
python -m pytest tests/test_store.py::test_add_url -v
```

`docker compose up -d qdrant` starts the vector database. `uvicorn api:app --reload` runs the API and dashboard at `http://localhost:8000`. `docker compose up -d` starts the full stack.

## Coding Style & Naming Conventions

Write Python with 4-space indentation, clear function names, and explicit error handling. Prefer helpers from `store.py`, `shared.py`, and `search_utils.py` over duplicating logic. Use `snake_case` for functions, variables, and test names; use `PascalCase` for Pydantic models such as `URLInput`.

Keep changes scoped. Avoid unrelated UI, config, or formatting churn in the same commit.

## Testing Guidelines

Tests use `pytest`, `pytest-asyncio`, and FastAPI `TestClient`. Name tests as `test_<behavior>` and place them in the closest matching test file. Use `tests/conftest.py` fixtures for temporary SQLite state and API clients.

Mark tests that require live Qdrant with `@pytest.mark.qdrant`, and run routine local checks with:

```powershell
python -m pytest tests/ -v -m "not qdrant"
```

## Commit & Pull Request Guidelines

Recent commits use short, imperative messages, sometimes with a conventional prefix, for example `feat: add file upload with API, MCP, and UI support`, `Update UI cleaner`, and `Fix max depth default to 1`.

For pull requests, include a summary, changed endpoints or UI areas, test commands run, and any configuration impact. Link related issues or OpenSpec changes when applicable. Include screenshots only for visible UI changes.

## Security & Configuration Tips

Do not commit local data, model caches, secrets, or generated database files. Keep environment-specific settings in environment variables or compose overrides. Be careful with delete endpoints: Qdrant cleanup should succeed before removing SQLite records.
