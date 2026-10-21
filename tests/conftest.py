"""Shared pytest fixtures for the PrimoGreedy API tests.

The tests must never touch the production ``data.duckdb``.  We point
``DUCKDB_PATH`` at a throwaway file *before* importing ``api`` so the module
binds to the temp database, then initialise the schema on it.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# --- Environment must be set before `import api` ---------------------------
_TMP_DIR = Path(tempfile.mkdtemp(prefix="primogreedy-tests-"))
_TEST_DB = _TMP_DIR / "test.duckdb"
os.environ["DUCKDB_PATH"] = str(_TEST_DB)
os.environ["VPS_API_KEY"] = "test-api-key"
os.environ["PRICE_TTL"] = "60"

# Make the project root importable regardless of where pytest is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api  # noqa: E402  (import after env is configured)

TEST_API_KEY = os.environ["VPS_API_KEY"]


@pytest.fixture(scope="session", autouse=True)
def _init_database():
    """Create the schema once for the whole session."""
    api.init_db()
    yield


@pytest.fixture()
def clean_tables():
    """Truncate the mutable tables between tests that write."""
    with api.get_db_ctx() as con:
        for t in ("seen_tickers", "paper_portfolio", "analyses", "watchlist"):
            con.execute(f"DELETE FROM {t}")
    yield


@pytest.fixture()
def client():
    """FastAPI TestClient with the app's lifespan (table init) run.

    Rate limiting is disabled by default so tests are independent; the
    dedicated rate-limit test re-enables it explicitly.
    """
    from fastapi.testclient import TestClient

    api.limiter.enabled = False
    with TestClient(api.app) as c:
        yield c
    api.limiter.enabled = True


@pytest.fixture()
def auth_headers():
    return {"x-api-key": TEST_API_KEY}
