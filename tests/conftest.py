import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def pytest_collection_modifyitems(config, items):
    needed = [PROJECT_DIR / "data/processed/forecasts_future.csv", PROJECT_DIR / "models/model_card.json"]
    if not all(p.exists() for p in needed):
        skip = pytest.mark.skip(reason="Run notebooks 01-03 first (forecast outputs / models missing).")
        for item in items:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c
