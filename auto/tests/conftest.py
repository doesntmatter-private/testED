import pytest

from autodiag import config
from autodiag.knowledge import KnowledgeStore, ingest_path
from autodiag.obd import SimulatedReader


@pytest.fixture
def store():
    s = KnowledgeStore(":memory:")
    ingest_path(s, config.SEED_KNOWLEDGE_DIR)
    yield s
    s.close()


@pytest.fixture
def misfire_snapshot():
    return SimulatedReader(config.FIXTURES_DIR / "p0301_misfire.json").read()


@pytest.fixture
def lean_snapshot():
    return SimulatedReader(config.FIXTURES_DIR / "p0171_lean.json").read()
