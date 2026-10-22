import pytest

from app.core.llm import reset_llm_client


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    """Tests never hit a real provider; the rule-based path is the default."""
    monkeypatch.setenv("LLM_ENABLED", "0")
    reset_llm_client()
    yield
    reset_llm_client()


@pytest.fixture(autouse=True)
def _isolated_data_dir(monkeypatch, tmp_path):
    """Persisted memory / external datasets go to a per-test temp dir, never the real data/."""
    monkeypatch.setenv("CREDIT_DATA_DIR", str(tmp_path))
    yield
