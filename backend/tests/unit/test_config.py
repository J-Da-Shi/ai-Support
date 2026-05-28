import os
from pathlib import Path
import pytest
from core.config import Settings, load_settings


def test_load_settings_from_env(tmp_path, monkeypatch):
    notes = tmp_path / "notes"
    notes.mkdir()
    resume = tmp_path / "resume.md"
    resume.write_text("hi", encoding="utf-8")
    monkeypatch.setenv("NOTES_DIR", str(notes))
    monkeypatch.setenv("RESUME_PATH", str(resume))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dk-test")
    monkeypatch.setenv("RETRIEVAL_THRESHOLD", "0.42")

    s = load_settings()
    assert s.notes_dir == notes
    assert s.resume_path == resume
    assert s.retrieval_threshold == 0.42
    assert s.llm_provider == "deepseek"
    assert s.embedding_model == "text-embedding-3-small"


def test_missing_api_key_for_provider_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_DIR", str(tmp_path))
    monkeypatch.setenv("RESUME_PATH", str(tmp_path / "resume.md"))
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        load_settings().validate_for_runtime()
