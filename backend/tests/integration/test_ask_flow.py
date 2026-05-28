import asyncio
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.indexer import build_or_update_index
from core.retriever import Retriever
from routes.ask import router as ask_router
from routes.asr import router as asr_router


class FakeEmbedder:
    model = "m"
    async def embed(self, texts):
        return [_dv(t) for t in texts]


def _dv(t: str) -> list[float]:
    rng = np.random.default_rng(abs(hash(t)) % (2**32))
    return rng.normal(size=1536).astype("float32").tolist()


class FakeLLM:
    async def stream(self, prompt: str) -> AsyncIterator[str]:
        for ch in "假装的回答":
            yield ch


def _build_app(notes_dir: Path, data_dir: Path, threshold: float, resume_text: str):
    # 不使用 main.app（其 lifespan 会试图加载真实 settings/index）；
    # 在 Python 3.13 下用 asyncio.run 跑一次 setup，并构造一个干净的 FastAPI 实例
    embedder = FakeEmbedder()

    async def _setup():
        bundle = await build_or_update_index(notes_dir, data_dir, embedder, 500, 50)
        return bundle

    bundle = asyncio.run(_setup())
    retriever = Retriever(bundle, embedder, 8, 3, threshold, 0.0, 1.0)

    app = FastAPI()
    app.include_router(ask_router)
    app.include_router(asr_router)
    app.state.settings = type("S", (), {
        "asr_timeout_s": 5,
        "llm_first_token_timeout_s": 5,
        "llm_total_timeout_s": 30,
        "retrieval_threshold": threshold,
    })()
    app.state.retriever = retriever
    app.state.llm = FakeLLM()
    app.state.resume_text = resume_text
    app.state.asr = None
    return app


def _parse_sse(raw: str) -> list[tuple[str, str]]:
    events = []
    for block in raw.strip().split("\n\n"):
        ev = data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        events.append((ev, data))
    return events


def test_ask_hit_flow(tmp_path):
    notes = Path(__file__).parent.parent / "fixtures" / "notes_sample"
    app = _build_app(notes, tmp_path / "d", threshold=0.0, resume_text="我是测试简历")
    with TestClient(app) as c:
        with c.stream("GET", "/api/ask", params={"query": "RR vs RC"}) as r:
            text = "".join(r.iter_text())
    events = _parse_sse(text)
    types = [e for e, _ in events]
    assert "mode" in types
    assert "chunks" in types
    assert types.count("token") >= 1
    assert types[-1] == "done"

    import json
    mode_ev = next(d for e, d in events if e == "mode")
    assert json.loads(mode_ev)["mode"] == "hit"


def test_ask_fallback_flow(tmp_path):
    notes = Path(__file__).parent.parent / "fixtures" / "notes_sample"
    app = _build_app(notes, tmp_path / "d", threshold=99.0, resume_text="我做过订单系统")
    with TestClient(app) as c:
        with c.stream("GET", "/api/ask", params={"query": "Rust 生命周期"}) as r:
            text = "".join(r.iter_text())
    events = _parse_sse(text)
    import json
    mode_ev = next(d for e, d in events if e == "mode")
    assert json.loads(mode_ev)["mode"] == "fallback"
    assert any(e == "token" for e, _ in events)


def test_health_reports_config_error_when_keys_missing(monkeypatch):
    # Build a fresh app, run real lifespan with empty keys; expect 200 + config_error
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # need a fresh import of main so lifespan picks up env
    import sys
    if "main" in sys.modules:
        del sys.modules["main"]
    import main  # noqa: E402
    with TestClient(main.app) as c:
        r = c.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert "API_KEY" in (body.get("config_error") or "")


def test_ask_returns_config_error_when_degraded(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    import sys
    if "main" in sys.modules:
        del sys.modules["main"]
    import main  # noqa: E402
    with TestClient(main.app) as c:
        with c.stream("GET", "/api/ask", params={"query": "x"}) as r:
            text = "".join(r.iter_text())
    events = _parse_sse(text)
    types = [e for e, _ in events]
    assert "error" in types
    err_data = next(d for e, d in events if e == "error")
    import json
    assert json.loads(err_data)["stage"] == "config"
