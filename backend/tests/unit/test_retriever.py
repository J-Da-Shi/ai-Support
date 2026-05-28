from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from core.indexer import build_or_update_index
from core.models import AskMode
from core.retriever import Retriever


class FakeEmbedder:
    model = "m"
    async def embed(self, texts):
        return [_dummy_vec(t) for t in texts]


def _dummy_vec(t: str) -> list[float]:
    rng = np.random.default_rng(abs(hash(t)) % (2**32))
    return rng.normal(size=1536).astype("float32").tolist()


@pytest.mark.asyncio
async def test_retriever_hits_known_query(tmp_path):
    notes = Path(__file__).parent.parent / "fixtures" / "notes_sample"
    embedder = FakeEmbedder()
    bundle = await build_or_update_index(notes, tmp_path / "data", embedder, 500, 50)

    r = Retriever(
        bundle=bundle, embedder=embedder,
        top_k=8, rerank_top_k=3, threshold=0.0,    # 阈值置 0：dummy 向量未必稳定，HIT 由 BM25 占主导
        vector_weight=0.0, bm25_weight=1.0,
    )
    res = await r.search("RR vs RC")
    assert res.mode == AskMode.HIT
    paths = [c.chunk.file_path for c in res.chunks]
    assert any("事务隔离" in p for p in paths)
    assert len(res.chunks) <= 3


@pytest.mark.asyncio
async def test_retriever_fallback_when_below_threshold(tmp_path):
    notes = Path(__file__).parent.parent / "fixtures" / "notes_sample"
    embedder = FakeEmbedder()
    bundle = await build_or_update_index(notes, tmp_path / "data", embedder, 500, 50)

    r = Retriever(
        bundle=bundle, embedder=embedder,
        top_k=8, rerank_top_k=3, threshold=0.99,   # 极高阈值
        vector_weight=0.7, bm25_weight=0.3,
    )
    res = await r.search("Rust 生命周期")
    assert res.mode == AskMode.FALLBACK
    assert len(res.chunks) <= 3                    # 仍返回弱相关片段
    assert res.top1_score < 0.99


@pytest.mark.asyncio
async def test_retriever_empty_when_no_chunks(tmp_path):
    embedder = FakeEmbedder()
    empty_notes = tmp_path / "empty"
    empty_notes.mkdir()
    bundle = await build_or_update_index(empty_notes, tmp_path / "data", embedder, 500, 50)

    r = Retriever(bundle=bundle, embedder=embedder, top_k=8, rerank_top_k=3, threshold=0.5, vector_weight=0.7, bm25_weight=0.3)
    res = await r.search("anything")
    assert res.mode == AskMode.EMPTY
    assert res.chunks == []
