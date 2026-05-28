from pathlib import Path

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
    # 查询词与语料词表无交集 → BM25 全 0 → top1 = vector_weight * vec_clamped
    res = await r.search("Rust 生命周期")
    assert res.mode == AskMode.FALLBACK
    assert len(res.chunks) <= 3                    # 仍返回弱相关片段
    # BM25 全 0 时，top1 上界为 vector_weight；vec_clamped ∈ [0,1]
    assert 0.0 <= res.top1_score <= 0.7 + 1e-6
    assert res.top1_score < 0.99


@pytest.mark.asyncio
async def test_top1_score_correlates_with_match_quality(tmp_path):
    """证明 top1_score 不再被 min-max 钉死成阶梯函数：
    控制余弦相似度，验证 top1_score 随匹配质量线性变化。"""
    notes = Path(__file__).parent.parent / "fixtures" / "notes_sample"
    embedder = FakeEmbedder()
    bundle = await build_or_update_index(notes, tmp_path / "data", embedder, 500, 50)

    # 从 FAISS index 中重建第一个 chunk 的归一化向量作为「目标」
    target_unit = bundle.faiss_index.reconstruct(0).astype("float32")

    class CtrlEmbedder:
        model = "m"
        def __init__(self, vec):
            self._vec = vec
        async def embed(self, texts):
            return [self._vec.tolist() for _ in texts]

    # 高相似度：查询向量 = 目标向量本身（cos≈1 → clamped≈1.0）
    r_hi = Retriever(
        bundle=bundle, embedder=CtrlEmbedder(target_unit),
        top_k=8, rerank_top_k=3, threshold=0.0,
        vector_weight=1.0, bm25_weight=0.0,
    )
    # 低相似度：构造与 target 正交的向量（cos≈0 → clamped≈0.5）
    rng = np.random.default_rng(7)
    rand = rng.normal(size=target_unit.shape).astype("float32")
    proj = float(np.dot(rand, target_unit)) * target_unit
    ortho = rand - proj
    ortho = ortho / (np.linalg.norm(ortho) + 1e-12)

    r_lo = Retriever(
        bundle=bundle, embedder=CtrlEmbedder(ortho),
        top_k=8, rerank_top_k=3, threshold=0.0,
        vector_weight=1.0, bm25_weight=0.0,
    )

    res_hi = await r_hi.search("q")
    res_lo = await r_lo.search("q")
    # 完全匹配 ≈ 1.0；正交 ≈ 0.5（因为 (0+1)/2 = 0.5）
    assert res_hi.top1_score > 0.95
    assert res_lo.top1_score < 0.7
    # 高质量与低质量之间应有显著差距，证明分数不是阶梯函数
    assert res_hi.top1_score - res_lo.top1_score > 0.2


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
