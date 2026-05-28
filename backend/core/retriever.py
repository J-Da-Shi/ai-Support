import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from core.models import AskMode, ScoredChunk

log = logging.getLogger(__name__)


class _EmbProto(Protocol):
    async def embed(self, texts): ...


@dataclass
class RetrievalResult:
    mode: AskMode
    chunks: list[ScoredChunk]
    top1_score: float


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


class Retriever:
    def __init__(
        self,
        bundle,                # IndexBundle
        embedder: _EmbProto,
        top_k: int,
        rerank_top_k: int,
        threshold: float,
        vector_weight: float,
        bm25_weight: float,
        embed_timeout: float | None = None,
    ):
        self.bundle = bundle
        self.embedder = embedder
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.threshold = threshold
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.embed_timeout = embed_timeout

        if bundle.chunks:
            corpus = [_tokenize(c.text) for c in bundle.chunks]
            self.bm25 = BM25Okapi(corpus)
        else:
            self.bm25 = None

    async def _embed_query_with_retry(self, query: str) -> list[float] | None:
        """Spec §7: 1 retry with 200ms backoff. Returns None on final failure -> BM25 fallback."""
        attempt_timeout = self.embed_timeout
        for attempt in (1, 2):
            try:
                coro = self.embedder.embed([query])
                if attempt_timeout:
                    result = await asyncio.wait_for(coro, timeout=attempt_timeout)
                else:
                    result = await coro
                return result[0]
            except asyncio.TimeoutError:
                if attempt == 2:
                    log.warning("embedder timeout after %d attempts; falling back to BM25", attempt)
                    return None
                log.info("embedder timeout, retrying after 200ms")
                await asyncio.sleep(0.2)
            except Exception as e:
                if attempt == 2:
                    log.warning("embedder failed after %d attempts; falling back to BM25: %s", attempt, e)
                    return None
                log.info("embedder error, retrying after 200ms: %s", e)
                await asyncio.sleep(0.2)
        return None

    async def search(self, query: str) -> RetrievalResult:
        if not self.bundle.chunks or self.bm25 is None:
            return RetrievalResult(AskMode.EMPTY, [], 0.0)

        # 向量召回（带超时与一次重试；失败/超时降级 BM25-only）
        qvec = await self._embed_query_with_retry(query)

        n = len(self.bundle.chunks)
        if qvec is not None:
            v = np.array([qvec], dtype="float32")
            # 归一化与索引内一致
            faiss_normalize(v)
            k = min(self.top_k, n)
            sims, ids = self.bundle.faiss_index.search(v, k)
            vec_pairs = list(zip(ids[0].tolist(), sims[0].tolist()))
        else:
            vec_pairs = [(i, 0.0) for i in range(min(self.top_k, n))]

        # BM25 全量打分
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        # 语料级 BM25 最大值，用于稳定归一化（不依赖候选集）
        bm25_max = float(max(bm25_scores)) if len(bm25_scores) else 0.0

        # 候选集 = 向量 top_k 与 bm25 top_k 的并集
        bm25_top_idx = np.argsort(bm25_scores)[::-1][: self.top_k].tolist()
        candidate_ids = list({i for i, _ in vec_pairs} | set(bm25_top_idx))

        vec_lookup = {i: s for i, s in vec_pairs}
        cand_vec = [vec_lookup.get(i, 0.0) for i in candidate_ids]
        cand_bm = [float(bm25_scores[i]) for i in candidate_ids]
        # 稳定标度：余弦相似度 [-1,1] -> [0,1]；BM25 用语料级最大值归一化
        nv = [(s + 1.0) / 2.0 for s in cand_vec]
        nb = [b / bm25_max if bm25_max > 0 else 0.0 for b in cand_bm]
        mixed = [
            self.vector_weight * v + self.bm25_weight * b
            for v, b in zip(nv, nb)
        ]

        order = sorted(range(len(candidate_ids)), key=lambda i: mixed[i], reverse=True)
        top = order[: self.rerank_top_k]
        scored = [
            ScoredChunk(
                chunk=self.bundle.chunks[candidate_ids[k]],
                score=mixed[k],
                vector_score=cand_vec[k],
                bm25_score=cand_bm[k],
            )
            for k in top
        ]
        top1 = scored[0].score if scored else 0.0
        mode = AskMode.HIT if top1 >= self.threshold else AskMode.FALLBACK
        return RetrievalResult(mode=mode, chunks=scored, top1_score=top1)


def faiss_normalize(v: np.ndarray) -> None:
    import faiss
    faiss.normalize_L2(v)
