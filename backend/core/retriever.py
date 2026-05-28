from dataclasses import dataclass
from typing import Protocol

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from core.models import AskMode, Chunk, ScoredChunk


class _EmbProto(Protocol):
    async def embed(self, texts): ...


@dataclass
class RetrievalResult:
    mode: AskMode
    chunks: list[ScoredChunk]
    top1_score: float


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


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
    ):
        self.bundle = bundle
        self.embedder = embedder
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.threshold = threshold
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        if bundle.chunks:
            corpus = [_tokenize(c.text) for c in bundle.chunks]
            self.bm25 = BM25Okapi(corpus)
        else:
            self.bm25 = None

    async def search(self, query: str) -> RetrievalResult:
        if not self.bundle.chunks or self.bm25 is None:
            return RetrievalResult(AskMode.EMPTY, [], 0.0)

        # 向量召回
        try:
            qvec = (await self.embedder.embed([query]))[0]
        except Exception:
            qvec = None

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

        # 候选集 = 向量 top_k 与 bm25 top_k 的并集
        bm25_top_idx = np.argsort(bm25_scores)[::-1][: self.top_k].tolist()
        candidate_ids = list({i for i, _ in vec_pairs} | set(bm25_top_idx))

        vec_lookup = {i: s for i, s in vec_pairs}
        cand_vec = [vec_lookup.get(i, 0.0) for i in candidate_ids]
        cand_bm = [float(bm25_scores[i]) for i in candidate_ids]
        nv = _normalize(cand_vec)
        nb = _normalize(cand_bm)
        mixed = [
            self.vector_weight * v + self.bm25_weight * b
            for v, b in zip(nv, nb)
        ]

        order = sorted(range(len(candidate_ids)), key=lambda k: mixed[k], reverse=True)
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
