import argparse
import asyncio
import hashlib
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Protocol

import faiss
import numpy as np

from core.chunker import chunk_markdown_file
from core.config import load_settings
from core.embedder import OpenAIEmbedder
from core.models import Chunk


class _EmbProto(Protocol):
    async def embed(self, texts): ...


class IndexBundle:
    def __init__(self, faiss_index, chunks: list[Chunk], embedding_model: str):
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.embedding_model = embedding_model


def _file_hash(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _embed_text_for_chunk(c: Chunk) -> str:
    """嵌入时把面包屑拼到正文前面，提升短 chunk 的可识别性。"""
    crumb = " > ".join(c.heading_path) if c.heading_path else ""
    return f"{crumb}: {c.text}" if crumb else c.text


def _scan_notes(notes_dir: Path) -> list[Path]:
    return sorted(p for p in notes_dir.rglob("*.md") if p.is_file())


async def build_or_update_index(
    notes_dir: Path,
    data_dir: Path,
    embedder: _EmbProto,
    max_tokens: int,
    overlap_tokens: int,
) -> IndexBundle:
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_dir / "meta.json"
    chunks_path = data_dir / "chunks.json"
    index_path = data_dir / "index.pkl"

    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    files_hash: dict[str, str] = meta.get("files_hash", {})

    files = _scan_notes(notes_dir)
    current_hash = {str(p.relative_to(notes_dir)): _file_hash(p) for p in files}

    # 读旧 chunks（若有）
    old_chunks_by_file: dict[str, list[Chunk]] = {}
    if chunks_path.exists():
        try:
            data = json.loads(chunks_path.read_text(encoding="utf-8"))
            for d in data:
                c = Chunk(**d)
                old_chunks_by_file.setdefault(c.file_path, []).append(c)
        except (json.JSONDecodeError, TypeError):
            old_chunks_by_file = {}

    # 决定哪些文件需要重切
    new_chunks: list[Chunk] = []
    chunks_to_embed: list[Chunk] = []
    reuse_vectors: dict[str, list[float]] = {}

    # 加载旧向量 → reuse 池
    if index_path.exists():
        try:
            with open(index_path, "rb") as f:
                old_payload = pickle.load(f)
            for c, vec in zip(old_payload["chunks"], old_payload["vectors"]):
                reuse_vectors[c["id"]] = vec
        except Exception:
            reuse_vectors = {}

    for p in files:
        rel = str(p.relative_to(notes_dir))
        unchanged = files_hash.get(rel) == current_hash[rel] and rel in old_chunks_by_file
        if unchanged:
            for c in old_chunks_by_file[rel]:
                new_chunks.append(c)
        else:
            for c in chunk_markdown_file(p, rel, max_tokens, overlap_tokens):
                new_chunks.append(c)
                if c.id not in reuse_vectors:
                    chunks_to_embed.append(c)

    # 嵌入未命中
    if chunks_to_embed:
        texts = [_embed_text_for_chunk(c) for c in chunks_to_embed]
        vecs = await embedder.embed(texts)
        for c, v in zip(chunks_to_embed, vecs):
            reuse_vectors[c.id] = v

    # 组 FAISS
    if not new_chunks:
        index = faiss.IndexFlatIP(1)
        bundle = IndexBundle(index, [], meta.get("embedding_model", ""))
        meta_path.write_text(json.dumps({"files_hash": current_hash, "embedding_model": "", "last_indexed_at": int(time.time())}), encoding="utf-8")
        chunks_path.write_text(json.dumps([], ensure_ascii=False), encoding="utf-8")
        with open(index_path, "wb") as f:
            pickle.dump({"chunks": [], "vectors": []}, f)
        return bundle

    matrix = np.array([reuse_vectors[c.id] for c in new_chunks], dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    # 持久化
    with open(index_path, "wb") as f:
        pickle.dump(
            {"chunks": [asdict(c) for c in new_chunks], "vectors": matrix.tolist()},
            f,
        )
    chunks_path.write_text(json.dumps([asdict(c) for c in new_chunks], ensure_ascii=False), encoding="utf-8")
    meta_path.write_text(
        json.dumps({
            "files_hash": current_hash,
            "embedding_model": getattr(embedder, "model", ""),
            "last_indexed_at": int(time.time()),
        }),
        encoding="utf-8",
    )
    return IndexBundle(index, new_chunks, getattr(embedder, "model", ""))


def load_index(data_dir: Path) -> IndexBundle | None:
    index_path = data_dir / "index.pkl"
    meta_path = data_dir / "meta.json"
    if not index_path.exists() or not meta_path.exists():
        return None
    try:
        with open(index_path, "rb") as f:
            payload = pickle.load(f)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        # 备份损坏文件
        backup = index_path.with_suffix(f".broken-{int(time.time())}")
        index_path.rename(backup)
        return None
    chunks = [Chunk(**c) for c in payload["chunks"]]
    if not chunks:
        return IndexBundle(faiss.IndexFlatIP(1), [], meta.get("embedding_model", ""))
    matrix = np.array(payload["vectors"], dtype="float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return IndexBundle(index, chunks, meta.get("embedding_model", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="忽略旧索引，全量重建")
    args = parser.parse_args()

    settings = load_settings()
    settings.validate_for_runtime()
    data_dir = Path(__file__).resolve().parent.parent / "data"

    if args.rebuild:
        for f in ["index.pkl", "chunks.json", "meta.json"]:
            p = data_dir / f
            if p.exists():
                p.unlink()

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        cache_path=data_dir / "embedding_cache.json",
    )

    bundle = asyncio.run(build_or_update_index(
        notes_dir=settings.notes_dir,
        data_dir=data_dir,
        embedder=embedder,
        max_tokens=settings.max_chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    ))
    print(f"Indexed {len(bundle.chunks)} chunks from {settings.notes_dir}")


if __name__ == "__main__":
    main()
