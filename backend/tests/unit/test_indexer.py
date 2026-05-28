from pathlib import Path
from unittest.mock import AsyncMock
import pytest

from core.indexer import build_or_update_index, IndexBundle


class FakeEmbedder:
    model = "fake-embedding"

    def __init__(self):
        self.calls = 0
    async def embed(self, texts):
        self.calls += 1
        # 1536 维 dummy 向量，前两个值用文本 hash 做区分
        out = []
        for t in texts:
            h = hash(t)
            v = [float((h >> i) & 1) for i in range(1536)]
            out.append(v)
        return out


@pytest.mark.asyncio
async def test_build_index_from_fixtures(tmp_path):
    notes = Path(__file__).parent.parent / "fixtures" / "notes_sample"
    data_dir = tmp_path / "data"
    embedder = FakeEmbedder()

    bundle = await build_or_update_index(
        notes_dir=notes,
        data_dir=data_dir,
        embedder=embedder,
        max_tokens=500,
        overlap_tokens=50,
    )

    assert isinstance(bundle, IndexBundle)
    assert len(bundle.chunks) >= 4    # 三个文件，至少各 1 段
    assert bundle.faiss_index.ntotal == len(bundle.chunks)
    assert (data_dir / "index.pkl").exists()
    assert (data_dir / "chunks.json").exists()
    assert (data_dir / "meta.json").exists()


@pytest.mark.asyncio
async def test_incremental_skips_unchanged(tmp_path):
    notes = Path(__file__).parent.parent / "fixtures" / "notes_sample"
    data_dir = tmp_path / "data"
    embedder = FakeEmbedder()

    await build_or_update_index(notes, data_dir, embedder, 500, 50)
    first_calls = embedder.calls

    # 第二次同样输入：embed 不再被调用（chunks 全部命中缓存层 / 文件未变）
    await build_or_update_index(notes, data_dir, embedder, 500, 50)
    assert embedder.calls == first_calls


@pytest.mark.asyncio
async def test_modified_file_reindexes(tmp_path):
    # 复制 fixtures 到可写目录
    src = Path(__file__).parent.parent / "fixtures" / "notes_sample"
    notes = tmp_path / "notes"
    notes.mkdir()
    for p in src.rglob("*.md"):
        dst = notes / p.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

    data_dir = tmp_path / "data"
    embedder = FakeEmbedder()
    bundle1 = await build_or_update_index(notes, data_dir, embedder, 500, 50)
    n1 = len(bundle1.chunks)

    # 修改一个文件
    target = next(notes.rglob("controller-runtime.md"))
    target.write_text(target.read_text() + "\n\n## 新章节\n\n新内容。", encoding="utf-8")

    bundle2 = await build_or_update_index(notes, data_dir, embedder, 500, 50)
    assert len(bundle2.chunks) == n1 + 1


class FakeEmbedderB:
    model = "fake-b"

    def __init__(self):
        self.calls = 0
    async def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            h = hash(t)
            v = [float((h >> i) & 1) for i in range(1536)]
            out.append(v)
        return out


@pytest.mark.asyncio
async def test_changing_embedder_model_forces_reembed(tmp_path):
    notes = Path(__file__).parent.parent / "fixtures" / "notes_sample"
    data_dir = tmp_path / "data"

    embedder_a = FakeEmbedder()
    embedder_a.model = "fake-a"
    await build_or_update_index(notes, data_dir, embedder_a, 500, 50)
    assert embedder_a.calls > 0

    embedder_b = FakeEmbedderB()
    bundle = await build_or_update_index(notes, data_dir, embedder_b, 500, 50)

    # 即使没有文件改变，也应该因为 model 变化而全部重新 embed
    assert embedder_b.calls > 0
    assert bundle.faiss_index.ntotal == len(bundle.chunks)
