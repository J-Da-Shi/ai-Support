import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.embedder import OpenAIEmbedder


@pytest.mark.asyncio
async def test_embed_batches_and_caches(tmp_path):
    cache = tmp_path / "cache.json"
    fake_client = AsyncMock()
    fake_client.create = AsyncMock(return_value=type("R", (), {
        "data": [type("E", (), {"embedding": [0.1, 0.2, 0.3]})() for _ in range(2)]
    })())

    emb = OpenAIEmbedder(api_key="sk-x", model="m", cache_path=cache, _client=fake_client, batch_size=2)

    out1 = await emb.embed(["hello", "world"])
    assert len(out1) == 2 and out1[0] == [0.1, 0.2, 0.3]
    assert fake_client.create.await_count == 1

    # 第二次同样输入：完全命中缓存，不再调用
    out2 = await emb.embed(["hello", "world"])
    assert out2 == out1
    assert fake_client.create.await_count == 1

    # cache 落盘
    saved = json.loads(cache.read_text())
    assert len(saved) == 2


@pytest.mark.asyncio
async def test_embed_partial_cache_hit(tmp_path):
    cache = tmp_path / "cache.json"
    fake_client = AsyncMock()
    # 第一轮：embed 两个，写缓存
    fake_client.create = AsyncMock(return_value=type("R", (), {
        "data": [type("E", (), {"embedding": [1.0]})(), type("E", (), {"embedding": [2.0]})()]
    })())
    emb = OpenAIEmbedder(api_key="sk", model="m", cache_path=cache, _client=fake_client, batch_size=10)
    await emb.embed(["a", "b"])

    # 第二轮：a 命中、c 未命中 → 只对 c 发请求
    fake_client.create = AsyncMock(return_value=type("R", (), {
        "data": [type("E", (), {"embedding": [3.0]})()]
    })())
    out = await emb.embed(["a", "c"])
    assert out == [[1.0], [3.0]]
    fake_client.create.assert_awaited_once()
    assert fake_client.create.call_args.kwargs["input"] == ["c"]
