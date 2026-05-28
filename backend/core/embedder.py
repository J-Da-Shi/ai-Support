import hashlib
import json
from pathlib import Path
from typing import Sequence

from openai import AsyncOpenAI


def _key(model: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update(text.encode())
    return h.hexdigest()


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str,
        cache_path: Path,
        batch_size: int = 100,
        base_url: str | None = None,
        _client=None,
    ):
        self.model = model
        self.cache_path = cache_path
        self.batch_size = batch_size
        self._cache: dict[str, list[float]] = {}
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._cache = {}
        if _client is not None:
            self._client = _client
        else:
            kwargs: dict = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = AsyncOpenAI(**kwargs).embeddings

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        misses: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            k = _key(self.model, t)
            cached = self._cache.get(k)
            if cached is not None:
                results[i] = cached
            else:
                misses.append((i, t))

        for start in range(0, len(misses), self.batch_size):
            batch = misses[start : start + self.batch_size]
            inputs = [t for _, t in batch]
            resp = await self._client.create(model=self.model, input=inputs)
            for (orig_idx, txt), e in zip(batch, resp.data):
                vec = list(e.embedding)
                results[orig_idx] = vec
                self._cache[_key(self.model, txt)] = vec

        if misses:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._cache), encoding="utf-8")
            tmp.replace(self.cache_path)

        return [r for r in results]  # type: ignore[return-value]
