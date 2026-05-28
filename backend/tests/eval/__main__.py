import asyncio
from pathlib import Path

import yaml

from core.config import load_settings
from core.embedder import OpenAIEmbedder
from core.indexer import build_or_update_index, load_index
from core.retriever import Retriever


async def main():
    settings = load_settings()
    settings.validate_for_runtime()
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    embedder = OpenAIEmbedder(settings.openai_api_key, settings.embedding_model, data_dir / "embedding_cache.json")
    bundle = load_index(data_dir) or await build_or_update_index(
        settings.notes_dir, data_dir, embedder, settings.max_chunk_tokens, settings.chunk_overlap_tokens,
    )

    cases = yaml.safe_load((Path(__file__).parent / "cases.yaml").read_text(encoding="utf-8"))
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    summary = {t: {"mode_correct": 0, "file_hit": 0, "total": 0} for t in thresholds}

    for t in thresholds:
        retr = Retriever(bundle, embedder, settings.retrieval_top_k, settings.rerank_top_k, t,
                         settings.vector_weight, settings.bm25_weight)
        for case in cases:
            res = await retr.search(case["query"])
            summary[t]["total"] += 1
            if res.mode.value == case["expected_mode"]:
                summary[t]["mode_correct"] += 1
            if "expected_files_contain" in case:
                paths = [c.chunk.file_path for c in res.chunks]
                if any(any(want in p for p in paths) for want in case["expected_files_contain"]):
                    summary[t]["file_hit"] += 1

    print(f"{'threshold':>10}  mode-acc  file-hit")
    for t, s in summary.items():
        n = max(s["total"], 1)
        print(f"{t:>10}  {s['mode_correct']/n:.2%}    {s['file_hit']/n:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
