import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from core.asr import WhisperASR
from core.config import load_settings
from core.embedder import OpenAIEmbedder
from core.indexer import build_or_update_index, load_index
from core.llm import build_provider
from core.resume import load_resume
from core.retriever import Retriever
from routes.asr import router as asr_router
from routes.ask import router as ask_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def _provider_api_key(settings) -> str:
    if settings.llm_provider == "deepseek":
        return settings.deepseek_api_key
    if settings.llm_provider == "openai":
        return settings.openai_api_key
    if settings.llm_provider == "claude":
        return settings.anthropic_api_key
    raise ValueError(f"Unknown provider: {settings.llm_provider}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    app.state.config_error = None
    app.state.retriever = None
    app.state.llm = None
    app.state.asr = None
    app.state.resume_text = ""

    try:
        settings.validate_for_runtime()
    except ValueError as e:
        log.warning("Config invalid, running in degraded mode: %s", e)
        app.state.config_error = str(e)
        yield
        return

    try:
        data_dir = Path(__file__).resolve().parent / "data"
        embedder = OpenAIEmbedder(
            api_key=settings.effective_embedding_key,
            model=settings.embedding_model,
            cache_path=data_dir / "embedding_cache.json",
            base_url=settings.embedding_base_url or None,
        )
        bundle = load_index(data_dir)
        if bundle is None or not bundle.chunks:
            log.info("No usable index, building from %s", settings.notes_dir)
            bundle = await build_or_update_index(
                settings.notes_dir, data_dir, embedder,
                settings.max_chunk_tokens, settings.chunk_overlap_tokens,
            )
        retriever = Retriever(
            bundle=bundle, embedder=embedder,
            top_k=settings.retrieval_top_k,
            rerank_top_k=settings.rerank_top_k,
            threshold=settings.retrieval_threshold,
            vector_weight=settings.vector_weight,
            bm25_weight=settings.bm25_weight,
            embed_timeout=settings.embed_timeout_s,
        )
        llm = build_provider(settings.llm_provider, api_key=_provider_api_key(settings), model=settings.llm_model)
        asr_client = WhisperASR(
            api_key=settings.openai_api_key,
            model=settings.asr_model,
            timeout=settings.asr_timeout_s + 2.0,   # inner > route wait_for
        )
        resume_text = load_resume(settings.resume_path)

        app.state.retriever = retriever
        app.state.llm = llm
        app.state.asr = asr_client
        app.state.resume_text = resume_text
        log.info("Ready: %d chunks, provider=%s", len(bundle.chunks), settings.llm_provider)
    except Exception as e:
        log.exception("Startup failed; running in degraded mode")
        app.state.config_error = f"Startup error: {e}"
    yield


app = FastAPI(title="Interview RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(asr_router)
app.include_router(ask_router)


@app.get("/api/health")
def health(request: Request):
    err = getattr(request.app.state, "config_error", None)
    return {"ok": err is None, "config_error": err}
