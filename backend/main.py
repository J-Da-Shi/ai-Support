import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from core.config import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    try:
        settings.validate_for_runtime()
        log.info("Settings validated. provider=%s notes_dir=%s", settings.llm_provider, settings.notes_dir)
    except ValueError as e:
        log.warning("Settings invalid (server still up for guidance): %s", e)
    app.state.settings = settings
    yield


app = FastAPI(title="Interview RAG", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"ok": True}
