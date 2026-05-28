import asyncio
import json
import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from core.models import AskMode
from core.prompts import fallback_prompt, hit_prompt

log = logging.getLogger(__name__)
router = APIRouter()


def _sse(event: str, data) -> bytes:
    payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


@router.get("/api/ask")
async def ask(request: Request, query: str):
    settings = request.app.state.settings
    retriever = request.app.state.retriever
    llm = request.app.state.llm
    resume_text: str = request.app.state.resume_text

    async def gen():
        t0 = time.monotonic()
        try:
            result = await retriever.search(query)
        except Exception as e:
            log.exception("retrieve failed")
            yield _sse("error", {"stage": "retrieve", "message": str(e), "recoverable": False})
            return

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        yield _sse("mode", {"mode": result.mode.value, "top1_score": result.top1_score, "elapsed_ms": elapsed_ms})
        yield _sse("chunks", [
            {
                "id": sc.chunk.id,
                "file_path": sc.chunk.file_path,
                "heading_path": sc.chunk.heading_path,
                "text": sc.chunk.text,
                "score": sc.score,
                "line_start": sc.chunk.line_start,
                "line_end": sc.chunk.line_end,
            }
            for sc in result.chunks
        ])

        if result.mode == AskMode.EMPTY:
            if not resume_text:
                yield _sse("done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)})
                return
            prompt = fallback_prompt(query, resume_text, [])
        elif result.mode == AskMode.FALLBACK:
            prompt = fallback_prompt(query, resume_text, result.chunks)
        else:
            prompt = hit_prompt(query, result.chunks)

        try:
            first_token_deadline = settings.llm_first_token_timeout_s
            total_deadline = settings.llm_total_timeout_s
            stream = llm.stream(prompt)
            first = True

            async def _next():
                return await stream.__anext__()

            stream_task_started = time.monotonic()
            while True:
                if await request.is_disconnected():
                    log.info("client disconnected, abort llm stream")
                    return
                timeout = first_token_deadline if first else max(0.1, total_deadline - (time.monotonic() - stream_task_started))
                try:
                    tok = await asyncio.wait_for(_next(), timeout=timeout)
                except asyncio.TimeoutError:
                    yield _sse("error", {"stage": "llm", "message": "timeout", "recoverable": False})
                    return
                except StopAsyncIteration:
                    break
                first = False
                yield _sse("token", tok)
        except Exception as e:
            log.exception("llm stream failed")
            yield _sse("error", {"stage": "llm", "message": str(e), "recoverable": False})
            return

        yield _sse("done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)})

    return StreamingResponse(gen(), media_type="text/event-stream")
