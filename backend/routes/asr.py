import asyncio
import logging
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/asr")
async def asr(request: Request, file: UploadFile = File(...)):
    settings = request.app.state.settings
    asr_client = request.app.state.asr
    config_error = getattr(request.app.state, "config_error", None)
    if asr_client is None:
        raise HTTPException(status_code=503, detail=f"ASR unavailable: {config_error or 'not configured'}")
    try:
        audio = await file.read()
        from io import BytesIO
        text = await asyncio.wait_for(
            asr_client.transcribe(
                BytesIO(audio), filename=file.filename or "audio.webm",
                mime=file.content_type or "audio/webm",
            ),
            timeout=settings.asr_timeout_s,
        )
        return {"text": text}
    except asyncio.TimeoutError:
        log.warning("ASR timeout")
        raise HTTPException(status_code=504, detail="ASR timeout")
    except Exception as e:
        log.exception("ASR failed")
        raise HTTPException(status_code=502, detail=f"ASR failed: {e}")
