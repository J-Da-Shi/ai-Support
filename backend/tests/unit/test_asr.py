import io
import pytest
import respx
import httpx
from core.asr import WhisperASR


@pytest.mark.asyncio
@respx.mock
async def test_whisper_returns_text():
    respx.post("https://api.openai.com/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "你好世界"})
    )
    asr = WhisperASR(api_key="sk", model="whisper-1")
    text = await asr.transcribe(io.BytesIO(b"\x00\x01"), filename="a.webm", mime="audio/webm")
    assert text == "你好世界"


@pytest.mark.asyncio
@respx.mock
async def test_whisper_raises_on_error():
    respx.post("https://api.openai.com/v1/audio/transcriptions").mock(
        return_value=httpx.Response(503, text="busy")
    )
    asr = WhisperASR(api_key="sk", model="whisper-1")
    with pytest.raises(RuntimeError, match="Whisper"):
        await asr.transcribe(io.BytesIO(b"\x00"), filename="a.webm", mime="audio/webm")
