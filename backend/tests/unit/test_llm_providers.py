import pytest
import respx
import httpx
from core.llm.deepseek import DeepSeekProvider


@pytest.mark.asyncio
@respx.mock
async def test_deepseek_streams_tokens():
    body = (
        b'data: {"choices":[{"delta":{"content":"\xe4\xb8\x80"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"\xe4\xba\x8c"}}]}\n\n'
        b'data: [DONE]\n\n'
    )
    respx.post("https://api.deepseek.com/chat/completions").mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
    )
    provider = DeepSeekProvider(api_key="dk", model="deepseek-chat")
    out = []
    async for tok in provider.stream("hi"):
        out.append(tok)
    assert "".join(out) == "一二"


@pytest.mark.asyncio
@respx.mock
async def test_deepseek_skips_empty_choices_frames():
    body = (
        b'data: {"choices":[]}\n\n'                                    # empty list, must not crash
        b'data: {}\n\n'                                                # missing key
        b'data: {"choices":[{"delta":{"content":"\xe4\xb8\xad"}}]}\n\n'  # 中
        b'data: [DONE]\n\n'
    )
    respx.post("https://api.deepseek.com/chat/completions").mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
    )
    provider = DeepSeekProvider(api_key="dk", model="deepseek-chat")
    out = []
    async for tok in provider.stream("hi"):
        out.append(tok)
    assert "".join(out) == "中"


@pytest.mark.asyncio
@respx.mock
async def test_deepseek_handles_http_error():
    respx.post("https://api.deepseek.com/chat/completions").mock(
        return_value=httpx.Response(500, text="boom")
    )
    provider = DeepSeekProvider(api_key="dk", model="deepseek-chat")
    with pytest.raises(RuntimeError, match="DeepSeek"):
        async for _ in provider.stream("hi"):
            pass
