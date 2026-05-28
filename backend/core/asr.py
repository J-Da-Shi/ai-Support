from typing import BinaryIO

import httpx


class WhisperASR:
    URL = "https://api.openai.com/v1/audio/transcriptions"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def transcribe(self, audio: BinaryIO, filename: str, mime: str, language: str = "zh") -> str:
        files = {"file": (filename, audio, mime)}
        data = {"model": self.model, "language": language}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.URL, headers=headers, files=files, data=data)
            if resp.status_code >= 400:
                raise RuntimeError(f"Whisper API error {resp.status_code}: {resp.text}")
            return resp.json()["text"]
