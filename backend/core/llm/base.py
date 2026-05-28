from typing import AsyncIterator, Protocol


class LLMProvider(Protocol):
    async def stream(self, prompt: str) -> AsyncIterator[str]: ...


def build_provider(name: str, *, api_key: str, model: str) -> "LLMProvider":
    if name == "deepseek":
        from core.llm.deepseek import DeepSeekProvider
        return DeepSeekProvider(api_key=api_key, model=model)
    if name == "openai":
        from core.llm.openai import OpenAIProvider
        return OpenAIProvider(api_key=api_key, model=model)
    if name == "claude":
        from core.llm.claude import ClaudeProvider
        return ClaudeProvider(api_key=api_key, model=model)
    raise ValueError(f"Unknown LLM provider: {name}")
