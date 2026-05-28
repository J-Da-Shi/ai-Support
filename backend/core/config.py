from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 路径
    notes_dir: Path = Field(default=Path("./notes"))
    resume_path: Path = Field(default=Path("./resume.md"))

    # API keys
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    anthropic_api_key: str = ""
    embedding_api_key: str = ""        # 优先用于嵌入；空则回落 openai_api_key

    # base_url 自定义（OpenAI 兼容网关：阿里云 DashScope / SiliconFlow / 自建）
    embedding_base_url: str = ""
    llm_base_url: str = ""              # 仅 OpenAI provider 用

    # 模型
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 0              # 0 = 自动推断；阿里 v4 可指定
    llm_provider: str = "deepseek"      # deepseek | openai | claude
    llm_model: str = "deepseek-chat"
    asr_provider: str = "browser"       # browser (Web Speech) | openai (Whisper)
    asr_model: str = "whisper-1"

    # 检索
    retrieval_top_k: int = 8
    rerank_top_k: int = 3
    retrieval_threshold: float = 0.5
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    max_chunk_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # 超时
    asr_timeout_s: float = 8.0
    embed_timeout_s: float = 5.0
    llm_first_token_timeout_s: float = 10.0
    llm_total_timeout_s: float = 30.0

    @property
    def effective_embedding_key(self) -> str:
        return self.embedding_api_key or self.openai_api_key

    def validate_for_runtime(self) -> "Settings":
        """检查所选 provider 的 API key 是否就位。"""
        provider_key = {
            "deepseek": ("DEEPSEEK_API_KEY", self.deepseek_api_key),
            "openai": ("OPENAI_API_KEY", self.openai_api_key),
            "claude": ("ANTHROPIC_API_KEY", self.anthropic_api_key),
        }
        if self.llm_provider not in provider_key:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.llm_provider}")
        name, val = provider_key[self.llm_provider]
        if not val:
            raise ValueError(f"{name} required for LLM_PROVIDER={self.llm_provider}")
        if self.asr_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for ASR_PROVIDER=openai")
        if not self.effective_embedding_key:
            raise ValueError(
                "EMBEDDING_API_KEY (or OPENAI_API_KEY) required for embeddings"
            )
        return self


def load_settings() -> Settings:
    return Settings()
