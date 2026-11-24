from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Read the repo-root .env (shared with docker-compose) and backend/.env, so
# commands work from either folder. Real environment variables still win.
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_ENV_FILES = (_BACKEND_DIR.parent / ".env", _BACKEND_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://veritas:veritas@localhost:5432/veritas"
    )

    # Redis
    redis_url: str = Field(default="redis://localhost:6379")

    # LLM (Groq) — model names come from env, never hardcoded in logic
    groq_api_key: str = Field(default="")
    llm_model: str = Field(default="openai/gpt-oss-120b")
    llm_fast_model: str = Field(default="openai/gpt-oss-20b")
    # Sent as reasoning_effort to reasoning models (gpt-oss); "" disables it.
    llm_reasoning_effort: str = Field(default="low")

    # Auth
    secret_key: str = Field(default="change-me-in-production")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=1440)

    # Observability (optional — empty string means disabled)
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # Ingestion
    chunk_size: int = Field(default=450)
    chunk_overlap: int = Field(default=60)

    # Retrieval
    retrieval_top_k: int = Field(default=20)
    rerank_top_k: int = Field(default=5)

    # Cache
    exact_cache_ttl: int = Field(default=86400)
    semantic_cache_threshold: float = Field(default=0.95)
    rate_limit_per_hour: int = Field(default=10)

    # MCP Server
    mcp_api_key: str = Field(default="")
    mcp_workspace_id: int = Field(default=1)

    # Public demo (/chat/demo, no login) — workspace that holds sample_docs
    demo_workspace_id: int = Field(default=1)

    # CORS — comma-separated list of allowed frontend origins in production,
    # e.g. "https://veritas.vercel.app". In development all origins are allowed.
    cors_origins: str = Field(default="")

    # App
    environment: str = Field(default="development")
    debug: bool = Field(default=False)

    @property
    def allowed_origins(self) -> list[str]:
        if self.environment == "development":
            return ["*"]
        return [o.strip().rstrip("/") for o in self.cors_origins.split(",") if o.strip()]

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
