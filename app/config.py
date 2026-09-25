"""
Centralized application settings, loaded once from environment / .env.
All modules should import `settings` from here instead of calling os.getenv directly.

Modified for the IT support ticket-agent adaptation: optional Jev decision
settings were added for typed intent routing and ticket triage.
"""
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_db"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    max_verification_retries: int = 2

    # Jev / TypeSafe System One decisions
    # Keep this optional: when no key is present or the call fails, the
    # existing LLM and deterministic fallback routes still handle the request.
    jev_api_key: str = ""
    typesafe_api_key: str = ""
    jev_api_url: str = "https://api.typesafe.ai/v1/systemone"
    jev_model: str = "jev-latest"
    jev_timeout_seconds: float = 3.0
    jev_min_confidence: float = 0.62
    jev_human_review_threshold: float = 0.65

    # MCP
    mcp_sqlite_path: str = "enterprise_mcp.db"
    # Used to resolve "my"/"me" in queries like "tickets assigned to me today"
    # until real auth/session identity exists.
    mcp_default_user_id: str = "E001"

    # Streamlit
    backend_url: str = "http://localhost:8000"

    @field_validator("database_url")
    @classmethod
    def _force_async_driver(cls, v: str) -> str:
        """SQLAlchemy's async engine requires an async DBAPI driver. If a
        sync psycopg2 URL slips into .env (e.g. copy-pasted from the
        original notebook), rewrite it to asyncpg instead of crashing at
        engine-creation time."""
        if v.startswith("postgresql+psycopg2://"):
            return v.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
