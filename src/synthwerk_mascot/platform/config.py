from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables prefixed with SYNTHWERK_MASCOT_.

    The mascot has no secrets: no database, no token, no outbound call.
    """

    model_config = SettingsConfigDict(
        env_prefix="SYNTHWERK_MASCOT_", env_file=".env", extra="ignore", populate_by_name=True
    )

    platform_env: Literal["local", "ci", "dev", "stg", "prd"] = Field(
        default="local",
        validation_alias="SYNTHWERK_PLATFORM_ENV",
        description="Shared platform setting (no service prefix): the deployment environment.",
    )
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = Field(
        default="json", description="console is used only when the environment is local."
    )
    commit: str = Field(
        default="0000000",
        pattern=r"^[0-9a-f]{7,40}$",
        description="Git commit of the build, shown by /readyz. The image sets it at build time.",
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Browser origins allowed to call POST /v1/answers. Never '*' in production.",
    )
    model_dir: Path = Field(
        default=Path("models/multilingual-e5-small"),
        description="model.onnx and tokenizer.json of multilingual-e5-small.",
    )
    index_dir: Path = Field(
        default=Path("index"),
        description="vectors.f32, chunks.json and manifest.json, written by the image build.",
    )
    ort_threads: int = Field(
        default=2, ge=0, description="Threads of the ONNX Runtime session; 0 = ORT default."
    )
    top_k: int = Field(default=3, ge=1, le=10, description="Most passages in one answer.")
    min_score: float = Field(
        default=0.78,
        ge=0,
        le=1,
        description="Lowest cosine score a passage needs. e5 scores sit high (about 0.7 to 0.9).",
    )
    min_margin: float = Field(
        default=0.045,
        ge=0,
        le=1,
        description="The best score must beat the median score of all chunks by this much. "
        "An off-topic question scores about the same on every chunk.",
    )
    max_question_chars: int = Field(default=300, ge=10, le=2000)
    rate_limit_per_client: int = Field(
        default=6, ge=1, description="Questions one client may ask in one window."
    )
    rate_limit_global: int = Field(
        default=60, ge=1, description="Questions all clients together may ask in one window."
    )
    rate_limit_window_s: float = Field(default=60.0, gt=0)
    rate_limit_max_clients: int = Field(
        default=10_000,
        ge=1,
        description="Clients the limiter tracks at once; past that, new clients get 429.",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
