"""Application configuration.

All tunable behaviour is centralised here and can be overridden through
environment variables or a local ``.env`` file (see ``.env.example``).
Paths are resolved relative to the repository root so the API behaves the
same whether it is started from the repo, from ``/app`` in the container,
or from an arbitrary working directory.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application settings, loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # The ML-related fields below start with "model_", which collides
        # with Pydantic's reserved namespace unless it is opened up.
        protected_namespaces=(),
    )

    # API metadata
    API_TITLE: str = "Wood AI CML Optimization API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = (
        "Machine Learning API for Condition Monitoring Location (CML) optimisation"
    )
    ENVIRONMENT: str = "development"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Browsers block cross-origin calls unless the origin is listed here.
    # The default covers the bundled Streamlit dashboard only; production
    # deployments must set CORS_ORIGINS explicitly. "*" is rejected so a
    # deployment cannot accidentally open the API to every origin.
    CORS_ORIGINS: list[str] = Field(default=["http://localhost:8501"])

    # Filesystem layout
    MODEL_DIR: Path = BASE_DIR / "models"
    MODEL_FILENAME: str = "cml_elimination_model.joblib"
    DATA_DIR: Path = BASE_DIR / "data"
    SME_OVERRIDE_FILE: Path = BASE_DIR / "data" / "sme_overrides.json"

    # Engineering defaults (API 570 style remaining-life calculation)
    DEFAULT_MINIMUM_THICKNESS: float = 3.0  # mm
    DEFAULT_INSPECTION_INTERVAL: int = 36  # months
    SAFETY_FACTOR: float = 1.5

    # Feature engineering thresholds
    HIGH_CORROSION_THRESHOLD: float = 0.15  # mm/year
    THIN_WALL_THRESHOLD: float = 8.0  # mm

    # Model training
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5

    # Upload limits. Uploads are parsed entirely in memory, so this bound
    # is what stops a single request from exhausting the worker's RAM.
    MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024  # 25 MB
    MAX_UPLOAD_ROWS: int = 100_000

    # Number of scored rows echoed back in the response body. The full
    # count is always reported separately as ``total_results``.
    MAX_RESULTS_IN_RESPONSE: int = 100

    LOG_LEVEL: str = "INFO"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string, which is how env vars carry lists."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("CORS_ORIGINS")
    @classmethod
    def _reject_wildcard(cls, value: list[str]) -> list[str]:
        if "*" in value:
            raise ValueError(
                "CORS_ORIGINS must not contain '*'; list the dashboard origins explicitly"
            )
        return value

    @property
    def MODEL_PATH(self) -> Path:
        """Absolute path to the model artifact the API serves."""
        return self.MODEL_DIR / self.MODEL_FILENAME


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    Cached so that importing modules share one object, and so tests can
    clear the cache to load a different environment.
    """
    return Settings()


settings = get_settings()
