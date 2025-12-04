"""Application configuration."""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_TITLE: str = "Wood AI CML Optimization API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Machine Learning API for Condition Monitoring Location (CML) optimization"
    
    # Model Settings
    MODEL_DIR: Path = Path("models")
    MODEL_PATH: Optional[Path] = None
    
    # Data Settings
    DATA_DIR: Path = Path("data")
    UPLOAD_DIR: Path = Path("uploads")
    
    # SME Override Settings
    SME_OVERRIDE_FILE: Path = Path("data/sme_overrides.json")
    
    # Forecasting Settings
    DEFAULT_MINIMUM_THICKNESS: float = 3.0  # mm
    DEFAULT_INSPECTION_INTERVAL: int = 36  # months
    SAFETY_FACTOR: float = 1.5
    
    # Feature Engineering
    HIGH_CORROSION_THRESHOLD: float = 0.15  # mm/year
    THIN_WALL_THRESHOLD: float = 8.0  # mm
    
    # Model Training
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5
    
    # Performance
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50 MB
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Create necessary directories
if not os.path.exists(settings.MODEL_DIR):
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
if not os.path.exists(settings.DATA_DIR):
    os.makedirs(settings.DATA_DIR, exist_ok=True)
if not os.path.exists(settings.UPLOAD_DIR):
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)