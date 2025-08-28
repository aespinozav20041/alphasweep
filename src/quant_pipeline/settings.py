"""Application settings loaded from environment variables and .env file."""

from pathlib import Path
from pydantic import BaseSettings


class Settings(BaseSettings):
    env_name: str = "dev"

    class Config:
        env_file = Path(__file__).resolve().parents[2] / "conf" / ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Get application settings."""

    return Settings()


settings = get_settings()
