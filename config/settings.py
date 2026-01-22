"""
Configuration settings for Prediction Markets AI System.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Polymarket
    polymarket_private_key: Optional[str] = Field(default=None, alias="POLYMARKET_PRIVATE_KEY")
    polymarket_clob_url: str = Field(default="https://clob.polymarket.com", alias="POLYMARKET_CLOB_URL")
    polymarket_gamma_url: str = Field(default="https://gamma-api.polymarket.com", alias="POLYMARKET_GAMMA_URL")
    polymarket_data_url: str = Field(default="https://data-api.polymarket.com", alias="POLYMARKET_DATA_URL")

    # Kalshi
    kalshi_api_key: Optional[str] = Field(default=None, alias="KALSHI_API_KEY")
    kalshi_private_key: Optional[str] = Field(default=None, alias="KALSHI_PRIVATE_KEY")
    kalshi_use_demo: bool = Field(default=True, alias="KALSHI_USE_DEMO")
    kalshi_api_url: str = Field(default="https://api.kalshi.com/trade-api/v2", alias="KALSHI_API_URL")
    kalshi_demo_url: str = Field(default="https://demo-api.kalshi.com/trade-api/v2", alias="KALSHI_DEMO_URL")

    # Database
    database_url: str = Field(default="sqlite:///./data/prediction_markets.db", alias="DATABASE_URL")

    # AI
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Alerting
    slack_webhook_url: Optional[str] = Field(default=None, alias="SLACK_WEBHOOK_URL")
    discord_webhook_url: Optional[str] = Field(default=None, alias="DISCORD_WEBHOOK_URL")

    # Application
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    collection_interval: int = Field(default=60, alias="COLLECTION_INTERVAL")
    dashboard_port: int = Field(default=8501, alias="DASHBOARD_PORT")
    paper_trading: bool = Field(default=True, alias="PAPER_TRADING")

    @property
    def kalshi_base_url(self) -> str:
        """Get the appropriate Kalshi URL based on demo setting."""
        return self.kalshi_demo_url if self.kalshi_use_demo else self.kalshi_api_url

    @property
    def has_polymarket_credentials(self) -> bool:
        """Check if Polymarket credentials are configured."""
        return self.polymarket_private_key is not None

    @property
    def has_kalshi_credentials(self) -> bool:
        """Check if Kalshi credentials are configured."""
        return self.kalshi_api_key is not None and self.kalshi_private_key is not None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
