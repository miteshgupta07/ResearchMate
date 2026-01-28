"""
Configuration Module

Loads database configuration from environment variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(project_root / ".env")


class DatabaseConfig:
    """Database configuration loaded from environment variables."""
    
    HOST: str = os.getenv("DB_HOST", "localhost")
    PORT: str = os.getenv("DB_PORT", "5432")
    NAME: str = os.getenv("DB_NAME", "researchmate_db")
    USER: str = os.getenv("DB_USER", "postgres")
    PASSWORD: str = os.getenv("DB_PASSWORD", "")
    
    @classmethod
    def get_connection_url(cls) -> str:
        """
        Get the PostgreSQL connection URL for SQLAlchemy.
        
        Returns:
            PostgreSQL connection string in SQLAlchemy format
        """
        return f"postgresql+psycopg2://{cls.USER}:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}/{cls.NAME}"
