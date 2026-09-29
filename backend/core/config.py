
import os
import logging
from pathlib import Path

from dotenv import load_dotenv

# Settings are read at import time, and backend.main imports this module before
# anything else loads .env — so under uvicorn SECRET_KEY silently fell back to
# the public default below and any token signed with it was accepted. Load the
# repo-root .env here; real environment variables (Azure app settings) still win.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

_DEFAULT_SECRET_KEY = "your-secret-key-keep-it-secret"


class Settings:
    PROJECT_NAME: str = "Hayasa.ai"
    SECRET_KEY: str = os.getenv("SECRET_KEY", _DEFAULT_SECRET_KEY)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30  # 30 days
    
    # DB Settings (defaults)
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")
    DB_HOST: str = os.getenv("DB_HOST", "growton-restore-may26.postgres.database.azure.com")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_NAME: str = os.getenv("DB_NAME", "ai_hr_db")

settings = Settings()

if settings.SECRET_KEY == _DEFAULT_SECRET_KEY:
    logging.getLogger(__name__).warning(
        "SECRET_KEY is not set; using the insecure built-in default. Set SECRET_KEY in the environment or .env."
    )

# For backward compatibility with security.py which imports these directly
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
