
import os

# Security Configuration
SECRET_KEY = "your-secret-key-keep-it-secret"  # In production, use env var
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
