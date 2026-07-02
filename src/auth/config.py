# src/auth/config.py
"""Authentication and security configuration.

All sensitive values are loaded from environment variables.
Non-sensitive defaults are defined here as constants.
"""

import os
from pathlib import Path

# Load .env if not already loaded (reuses pattern from src/core/config.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_env_path = _PROJECT_ROOT / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                if key not in os.environ:
                    os.environ[key] = val


# ── JWT Settings ─────────────────────────────────────────────────────── #

JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "CHANGE-ME-IN-PRODUCTION-use-openssl-rand-hex-32")
JWT_ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
REFRESH_TOKEN_EXPIRE_DAYS: int = 7


# ── Cookie Settings ──────────────────────────────────────────────────── #

ACCESS_TOKEN_COOKIE_NAME: str = "access_token"
REFRESH_TOKEN_COOKIE_NAME: str = "refresh_token"
COOKIE_DOMAIN: str | None = os.getenv("COOKIE_DOMAIN", None)
COOKIE_SECURE: bool = os.getenv("COOKIE_SECURE", "false").lower() == "true"
COOKIE_SAMESITE: str = "lax"
COOKIE_PATH: str = "/"


# ── Password Hashing (Argon2) ────────────────────────────────────────── #

ARGON2_TIME_COST: int = 3
ARGON2_MEMORY_COST: int = 65536  # 64 MiB
ARGON2_PARALLELISM: int = 4
ARGON2_HASH_LEN: int = 32
ARGON2_SALT_LEN: int = 16


# ── Account Lockout ──────────────────────────────────────────────────── #

MAX_LOGIN_ATTEMPTS: int = 5
LOCKOUT_DURATION_MINUTES: int = 15


# ── Email Verification & Password Reset ──────────────────────────────── #

EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS: int = 24
PASSWORD_RESET_TOKEN_EXPIRE_HOURS: int = 1


# ── CORS (Auth-specific) ─────────────────────────────────────────────── #

ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
