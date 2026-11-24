from datetime import UTC, datetime, timedelta

import bcrypt
from jose import jwt

from app.core.config import settings

# ── Password ──────────────────────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── JWT ───────────────────────────────────────────────────────────────────────


def create_access_token(user_id: int, workspace_id: int) -> str:
    expire = datetime.now(UTC) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    payload = {
        "sub": str(user_id),
        "workspace_id": workspace_id,
        "exp": expire,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT.
    Raises jose.JWTError on any validation failure (expired, invalid signature, etc.).
    """
    return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
