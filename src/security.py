from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel


class AuthUser(BaseModel):
    user_id: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str


@dataclass(frozen=True)
class SecuritySettings:
    auth_enabled: bool
    jwt_secret: str
    jwt_ttl_seconds: int
    user_username: str
    user_password: str
    admin_username: str
    admin_password: str


def get_security_settings() -> SecuritySettings:
    enabled = os.getenv("JOBPILOT_AUTH_ENABLED", "false").strip().lower() in {"1", "true", "yes"}
    return SecuritySettings(
        auth_enabled=enabled,
        jwt_secret=os.getenv("JOBPILOT_JWT_SECRET", ""),
        jwt_ttl_seconds=max(300, int(os.getenv("JOBPILOT_JWT_TTL_SECONDS", "3600"))),
        user_username=os.getenv("JOBPILOT_USER_USERNAME", ""),
        user_password=os.getenv("JOBPILOT_USER_PASSWORD", ""),
        admin_username=os.getenv("JOBPILOT_ADMIN_USERNAME", ""),
        admin_password=os.getenv("JOBPILOT_ADMIN_PASSWORD", ""),
    )


def _password_matches(provided: str, configured: str) -> bool:
    if configured.startswith("sha256:"):
        digest = hashlib.sha256(provided.encode("utf-8")).hexdigest()
        return hmac.compare_digest(digest, configured.split(":", 1)[1])
    return bool(configured) and hmac.compare_digest(provided, configured)


def authenticate(username: str, password: str) -> AuthUser | None:
    settings = get_security_settings()
    if username == settings.admin_username and _password_matches(password, settings.admin_password):
        return AuthUser(user_id=username, role="admin")
    if username == settings.user_username and _password_matches(password, settings.user_password):
        return AuthUser(user_id=username, role="user")
    return None


def create_access_token(user: AuthUser) -> TokenResponse:
    settings = get_security_settings()
    if not settings.jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="认证已启用，但 JOBPILOT_JWT_SECRET 尚未配置。",
        )
    now = int(time.time())
    payload = {
        "sub": user.user_id,
        "role": user.role,
        "iat": now,
        "exp": now + settings.jwt_ttl_seconds,
        "iss": "jobpilot-agent",
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_ttl_seconds,
        role=user.role,
    )


_BEARER = HTTPBearer(auto_error=False)


def current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER)],
) -> AuthUser:
    settings = get_security_settings()
    if not settings.auth_enabled:
        return AuthUser(user_id="local-admin", role="admin")
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录。")
    if not settings.jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务器认证配置不完整。",
        )
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=["HS256"],
            issuer="jobpilot-agent",
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录凭证无效或已过期。") from exc
    user_id = payload.get("sub")
    role = payload.get("role")
    if not isinstance(user_id, str) or role not in {"user", "admin"}:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录凭证内容无效。")
    return AuthUser(user_id=user_id, role=role)


def admin_user(user: Annotated[AuthUser, Depends(current_user)]) -> AuthUser:
    if not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="该操作需要管理员权限。")
    return user


class SlidingWindowRateLimiter:
    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_seconds: int = 60) -> None:
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            events = self._requests[key]
            while events and events[0] < cutoff:
                events.popleft()
            if len(events) >= limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"请求过于频繁，请在 {window_seconds} 秒后重试。",
                )
            events.append(now)


RATE_LIMITER = SlidingWindowRateLimiter()


def enforce_run_rate_limit(request: Request, user: AuthUser) -> None:
    limit = max(1, int(os.getenv("JOBPILOT_RUNS_PER_MINUTE", "10")))
    client_host = request.client.host if request.client else "unknown"
    RATE_LIMITER.check(f"run:{user.user_id}:{client_host}", limit=limit)
