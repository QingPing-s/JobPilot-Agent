import pytest
from fastapi import HTTPException

from src.security import (
    AuthUser,
    SlidingWindowRateLimiter,
    authenticate,
    create_access_token,
    current_user,
)


def test_authenticate_and_issue_role_token(monkeypatch):
    monkeypatch.setenv("JOBPILOT_AUTH_ENABLED", "true")
    monkeypatch.setenv("JOBPILOT_JWT_SECRET", "test-secret-with-at-least-thirty-two-bytes")
    monkeypatch.setenv("JOBPILOT_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("JOBPILOT_ADMIN_PASSWORD", "secret")

    user = authenticate("admin", "secret")
    assert user == AuthUser(user_id="admin", role="admin")
    token = create_access_token(user)
    assert token.role == "admin"
    assert token.access_token


def test_sliding_window_rate_limiter_rejects_excess_requests():
    limiter = SlidingWindowRateLimiter()
    limiter.check("user", limit=1, window_seconds=60)
    with pytest.raises(HTTPException) as exc_info:
        limiter.check("user", limit=1, window_seconds=60)
    assert exc_info.value.status_code == 429


def test_public_access_uses_anonymous_user_role(monkeypatch):
    monkeypatch.setenv("JOBPILOT_AUTH_ENABLED", "true")
    monkeypatch.setenv("JOBPILOT_PUBLIC_ACCESS", "true")

    user = current_user(None)

    assert user == AuthUser(user_id="anonymous", role="user")
