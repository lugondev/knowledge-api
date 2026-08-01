"""Bearer key -> tenant. The only place a tenant is ever decided."""

from __future__ import annotations

from fastapi import HTTPException, Request

UNAUTHORIZED = HTTPException(
    status_code=401,
    detail="missing or invalid credential",
    headers={"WWW-Authenticate": "Bearer"},
)


def require_tenant(request: Request) -> str:
    header = request.headers.get("authorization") or ""
    scheme, _, key = header.partition(" ")
    if scheme.lower() != "bearer" or not key.strip():
        raise UNAUTHORIZED
    tenant = request.app.state.settings.api_keys.get(key.strip())
    if not tenant:
        raise UNAUTHORIZED
    return tenant
