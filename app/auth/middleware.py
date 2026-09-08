"""Middleware и зависимости для авторизации.

Логика:
    - Если APP_PASSWORD пуст → авторизация отключена (только localhost).
    - Если клиент с 127.0.0.1 → пропускаем без пароля (локальная разработка).
    - Иначе требуется валидная cookie _session.
    - API health/config эндпоинты всегда открыты (для мониторинга).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

from fastapi import Request
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings

# Срок жизни сессии: 30 дней (в секундах).
SESSION_TTL = 30 * 24 * 3600
COOKIE_NAME = "_session"

# Пути, которые всегда открыты (без авторизации).
PUBLIC_PATHS = {
    "/login", "/api/auth/login", "/api/health",
    "/static",  # префикс
}

# Секрет для подписи cookie (не меняется между перезапусками, т.к. на основе пароля).
_SESSION_SECRET = hashlib.sha256(
    f"{settings.app_password}:kronos-trading-v1".encode()
).hexdigest()


def _is_local(request: Request) -> bool:
    """Запрос пришёл локально (без прокси/туннеля)?

    Локальная разработка: запрос с 127.0.0.1 БЕЗ заголовков прокси.
    Если запрос прошёл через Cloudflare Tunnel или другой прокси —
    он НЕ считается локальным (есть CF-* или X-Forwarded-* заголовки).
    """
    # Заголовки прокси/туннеля есть → запрос не локальный.
    proxy_headers = (
        "cf-connecting-ip", "cf-ipcountry", "cf-ray",
        "x-forwarded-for", "x-real-ip",
    )
    if any(h in request.headers for h in proxy_headers):
        return False
    # Заголовков нет — проверяем прямой client IP.
    client = request.client
    if client is None:
        return False
    return client.host in ("127.0.0.1", "::1", "localhost")


def _auth_enabled() -> bool:
    """Авторизация включена (задан пароль)?"""
    return bool(settings.app_password)


def _make_session_token() -> str:
    """Создать подписанный токен сессии: payload.signature."""
    expires_at = int(time.time()) + SESSION_TTL
    payload = str(expires_at)
    sig = hmac.new(_SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _verify_session_token(token: str) -> bool:
    """Проверить токен сессии: подпись валидна и не истекла."""
    if not token or "." not in token:
        return False
    payload, sig = token.rsplit(".", 1)
    expected_sig = hmac.new(_SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return False
    try:
        expires_at = int(payload)
    except ValueError:
        return False
    return time.time() < expires_at


def verify_password(password: str) -> bool:
    """Проверить пароль пользователя (исп. при логине)."""
    if not _auth_enabled():
        return True
    return hmac.compare_digest(password, settings.app_password)


def is_authenticated(request: Request) -> bool:
    """Авторизован ли текущий запрос?"""
    # Авторизация выключена — пропускаем всех.
    if not _auth_enabled():
        return True
    # Локальный запрос — пропускаем.
    if _is_local(request):
        return True
    # Проверяем cookie.
    token = request.cookies.get(COOKIE_NAME)
    return _verify_session_token(token) if token else False


def set_session_cookie(response, secure: bool = False) -> None:
    """Установить cookie сессии на ответ."""
    token = _make_session_token()
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL,
        httponly=True,
        secure=secure,          # True только на HTTPS (Cloudflare даёт HTTPS).
        samesite="lax",
    )


def clear_session_cookie(response) -> None:
    """Удалить cookie сессии."""
    response.delete_cookie(COOKIE_NAME)


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware: перенаправляет неавторизованных на /login.

    Для HTMX-запросов (POST к /coin/.../auto и т.д.) отдаёт 401 вместо
    редиректа, чтобы HTMX не вставил HTML-страницу логина в target.
    """

    async def dispatch(self, request: Request, call_next):
        # Авторизация выключена — пропускаем.
        if not _auth_enabled():
            return await call_next(request)

        # Локальный запрос — пропускаем без пароля.
        if _is_local(request):
            return await call_next(request)

        path = request.url.path

        # Публичные пути — пропускаем.
        if path in ("/login", "/api/auth/login", "/api/health", "/logout") \
                or path.startswith("/static"):
            return await call_next(request)

        # Уже авторизован — пропускаем.
        if is_authenticated(request):
            return await call_next(request)

        # HTMX-запросы (AJAX) — отдаём 401, чтобы не ломать UI.
        if request.headers.get("hx-request") == "true" or path.startswith("/api/"):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                {"detail": "Не авторизован"},
                status_code=401,
            )

        # Обычный запрос — редирект на логин.
        return RedirectResponse(url="/login", status_code=303)
