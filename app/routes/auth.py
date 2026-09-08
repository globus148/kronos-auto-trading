"""Роуты авторизации: страница логина, POST-обработка, logout.

Шаблон: app/templates/login.html
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.auth.middleware import (
    COOKIE_NAME,
    clear_session_cookie,
    set_session_cookie,
    verify_password,
)

router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Страница ввода пароля."""
    from app.main import TEMPLATES
    error = request.query_params.get("error", "")
    return TEMPLATES.TemplateResponse("login.html", {
        "request": request,
        "error": error,
    })


@router.post("/api/auth/login")
async def login_submit(password: str = Form(...)):
    """Обработка формы логина. Устанавливает cookie и редиректит на дашборд."""
    if verify_password(password):
        resp = RedirectResponse(url="/dashboard", status_code=303)
        # secure=False: работает и по HTTP (localhost) и по HTTPS (Cloudflare).
        set_session_cookie(resp, secure=False)
        return resp
    # Неверный пароль — обратно на логин с ошибкой.
    return RedirectResponse(url="/login?error=1", status_code=303)


@router.get("/logout")
async def logout():
    """Выход: удалить cookie и редирект на логин."""
    resp = RedirectResponse(url="/login", status_code=303)
    clear_session_cookie(resp)
    return resp
