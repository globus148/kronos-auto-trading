@echo off
REM ============================================================
REM  Kronos Auto-Trading - запуск приложения (Windows)
REM  Двойной клик по этому файлу = запуск сайта.
REM ============================================================

chcp 65001 >nul
cd /d "%~dp0"

echo.
echo ============================================================
echo   Kronos Auto-Trading
echo ============================================================
echo.

REM Проверяем Python.
where python >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python не найден в PATH.
    echo Установите Python 3.11+ и добавьте в PATH.
    pause
    exit /b 1
)

REM Создаём папку для БД, если её нет.
if not exist "data" mkdir data

REM Создаём .env из примера, если его нет.
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo [INFO] Создан .env из .env.example
    )
)

REM Проверяем fastapi (индикатор что зависимости установлены).
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Устанавливаю зависимости из requirements.txt...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось установить зависимости.
        pause
        exit /b 1
    )
)

echo.
echo   Открой в браузере:  http://127.0.0.1:8000
echo   Остановить:        Ctrl+C в этом окне
echo.
echo ------------------------------------------------------------
echo.

REM Запускаем сервер.
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

pause
