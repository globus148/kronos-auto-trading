@echo off
chcp 65001 >nul
title Kronos Trading + Remote Tunnel

echo ============================================================
echo   Kronos Auto-Trading + Remote Tunnel
echo ============================================================
echo.

:: Создаём папку data если нет
if not exist data mkdir data

:: Копируем .env.example в .env если нет
if not exist .env (
    if exist .env.example copy .env.example .env >nul
    echo [OK] Создан .env из .env.example
)

:: Проверяем Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python не найден. Установи Python 3.10+ и перезапусти.
    pause
    exit /b 1
)

:: Проверяем зависимости
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [INFO] Установка зависимостей...
    pip install -r requirements.txt
)

:: Запуск бота в фоне
echo [1/2] Запуск бота на порту 8000...
start /b "" python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > trading.log 2>&1

:: Ждём загрузку (Kronos грузится ~10 сек)
echo [INFO] Ожидание загрузки бота...
timeout /t 12 /nobreak >nul

:: Проверяем что бот поднялся
curl -s -o nul http://127.0.0.1:8000/api/health >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Бот не запустился! Проверь trading.log
    pause
    exit /b 1
)
echo [OK] Бот запущен: http://127.0.0.1:8000

:: Проверяем и настраиваем прокси для ngrok (обход блокировок CRL в РФ)
set "HTTP_PROXY="
set "http_proxy="
netstat -ano | findstr "127.0.0.1:10809" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    set "HTTP_PROXY=http://127.0.0.1:10809"
    set "http_proxy=http://127.0.0.1:10809"
    echo [OK] Прокси Happ (10809) подключен для ngrok (обход блокировок CRL)
) else (
    netstat -ano | findstr "127.0.0.1:7890" | findstr "LISTENING" >nul 2>&1
    if not errorlevel 1 (
        set "HTTP_PROXY=http://127.0.0.1:7890"
        set "http_proxy=http://127.0.0.1:7890"
        echo [OK] Прокси Clash (7890) подключен для ngrok (обход блокировок CRL)
    ) else (
        echo [ВНИМАНИЕ] Happ не обнаружен на порту 10809. Убедитесь, что Happ запущен!
    )
)

:: Завершаем старые зависшие процессы ngrok если есть
taskkill /f /im ngrok.exe >nul 2>&1

:: Запуск ngrok туннеля
echo.
echo [2/2] Запуск ngrok туннеля...
echo ------------------------------------------------------------
echo.
echo Пароль для входа с телефона: trader2026
echo (сменить: .env - APP_PASSWORD=новый_пароль)
echo.
ngrok http 8000
echo.
echo Туннель отключён. Бот работает до закрытия окна.
pause
