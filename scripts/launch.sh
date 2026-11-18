#!/usr/bin/env bash
# ============================================================
#  Kronos Auto-Trading — запуск приложения (Bash / Git Bash)
#  Запуск:  ./scripts/launch.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."

echo ""
echo "============================================================"
echo "  Kronos Auto-Trading"
echo "============================================================"
echo ""

# Python: сначала venv проекта, иначе системный.
if [ -x "venv/bin/python" ]; then PY="venv/bin/python"
elif [ -x "venv/Scripts/python.exe" ]; then PY="venv/Scripts/python.exe"
elif command -v python &>/dev/null; then PY="python"
else
    echo "[ОШИБКА] Python не найден: нет venv и python в PATH."
    exit 1
fi

# Папка для БД.
mkdir -p data

# .env из примера.
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "[INFO] Создан .env из .env.example"
fi

# Зависимости (проверяем по fastapi).
if ! "$PY" -c "import fastapi" &>/dev/null; then
    echo "[INFO] Устанавливаю зависимости..."
    "$PY" -m pip install -r requirements.txt
fi

echo ""
echo "  Открой в браузере:  http://127.0.0.1:8000"
echo "  Остановить:        Ctrl+C"
echo ""
echo "------------------------------------------------------------"
echo ""

exec "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
