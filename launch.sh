#!/usr/bin/env bash
# ============================================================
#  Kronos Auto-Trading — запуск приложения (Bash / Git Bash)
#  Запуск:  ./launch.sh
# ============================================================
set -e
cd "$(dirname "$0")"

echo ""
echo "============================================================"
echo "  Kronos Auto-Trading"
echo "============================================================"
echo ""

# Проверяем Python.
if ! command -v python &>/dev/null; then
    echo "[ОШИБКА] Python не найден в PATH."
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
if ! python -c "import fastapi" &>/dev/null; then
    echo "[INFO] Устанавливаю зависимости..."
    python -m pip install -r requirements.txt
fi

echo ""
echo "  Открой в браузере:  http://127.0.0.1:8000"
echo "  Остановить:        Ctrl+C"
echo ""
echo "------------------------------------------------------------"
echo ""

exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
