"""Unit-тесты для WalletService (кошельки биржи, live-режим).

Проверяем:
1. Шифрование/дешифрование секрета (Fernet) — обратимость.
2. CRUD кошельков (create/list/get/set_default/update/delete) без валидации сети.
3. Маскировка api_key для UI.
4. Автоназначение первого кошелька дефолтным.
5. Снятие is_default с остальных при назначении нового дефолтного.
"""

from __future__ import annotations

import pytest
from sqlalchemy import delete, select, text

from core import wallet_service as ws
from core.wallet_service import WalletValidationError
from db.database import engine, get_session, init_db
from db.models import Wallet


# ─── Фикстуры ───────────────────────────────────────────────


@pytest.fixture
async def clean_db():
    """Чистая БД (только таблица wallet) для каждого теста."""
    await init_db()
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM wallet"))
    yield


@pytest.fixture(autouse=True)
def _reset_fernet_cache():
    """Сбрасывать кеш Fernet перед каждым тестом (на случай смены ключа)."""
    ws._fernet = None
    ws._warned_missing_key = False
    yield
    ws._fernet = None


# ─── Шифрование ─────────────────────────────────────────────


def test_encrypt_decrypt_roundtrip():
    """Секрет шифруется и расшифровывается обратно."""
    secret = "my_super_secret_api_key_12345"
    enc = ws._encrypt_secret(secret)
    assert enc != secret, "секрет не должен совпадать с зашифрованным"
    assert ws._decrypt_secret(enc) == secret


def test_encrypt_empty_secret():
    """Пустой секрет шифруется в пустую строку."""
    assert ws._encrypt_secret("") == ""
    assert ws._decrypt_secret("") == ""


def test_encrypt_produces_different_ciphertext():
    """Один и тот же секрет даёт разный шифротекст (Fernet adds IV)."""
    s = "secret123"
    e1 = ws._encrypt_secret(s)
    e2 = ws._encrypt_secret(s)
    assert e1 != e2, "Fernet должен давать разный шифротекст из-за IV"
    # Но оба расшифровываются в один секрет.
    assert ws._decrypt_secret(e1) == s
    assert ws._decrypt_secret(e2) == s


def test_mask_key():
    """Маскировка api_key скрывает середину ключа."""
    assert ws._mask_key("ABCDEFGHIJ1234567890") == "ABCD…7890"
    assert ws._mask_key("SHORT") == "*****"
    assert ws._mask_key("") == ""


# ─── CRUD ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_wallet_encrypts_secret(clean_db):
    """Создание кошелька шифрует секрет (в БД нет открытого секрета)."""
    async with get_session() as db:
        wallet = await ws.create_wallet(
            db, label="Binance main", exchange="binance",
            api_key="key_abc123", api_secret="secret_xyz789",
            validate=False,
        )
        assert wallet.id is not None
        assert wallet.api_key == "key_abc123"
        # Секрет НЕ хранится в открытом виде.
        assert wallet.api_secret_enc != "secret_xyz789"
        assert "secret_xyz789" not in wallet.api_secret_enc
        # Но расшифровывается.
        assert ws.get_decrypted_secret(wallet) == "secret_xyz789"
        # Первый кошелёк становится дефолтным автоматически.
        assert wallet.is_default is True


@pytest.mark.asyncio
async def test_create_wallet_requires_credentials(clean_db):
    """Без api_key/api_secret — ошибка валидации."""
    async with get_session() as db:
        with pytest.raises(WalletValidationError):
            await ws.create_wallet(db, "x", "binance", "", "secret", validate=False)
        with pytest.raises(WalletValidationError):
            await ws.create_wallet(db, "x", "binance", "key", "", validate=False)


@pytest.mark.asyncio
async def test_list_and_get_wallet(clean_db):
    """list_wallets возвращает все кошельки, get_wallet — по id."""
    async with get_session() as db:
        w1 = await ws.create_wallet(db, "W1", "binance", "k1", "s1", validate=False)
        w2 = await ws.create_wallet(
            db, "W2", "binance", "k2", "s2", is_default=False, validate=False,
        )
        wallets = await ws.list_wallets(db)
        assert len(wallets) == 2
        # Дефолтный — первым.
        assert wallets[0].is_default is True
        # get по id.
        fetched = await ws.get_wallet(db, w2.id)
        assert fetched is not None
        assert fetched.label == "W2"
        assert await ws.get_wallet(db, 999999) is None


@pytest.mark.asyncio
async def test_set_default_unsets_others(clean_db):
    """Назначение нового дефолтного снимает is_default с остальных."""
    async with get_session() as db:
        w1 = await ws.create_wallet(db, "W1", "binance", "k1", "s1", validate=False)
        w2 = await ws.create_wallet(
            db, "W2", "binance", "k2", "s2", is_default=False, validate=False,
        )
        assert w1.is_default is True
        # Делаем w2 дефолтным.
        ok = await ws.set_default(db, w2.id)
        assert ok is True
        # Перечитываем.
        w1_db = await ws.get_wallet(db, w1.id)
        w2_db = await ws.get_wallet(db, w2.id)
        assert w1_db.is_default is False
        assert w2_db.is_default is True
        # get_default_wallet возвращает w2.
        default = await ws.get_default_wallet(db)
        assert default.id == w2.id


@pytest.mark.asyncio
async def test_set_default_nonexistent(clean_db):
    """set_default для несуществующего id → False."""
    async with get_session() as db:
        assert await ws.set_default(db, 999999) is False


@pytest.mark.asyncio
async def test_update_wallet_label_and_keys(clean_db):
    """Обновление названия и ключей: ключи перешифровываются."""
    async with get_session() as db:
        wallet = await ws.create_wallet(
            db, "Old", "binance", "old_key", "old_secret", validate=False,
        )
        updated = await ws.update_wallet(
            db, wallet.id, label="New label",
            api_key="new_key", api_secret="new_secret", validate=False,
        )
        assert updated.label == "New label"
        assert updated.api_key == "new_key"
        assert ws.get_decrypted_secret(updated) == "new_secret"
        # Старый секрет больше не дешифруется новым значением.
        assert updated.api_secret_enc != "old_secret"


@pytest.mark.asyncio
async def test_delete_wallet_promotes_next_default(clean_db):
    """Удаление дефолтного кошелька назначает следующий активным."""
    async with get_session() as db:
        w1 = await ws.create_wallet(db, "W1", "binance", "k1", "s1", validate=False)
        w2 = await ws.create_wallet(
            db, "W2", "binance", "k2", "s2", is_default=False, validate=False,
        )
        w3 = await ws.create_wallet(
            db, "W3", "binance", "k3", "s3", is_default=False, validate=False,
        )
        # Удаляем дефолтный w1.
        ok = await ws.delete_wallet(db, w1.id)
        assert ok is True
        assert await ws.get_wallet(db, w1.id) is None
        # Какой-то из оставшихся стал дефолтным.
        default = await ws.get_default_wallet(db)
        assert default is not None
        assert default.id in (w2.id, w3.id)


@pytest.mark.asyncio
async def test_delete_nonexistent(clean_db):
    """Удаление несуществующего → False."""
    async with get_session() as db:
        assert await ws.delete_wallet(db, 999999) is False


@pytest.mark.asyncio
async def test_delete_last_wallet_leaves_no_default(clean_db):
    """Удаление последнего кошелька → нет дефолтного."""
    async with get_session() as db:
        w = await ws.create_wallet(db, "Only", "binance", "k", "s", validate=False)
        await ws.delete_wallet(db, w.id)
        assert await ws.get_default_wallet(db) is None
        assert await ws.list_wallets(db) == []


@pytest.mark.asyncio
async def test_wallet_info_masks_secret(clean_db):
    """WalletInfo не содержит открытый секрет, api_key маскируется."""
    async with get_session() as db:
        await ws.create_wallet(
            db, "W1", "binance", "ABCDEFGH1234567890", "secret",
            validate=False,
        )
        infos = await ws.list_wallets_public()
        assert len(infos) == 1
        info = infos[0]
        assert info.api_key_masked == "ABCD…7890"
        # В WalletInfo нет поля с открытым секретом.
        assert not hasattr(info, "api_secret")
        assert not hasattr(info, "api_secret_enc")
