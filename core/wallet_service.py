"""WalletService — управление кошельками биржи для live-режима.

Кошелёк = привязка API-ключей биржи (api_key + api_secret). Используется в
live-режиме для торговли реальными деньгами через CCXT.

Безопасность:
    - api_secret хранится в БД в зашифрованном виде (Fernet, симметричное
      шифрование). В открытом виде секрет не покидает WalletService.
    - Ключ шифрования берётся из settings.wallet_encryption_key. Если пуст —
      выводится предупреждение и используется производный ключ от app_password
      (только для прототипа; в продакшене задайте явный WALLET_ENCRYPTION_KEY).

API:
    - create_wallet(...)    — создать кошелёк (с опциональной валидацией ключей).
    - get_default_wallet()  — кошелёк по умолчанию (is_default=True).
    - list_wallets()        — все кошельки (без секретов в открытом виде).
    - get_wallet(id)        — конкретный кошелёк (с расшифровкой по запросу).
    - update_wallet(...)    — обновить данные кошелька.
    - set_default(id)       — сделать кошелёк активным.
    - delete_wallet(id)     — удалить кошелёк.
    - get_decrypted_secret(wallet) — расшифрованный api_secret (для MarketService).
"""

from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.database import get_session
from db.models import Wallet

logger = logging.getLogger("trading.wallet")


# ─── Шифрование секрета (Fernet) ──────────────────────────────


_fernet: "Fernet | None" = None  # type: ignore[name-defined]
_warned_missing_key = False


def _get_fernet():
    """Лениво создать Fernet с ключом из настроек.

    Если WALLET_ENCRYPTION_KEY не задан — выводим предупреждение и используем
    производный ключ от app_password (только для прототипа).
    """
    global _fernet, _warned_missing_key
    if _fernet is not None:
        return _fernet

    from cryptography.fernet import Fernet

    raw_key = settings.wallet_encryption_key.strip()
    if not raw_key:
        if not _warned_missing_key:
            logger.warning(
                "WALLET_ENCRYPTION_KEY не задан — секреты кошельков шифруются "
                "производным ключом от app_password. Для продакшена задайте "
                "явный WALLET_ENCRYPTION_KEY."
            )
            _warned_missing_key = True
        # Производный ключ: 32 байта SHA-256 → urlsafe base64 (формат Fernet).
        seed = (settings.app_password or "kronos-trading-dev-key").encode()
        raw_key = base64.urlsafe_b64encode(hashlib.sha256(seed).digest())
    elif isinstance(raw_key, str):
        raw_key = raw_key.encode()

    # Fernet принимает 32-byte urlsafe base64 ключ. Если строка уже в этом
    # формате (сгенерирована Fernet.generate_key) — оставляем как есть.
    try:
        _fernet = Fernet(raw_key)
    except Exception:
        # Не валидный Fernet-токен → хешируем до 32 байт → urlsafe base64.
        derived = hashlib.sha256(raw_key).digest()
        _fernet = Fernet(base64.urlsafe_b64encode(derived))
    return _fernet


def _encrypt_secret(secret: str) -> str:
    """Зашифровать api_secret → строка-токен Fernet."""
    if not secret:
        return ""
    f = _get_fernet()
    return f.encrypt(secret.encode()).decode()


def _decrypt_secret(enc: str) -> str:
    """Расшифровать api_secret из токена Fernet."""
    if not enc:
        return ""
    f = _get_fernet()
    return f.decrypt(enc.encode()).decode()


def get_decrypted_secret(wallet: Wallet) -> str:
    """Расшифрованный api_secret кошелька (для передачи в CCXT)."""
    return _decrypt_secret(wallet.api_secret_enc)


# ─── Результат операций ───────────────────────────────────────


@dataclass
class WalletValidationError(Exception):
    """Невалидные API-ключи (биржа отвергла запрос баланса)."""
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class WalletInfo:
    """Безопасное представление кошелька для UI (без секрета)."""
    id: int
    label: str
    exchange: str
    api_key: str
    api_key_masked: str
    is_default: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(cls, w: Wallet) -> "WalletInfo":
        return cls(
            id=w.id,
            label=w.label,
            exchange=w.exchange,
            api_key=w.api_key,
            api_key_masked=_mask_key(w.api_key),
            is_default=w.is_default,
            created_at=w.created_at,
            updated_at=w.updated_at,
        )


def _mask_key(api_key: str) -> str:
    """Маскированный api_key для UI: первые 4 и последние 4 символа."""
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}…{api_key[-4:]}"


# ─── Валидация ключей через биржу ─────────────────────────────


async def _validate_keys(exchange: str, api_key: str, api_secret: str) -> float:
    """Проверить ключи реальным запросом баланса.

    Возвращает свободный баланс USDT. Бросает WalletValidationError, если
    ключи невалидны или биржа недоступна.
    """
    import ccxt.async_support as ccxt
    from core.http_client import make_aiohttp_session

    session = make_aiohttp_session()
    try:
        ex = getattr(ccxt, exchange)({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
            "session": session,
        })
        try:
            await ex.load_markets()
            balance = await ex.fetch_balance()
            usdt_free = float(balance.get("free", {}).get("USDT", 0.0) or 0.0)
            logger.info(
                "Ключи биржи валидны (%s): свободный баланс USDT=%.2f",
                exchange, usdt_free,
            )
            return usdt_free
        finally:
            await ex.close()
    except ccxt.AuthenticationError as e:
        raise WalletValidationError(f"Неверные API-ключи: {e}") from e
    except ccxt.NetworkError as e:
        raise WalletValidationError(f"Сеть недоступна при проверке ключей: {e}") from e
    except Exception as e:
        raise WalletValidationError(f"Не удалось проверить ключи: {e}") from e
    finally:
        if not session.closed:
            await session.close()


# ─── CRUD ─────────────────────────────────────────────────────


async def create_wallet(
    db: AsyncSession,
    label: str,
    exchange: str,
    api_key: str,
    api_secret: str,
    is_default: bool = True,
    validate: bool = True,
) -> Wallet:
    """Создать кошелёк.

    Args:
        validate: проверить ключи реальным запросом баланса (по умолчанию).
            Для тестов можно отключить.
    """
    label = (label or "").strip() or "Кошелёк"
    exchange = (exchange or "binance").strip() or "binance"
    api_key = (api_key or "").strip()
    api_secret = (api_secret or "").strip()
    if not api_key or not api_secret:
        raise WalletValidationError("api_key и api_secret обязательны")

    if validate:
        await _validate_keys(exchange, api_key, api_secret)

    enc = _encrypt_secret(api_secret)

    # Если это первый кошелёк — делаем его дефолтным.
    existing_count = await _count_wallets(db)

    wallet = Wallet(
        label=label,
        exchange=exchange,
        api_key=api_key,
        api_secret_enc=enc,
        is_default=is_default or existing_count == 0,
    )
    db.add(wallet)
    await db.flush()

    if wallet.is_default:
        await _set_default_internal(db, wallet.id)
    await db.commit()
    await db.refresh(wallet)
    logger.info("Создан кошелёк #%s '%s' (%s)", wallet.id, label, exchange)
    return wallet


async def list_wallets(db: AsyncSession) -> list[Wallet]:
    """Все кошельки (is_default первым)."""
    stmt = select(Wallet).order_by(Wallet.is_default.desc(), Wallet.created_at.asc())
    return list((await db.execute(stmt)).scalars().all())


async def get_wallet(db: AsyncSession, wallet_id: int) -> Wallet | None:
    return (await db.execute(
        select(Wallet).where(Wallet.id == wallet_id)
    )).scalar_one_or_none()


async def get_default_wallet(db: AsyncSession) -> Wallet | None:
    """Кошелёк по умолчанию для live-режима."""
    stmt = select(Wallet).where(Wallet.is_default == True).limit(1)
    w = (await db.execute(stmt)).scalars().first()
    if w is not None:
        return w
    # Нет дефолтного — берём первый попавшийся.
    return (await db.execute(select(Wallet).limit(1))).scalars().first()


async def update_wallet(
    db: AsyncSession,
    wallet_id: int,
    label: str | None = None,
    api_key: str | None = None,
    api_secret: str | None = None,
    validate: bool = True,
) -> Wallet | None:
    """Обновить данные кошелька. Переданные поля (не None) перезаписываются."""
    wallet = await get_wallet(db, wallet_id)
    if wallet is None:
        return None

    new_key = (api_key or "").strip() if api_key is not None else None
    new_secret = (api_secret or "").strip() if api_secret is not None else None

    # Если менялись ключи — валидируем и перешифровываем.
    if new_key is not None or new_secret is not None:
        eff_key = new_key if new_key else wallet.api_key
        eff_secret = new_secret if new_secret else get_decrypted_secret(wallet)
        if not eff_key or not eff_secret:
            raise WalletValidationError("api_key и api_secret обязательны")
        if validate:
            await _validate_keys(wallet.exchange, eff_key, eff_secret)
        wallet.api_key = eff_key
        wallet.api_secret_enc = _encrypt_secret(eff_secret)

    if label is not None and label.strip():
        wallet.label = label.strip()
    wallet.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(wallet)
    logger.info("Обновлён кошелёк #%s", wallet_id)
    return wallet


async def set_default(db: AsyncSession, wallet_id: int) -> bool:
    """Сделать кошелёк активным (is_default=True), сняв флаг с остальных."""
    wallet = await get_wallet(db, wallet_id)
    if wallet is None:
        return False
    await _set_default_internal(db, wallet_id)
    await db.commit()
    logger.info("Кошелёк #%s назначен активным", wallet_id)
    return True


async def _set_default_internal(db: AsyncSession, wallet_id: int) -> None:
    """Снять is_default со всех, поставить только wallet_id (без commit)."""
    await db.execute(update(Wallet).values(is_default=False))
    await db.execute(
        update(Wallet).where(Wallet.id == wallet_id).values(is_default=True)
    )


async def delete_wallet(db: AsyncSession, wallet_id: int) -> bool:
    """Удалить кошелёк. Если удалили дефолтный — назначаем следующий."""
    wallet = await get_wallet(db, wallet_id)
    if wallet is None:
        return False
    was_default = wallet.is_default
    await db.execute(
        Wallet.__table__.delete().where(Wallet.id == wallet_id)  # type: ignore[attr-defined]
    )
    if was_default:
        # Назначаем дефолтным самый старый оставшийся.
        next_w = (await db.execute(
            select(Wallet).order_by(Wallet.created_at.asc()).limit(1)
        )).scalars().first()
        if next_w is not None:
            await _set_default_internal(db, next_w.id)
    await db.commit()
    logger.info("Удалён кошелёк #%s", wallet_id)
    return True


async def _count_wallets(db: AsyncSession) -> int:
    from sqlalchemy import func
    return int((await db.execute(select(func.count(Wallet.id)))).scalar() or 0)


# ─── Удобные обёртки с собственной сессией ────────────────────


async def list_wallets_public() -> list[WalletInfo]:
    """Все кошельки как WalletInfo (без секретов) — для UI."""
    async with get_session() as db:
        wallets = await list_wallets(db)
        return [WalletInfo.from_model(w) for w in wallets]
