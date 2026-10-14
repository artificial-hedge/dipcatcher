"""Local authorization file for operator order routing.

The file is not a broker credential and not an institutional promotion
receipt. It is the operator's explicit permission for this checkout to
route orders for one named account, inside hard limits, until it expires.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_fund.live.errors import LiveRefused

ACK_PHRASE = "I authorize dipcatcher to route orders for this account"
LIVE_CONFIRM_ENV = "DIPCATCHER_LIVE_CONFIRM"
LIVE_CONFIRM_VALUE = "ROUTE_LIVE_ORDERS"
BROKER_KEY_ENV = "DIPCATCHER_BROKER_KEY"
BROKER_SECRET_ENV = "DIPCATCHER_BROKER_SECRET"
HMAC_ENV = "DIPCATCHER_PRETRADE_HMAC_KEY"

_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,14}$")
_ACCOUNT = re.compile(r"^[A-Za-z0-9-]{1,64}$")
_HEX = re.compile(r"^[0-9a-fA-F]+$")
_MAX_LIFE = timedelta(days=90)
_FUTURE_SKEW = timedelta(minutes=5)


class BrokerCredentials:
    """Key material kept off the authorization file and out of audit records."""

    __slots__ = ("key_id", "secret")

    def __init__(self, key_id: str, secret: str) -> None:
        self.key_id = key_id
        self.secret = secret


class LiveAuthorization(BaseModel):
    """Schema ``live_authorization.v1``. Extra keys are rejected."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_name: str = Field(alias="schema")
    environment: str
    broker: str
    account_id: str
    operator: str
    acknowledge: str
    symbol_allowlist: list[str]
    max_order_notional: float = Field(gt=0.0, allow_inf_nan=False)
    max_orders_per_day: int = Field(ge=1, le=500)
    created_at: datetime
    expires_at: datetime

    @field_validator("schema_name")
    @classmethod
    def _schema(cls, value: str) -> str:
        if value != "live_authorization.v1":
            raise ValueError("schema must be live_authorization.v1")
        return value

    @field_validator("environment")
    @classmethod
    def _environment(cls, value: str) -> str:
        if value not in {"paper", "live"}:
            raise ValueError("environment must be paper or live")
        return value

    @field_validator("broker")
    @classmethod
    def _broker(cls, value: str) -> str:
        if value != "alpaca":
            raise ValueError("broker must be alpaca")
        return value

    @field_validator("account_id")
    @classmethod
    def _account(cls, value: str) -> str:
        if not _ACCOUNT.fullmatch(value):
            raise ValueError("account_id must be 1-64 letters, digits, or hyphens")
        return value

    @field_validator("operator")
    @classmethod
    def _operator(cls, value: str) -> str:
        text = value.strip()
        if not text or len(text) > 80 or any(ch in text for ch in "\r\n\t"):
            raise ValueError("operator must be 1-80 characters without line breaks")
        return text

    @field_validator("acknowledge")
    @classmethod
    def _acknowledge(cls, value: str) -> str:
        if value != ACK_PHRASE:
            raise ValueError("acknowledge phrase does not match")
        return value

    @field_validator("created_at", "expires_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _window_and_symbols(self) -> LiveAuthorization:
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        if self.expires_at - self.created_at > _MAX_LIFE:
            raise ValueError("authorization lifetime cannot exceed 90 days")
        symbols = [item.strip().upper() for item in self.symbol_allowlist]
        if not symbols:
            raise ValueError("symbol_allowlist is empty")
        if len(symbols) > 500:
            raise ValueError("symbol_allowlist exceeds 500 names")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbol_allowlist has duplicates")
        for symbol in symbols:
            if not _SYMBOL.fullmatch(symbol):
                raise ValueError(f"symbol {symbol!r} is not an allowed ticker")
        self.symbol_allowlist = symbols
        return self


def load_authorization(path: str | Path) -> LiveAuthorization:
    """Load and schema-check an authorization file. Does not check the clock."""
    text = Path(path).read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise LiveRefused("authorization file must be a JSON object")
    try:
        return LiveAuthorization.model_validate(raw)
    except ValueError as exc:
        raise LiveRefused(str(exc)) from exc


def authorization_sha256(path: str | Path) -> str:
    """Hash the file bytes so arming is bound to the exact authorization on disk."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest


def assert_current(auth: LiveAuthorization, *, now: datetime) -> None:
    """Refuse an authorization that is not yet valid or has expired."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise LiveRefused("clock must be timezone-aware")
    if auth.created_at - now > _FUTURE_SKEW:
        raise LiveRefused("authorization created_at is in the future")
    if auth.expires_at <= now:
        raise LiveRefused("authorization has expired")


def hmac_key_from_env(environ: Mapping[str, str] | None = None) -> bytes:
    """Read the pre-trade HMAC key. Hex only, at least 16 bytes, never logged."""
    env = os.environ if environ is None else environ
    raw = env.get(HMAC_ENV, "").strip()
    if raw.startswith(("0x", "0X")):
        raw = raw[2:]
    if len(raw) < 32 or len(raw) % 2 or _HEX.fullmatch(raw) is None:
        raise LiveRefused(f"{HMAC_ENV} must be at least 16 bytes of hex")
    return bytes.fromhex(raw)


def credentials_from_env(environ: Mapping[str, str] | None = None) -> BrokerCredentials:
    """Read broker key material. Missing values fail closed."""
    env = os.environ if environ is None else environ
    key_id = env.get(BROKER_KEY_ENV, "").strip()
    secret = env.get(BROKER_SECRET_ENV, "").strip()
    if not key_id or not secret:
        raise LiveRefused(f"{BROKER_KEY_ENV} and {BROKER_SECRET_ENV} are required")
    if len(key_id) < 8 or len(secret) < 8 or len(key_id) > 256 or len(secret) > 256:
        raise LiveRefused("broker credentials must be 8 to 256 characters")
    return BrokerCredentials(key_id, secret)


def assert_live_confirmed(environment: str, environ: Mapping[str, str] | None = None) -> None:
    """Paper needs no extra switch. Live needs an explicit process environment value."""
    if environment == "paper":
        return
    if environment != "live":
        raise LiveRefused("environment must be paper or live")
    env = os.environ if environ is None else environ
    if env.get(LIVE_CONFIRM_ENV, "").strip() != LIVE_CONFIRM_VALUE:
        raise LiveRefused(f"live routing requires {LIVE_CONFIRM_ENV}={LIVE_CONFIRM_VALUE}")


def credential_flags(environ: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Presence only. Values are not returned."""
    env = os.environ if environ is None else environ
    return {
        "broker_key_present": bool(env.get(BROKER_KEY_ENV, "").strip()),
        "broker_secret_present": bool(env.get(BROKER_SECRET_ENV, "").strip()),
        "pretrade_hmac_present": bool(env.get(HMAC_ENV, "").strip()),
        "live_confirm_present": env.get(LIVE_CONFIRM_ENV, "").strip() == LIVE_CONFIRM_VALUE,
    }
