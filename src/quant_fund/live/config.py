"""Live-gateway configuration. The venue in this file is the only venue that can arm."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from quant_fund.pretrade.config import PretradeConfig, SessionConfig, load_pretrade_config

Venue = Literal["loopback", "alpaca-paper", "alpaca-live"]
CalendarName = Literal["us_rth", "continuous"]


class LiveConfig(BaseModel):
    """Schema version 1. Extra keys are rejected."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    venue: Venue
    starting_cash: float = Field(gt=0.0, allow_inf_nan=False)
    calendar: CalendarName = "us_rth"
    pretrade_config: str
    state_dir: str = "data/live/default"
    cash_tolerance: float = Field(default=1e-6, ge=0.0, allow_inf_nan=False)
    position_tolerance: float = Field(default=1e-6, ge=0.0, allow_inf_nan=False)


def load_live_config(path: str | Path) -> LiveConfig:
    """Load YAML and reject unknown keys, versions, and non-finite limits."""
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("live config must be a YAML mapping")
    return LiveConfig.model_validate(raw)


def resolve_pretrade_path(config: LiveConfig, config_path: str | Path) -> Path:
    """Resolve the pre-trade file from the working directory, then beside the live file."""
    raw = Path(config.pretrade_config)
    if raw.is_absolute():
        return raw
    from_cwd = Path.cwd() / raw
    if from_cwd.exists():
        return from_cwd
    return Path(config_path).resolve().parent / raw


def load_gateway_pretrade(
    config: LiveConfig,
    config_path: str | Path,
    *,
    hmac_key: bytes,
) -> tuple[PretradeConfig, str, str]:
    """Load pre-trade limits and apply the gateway calendar.

    ``us_rth`` keeps the pre-trade file's session. ``continuous`` is a UTC
    day that stays open on weekends, for venues the operator marked that way.
    The returned digest is the signed payload the arm file must match.
    """
    path = resolve_pretrade_path(config, config_path)
    loaded, _, _ = load_pretrade_config(path, hmac_key=hmac_key)
    if config.calendar == "continuous":
        loaded = loaded.model_copy(
            update={
                "session": SessionConfig(
                    timezone="UTC",
                    open_minute=0,
                    close_minute=24 * 60,
                    weekends_open=True,
                )
            }
        )
    from quant_fund.pretrade.config import sign_config

    digest, signature = sign_config(loaded, hmac_key)
    return loaded, digest, signature
