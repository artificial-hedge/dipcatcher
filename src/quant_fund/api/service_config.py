"""One startup contract for the lab API and the read-only research API.

Operators set the environment variables in ``ENV_CONTRACT``. ``dipcatcher api``
flags ``--host`` and ``--port`` override ``DIPCATCHER_BIND_HOST`` and
``DIPCATCHER_BIND_PORT``. The process does not bind until ``for_lab`` or
``for_research`` returns.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from quant_fund.api.kill_file import validate_kill_switch_path

ENV_CONTRACT = (
    "DIPCATCHER_ALERT_INTERVAL_S",
    "DIPCATCHER_BIND_HOST",
    "DIPCATCHER_BIND_PORT",
    "DIPCATCHER_CONFIG",
    "DIPCATCHER_DATA_ROOT",
    "DIPCATCHER_HALT_CANCELS_OPEN_ORDERS",
    "DIPCATCHER_KILL_DRAIN_DEBOUNCE_S",
    "DIPCATCHER_KILL_SWITCH_PATH",
    "DIPCATCHER_LOG_LEVEL",
    "DIPCATCHER_METRICS_HOST",
    "DIPCATCHER_METRICS_LOOPBACK_SCRAPE",
    "DIPCATCHER_METRICS_PORT",
    "DIPCATCHER_OBSERVE",
    "DIPCATCHER_OTEL_ENDPOINT",
    "DIPCATCHER_RATE_LIMIT_RPS",
    "DIPCATCHER_REQUIRE_DATA_WRITE",
    "DIPCATCHER_SHUTDOWN_GRACE_S",
    "DIPCATCHER_SLO_PATH",
    "PORT",
    "QUANT_API_KEY",
    "RESEARCH_API_HOST",
    "RESEARCH_API_KEY",
    "RESEARCH_API_PORT",
)

_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})
_LAB_KEY = "non-loopback API binding requires QUANT_API_KEY; refusing unauthenticated exposure"
_RESEARCH_KEY = "non-loopback binding requires RESEARCH_API_KEY; refusing unauthenticated exposure"


class ServiceConfigError(ValueError):
    """Startup validation failed. ``problems`` contain no secret values."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class ServiceSettings:
    service: str
    host: str
    port: int
    log_level: str
    config_path: str | None
    observe: bool
    metrics_host: str
    metrics_port: int | None
    otel_configured: bool
    shutdown_grace_s: int
    kill_switch_path: str | None
    slo_path: str | None
    alert_interval_s: int
    api_key_configured: bool
    metrics_loopback_scrape: bool = False

    def public_dict(self) -> dict[str, str | int | bool | None]:
        return {
            "alert_interval_s": self.alert_interval_s,
            "api_key_configured": self.api_key_configured,
            "config_path": self.config_path,
            "host": self.host,
            "kill_switch_path": self.kill_switch_path,
            "log_level": self.log_level,
            "metrics_host": self.metrics_host,
            "metrics_loopback_scrape": self.metrics_loopback_scrape,
            "metrics_port": self.metrics_port,
            "observe": self.observe,
            "otel_configured": self.otel_configured,
            "port": self.port,
            "service": self.service,
            "shutdown_grace_s": self.shutdown_grace_s,
            "slo_path": self.slo_path,
        }

    @classmethod
    def for_lab(cls, host: str | None, port: int | None) -> ServiceSettings:
        problems: list[str] = []
        chosen_host = host if host is not None else _env("DIPCATCHER_BIND_HOST") or "127.0.0.1"
        chosen_port, port_problem = _resolve_port(port, "DIPCATCHER_BIND_PORT", "--port", 8000)
        if port_problem:
            problems.append(port_problem)
        settings, more = _assemble(
            "lab",
            chosen_host,
            chosen_port,
            "QUANT_API_KEY",
            _LAB_KEY,
            _lab_config_problem(),
        )
        problems.extend(more)
        if problems:
            raise ServiceConfigError(problems)
        return settings

    @classmethod
    def for_research(cls) -> ServiceSettings:
        problems: list[str] = []
        chosen_host = _env("RESEARCH_API_HOST") or "127.0.0.1"
        chosen_port, port_problem = _resolve_port(
            None, "RESEARCH_API_PORT", "RESEARCH_API_PORT", 8010
        )
        if port_problem:
            problems.append(port_problem)
        settings, more = _assemble(
            "research",
            chosen_host,
            chosen_port,
            "RESEARCH_API_KEY",
            _RESEARCH_KEY,
            None,
        )
        problems.extend(more)
        if problems:
            raise ServiceConfigError(problems)
        return settings


def slo_candidates() -> tuple[Path, ...]:
    """Checkout path first, then the container path used by the runtime image."""
    module_root = Path(__file__).resolve().parents[3]
    return (
        module_root / "deploy" / "observability" / "slo.yml",
        Path("/app/deploy/observability/slo.yml"),
    )


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _key_configured(name: str) -> bool:
    """True when the env value or its ``*_FILE`` path is non-blank.

    Whitespace-only values stay unset. A file path counts so a non-loopback
    bind can use a file-backed key without putting the secret in the env.
    """
    if os.environ.get(name, "").strip():
        return True
    return bool(os.environ.get(f"{name}_FILE", "").strip())


def _resolve_port(
    explicit: int | None, env_name: str, label: str, default: int
) -> tuple[int, str | None]:
    if explicit is not None:
        if explicit < 1 or explicit > 65535:
            return default, f"{label} must be an integer between 1 and 65535"
        return explicit, None
    chosen, problem = _int_env(env_name, default=None, low=1, high=65535)
    if problem:
        return default, problem
    if chosen is not None:
        return chosen, None
    fallback, fallback_problem = _int_env("PORT", default=default, low=1, high=65535)
    if fallback_problem or fallback is None:
        return default, fallback_problem or "PORT must be an integer between 1 and 65535"
    return fallback, None


def _int_env(
    name: str, *, default: int | None, low: int, high: int
) -> tuple[int | None, str | None]:
    raw = _env(name)
    if not raw:
        return default, None
    if len(raw) > 8 or not raw.isdigit():
        return None, f"{name} must be an integer between {low} and {high}"
    value = int(raw)
    if value < low or value > high:
        return None, f"{name} must be an integer between {low} and {high}"
    return value, None


def _flag(name: str, *, default: bool) -> tuple[bool, str | None]:
    raw = _env(name).lower()
    if not raw:
        return default, None
    if raw in _TRUE:
        return True, None
    if raw in _FALSE:
        return False, None
    return default, f"{name} must be 0 or 1"


def _check_host(host: str) -> str | None:
    if not host or len(host) > 253 or any(char.isspace() for char in host) or "://" in host:
        return "bind host is invalid"
    return None


def _check_otel(raw: str) -> str | None:
    if not raw:
        return None
    if len(raw) > 512:
        return "DIPCATCHER_OTEL_ENDPOINT is invalid"
    parts = urlsplit(raw)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return "DIPCATCHER_OTEL_ENDPOINT must be an http(s) URL"
    if parts.username or parts.password:
        return "DIPCATCHER_OTEL_ENDPOINT must not embed credentials"
    return None


def _lab_config_problem() -> str | None:
    raw = _env("DIPCATCHER_CONFIG") or "configs/research.yaml"
    if len(raw) > 512:
        return "DIPCATCHER_CONFIG is invalid"
    import yaml
    from fastapi import HTTPException

    from quant_fund.api.app import resolve_allowed_config_path
    from quant_fund.config import load_config

    try:
        load_config(resolve_allowed_config_path(raw))
    except HTTPException:
        return "DIPCATCHER_CONFIG is not a readable config under configs/"
    except (OSError, ValueError, yaml.YAMLError):
        return "DIPCATCHER_CONFIG failed validation"
    return None


def _discover_slo(explicit: str, observe: bool) -> tuple[str | None, str | None]:
    if not observe:
        if explicit:
            return None, "DIPCATCHER_SLO_PATH requires DIPCATCHER_OBSERVE=1"
        return None, None
    if explicit:
        if len(explicit) > 512:
            return None, "DIPCATCHER_SLO_PATH is invalid"
        path = Path(explicit)
        return _load_slo(path, required=True)
    for candidate in slo_candidates():
        if candidate.is_file():
            return _load_slo(candidate, required=True)
    return None, None


def _load_slo(path: Path, *, required: bool) -> tuple[str | None, str | None]:
    if path.is_symlink() or not path.is_file():
        if required:
            return None, "DIPCATCHER_SLO_PATH is not a readable file"
        return None, None
    import yaml

    from quant_fund.observe.slo import load_slos

    try:
        load_slos(path)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        return None, "DIPCATCHER_SLO_PATH failed validation"
    return str(path), None


_PRODUCTION_PROFILES = frozenset({"production", "prod"})
_PRODUCTION_RATE = "DIPCATCHER_RATE_LIMIT_RPS must be set above 0 in production"
_INVALID_RATE = "DIPCATCHER_RATE_LIMIT_RPS is invalid"


def _production_profile() -> bool:
    """True when the deploy profile asks for production boot checks.

    ``DIPCATCHER_PROFILE`` stays out of ``ENV_CONTRACT`` so the example env
    file can keep the non-production default. ``deploy/prod.env`` sets it.
    """
    return _env("DIPCATCHER_PROFILE").lower() in _PRODUCTION_PROFILES


def _parsed_rate(raw: str) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0.0 or value > 1_000_000.0:
        return None
    return value


def _rate_limit_problems() -> list[str]:
    raw = _env("DIPCATCHER_RATE_LIMIT_RPS")
    if not raw:
        return []
    value = _parsed_rate(raw)
    if _production_profile() and (value is None or value <= 0.0):
        return [_PRODUCTION_RATE]
    if not _production_profile() and value is None:
        return [_INVALID_RATE]
    return []


def _bind_problems(host: str, key_name: str, key_message: str) -> list[str]:
    problems: list[str] = []
    host_problem = _check_host(host)
    if host_problem:
        problems.append(host_problem)
    if host not in _LOOPBACK and not _key_configured(key_name):
        problems.append(key_message)
    return problems


def _log_level() -> tuple[str, str | None]:
    level = _env("DIPCATCHER_LOG_LEVEL") or "INFO"
    if level not in _LEVELS:
        message = "DIPCATCHER_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL"
        return "INFO", message
    return level, None


def _required_int(name: str, default: int, low: int, high: int) -> tuple[int, str | None]:
    value, problem = _int_env(name, default=default, low=low, high=high)
    if problem or value is None:
        return default, problem or f"{name} is invalid"
    return value, None


def _metrics_flag_without_port() -> list[str]:
    if _env("DIPCATCHER_METRICS_LOOPBACK_SCRAPE"):
        return ["DIPCATCHER_METRICS_LOOPBACK_SCRAPE requires DIPCATCHER_METRICS_PORT"]
    return []


def _metrics_bind_auth(host: str, enabled: bool, key_name: str) -> list[str]:
    problems: list[str] = []
    loopback = host in _LOOPBACK
    keyed = _key_configured(key_name)
    if enabled and not loopback:
        problems.append(
            "DIPCATCHER_METRICS_LOOPBACK_SCRAPE requires a loopback DIPCATCHER_METRICS_HOST"
        )
    if not loopback and not keyed:
        problems.append(f"off-loopback /metrics requires {key_name}")
    if not enabled and not keyed:
        problems.append(
            f"DIPCATCHER_METRICS_PORT requires {key_name} or DIPCATCHER_METRICS_LOOPBACK_SCRAPE=1"
        )
    return problems


def _metrics_auth(host: str, port: int | None, key_name: str) -> tuple[bool, list[str]]:
    problems: list[str] = []
    enabled, flag_problem = _flag("DIPCATCHER_METRICS_LOOPBACK_SCRAPE", default=False)
    if flag_problem:
        problems.append(flag_problem)
    if port is None:
        return False, problems + _metrics_flag_without_port()
    problems.extend(_metrics_bind_auth(host, enabled, key_name))
    return bool(enabled and host in _LOOPBACK), problems


def _metrics(observe: bool, key_name: str) -> tuple[str, int | None, bool, list[str]]:
    problems: list[str] = []
    metrics_port, metrics_problem = _int_env(
        "DIPCATCHER_METRICS_PORT", default=None, low=1, high=65535
    )
    if metrics_problem:
        problems.append(metrics_problem)
    metrics_host = _env("DIPCATCHER_METRICS_HOST") or "127.0.0.1"
    if metrics_port is not None:
        if _check_host(metrics_host):
            problems.append("DIPCATCHER_METRICS_HOST is invalid")
        if not observe:
            problems.append("DIPCATCHER_METRICS_PORT requires DIPCATCHER_OBSERVE=1")
    scrape, auth_problems = _metrics_auth(metrics_host, metrics_port, key_name)
    problems.extend(auth_problems)
    return metrics_host, metrics_port, scrape, problems


def _otel(observe: bool) -> tuple[bool, list[str]]:
    problems: list[str] = []
    raw = _env("DIPCATCHER_OTEL_ENDPOINT")
    problem = _check_otel(raw)
    if problem:
        problems.append(problem)
    if raw and not observe:
        problems.append("DIPCATCHER_OTEL_ENDPOINT requires DIPCATCHER_OBSERVE=1")
    return bool(raw) and problem is None, problems


def _kill_path() -> tuple[str | None, str | None]:
    raw = _env("DIPCATCHER_KILL_SWITCH_PATH")
    if not raw:
        return None, None
    problem = validate_kill_switch_path(Path(raw))
    if problem:
        return None, problem
    return raw, None


def _secret_problems() -> list[str]:
    from quant_fund.api.secret_files import apply_secret_files

    return apply_secret_files()


def _assemble(
    service: str,
    host: str,
    port: int,
    key_name: str,
    key_message: str,
    config_problem: str | None,
) -> tuple[ServiceSettings, list[str]]:
    problems = _secret_problems()
    problems.extend(_bind_problems(host, key_name, key_message))
    level, level_problem = _log_level()
    if level_problem:
        problems.append(level_problem)
    observe, observe_problem = _flag("DIPCATCHER_OBSERVE", default=False)
    if observe_problem:
        problems.append(observe_problem)
    grace, grace_problem = _required_int("DIPCATCHER_SHUTDOWN_GRACE_S", 20, 0, 600)
    if grace_problem:
        problems.append(grace_problem)
    interval, interval_problem = _required_int("DIPCATCHER_ALERT_INTERVAL_S", 60, 1, 86400)
    if interval_problem:
        problems.append(interval_problem)
    metrics_host, metrics_port, metrics_loopback_scrape, metric_problems = _metrics(
        observe, key_name
    )
    problems.extend(metric_problems)
    otel_configured, otel_problems = _otel(observe)
    problems.extend(otel_problems)
    problems.extend(_rate_limit_problems())
    slo_path, slo_problem = _discover_slo(_env("DIPCATCHER_SLO_PATH"), observe)
    if slo_problem:
        problems.append(slo_problem)
    kill_path, kill_problem = _kill_path()
    if kill_problem:
        problems.append(kill_problem)
    if config_problem:
        problems.append(config_problem)
    if service == "research":
        config_path = None
    else:
        config_path = _env("DIPCATCHER_CONFIG") or "configs/research.yaml"
    settings = ServiceSettings(
        service=service,
        host=host,
        port=port,
        log_level=level,
        config_path=config_path,
        observe=observe,
        metrics_host=metrics_host,
        metrics_port=metrics_port,
        otel_configured=otel_configured,
        shutdown_grace_s=grace,
        kill_switch_path=kill_path,
        slo_path=slo_path,
        alert_interval_s=interval,
        api_key_configured=_key_configured(key_name),
        metrics_loopback_scrape=metrics_loopback_scrape,
    )
    return settings, problems
