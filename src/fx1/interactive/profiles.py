"""Model-endpoint store for ``fxi``.

fx-1 ships two hosted model names; each profile here is one model's
endpoint — the API key and base URL used to talk to it:

- ``fx1``      — the full fx-1 model.
- ``fx1-lite`` — the light fx-1 model.

Values live in ``$FX1_CONFIG_DIR/credentials.json`` (default
``~/.fx1/credentials.json``), mode 0600, written atomically. Reporting is
presence-only: raw keys never leave this module except as injected
environment values (mirrors ``fx1.doctor`` and the harness conventions).

Injection: applying a profile sets ``MOONSHOT_API_KEY`` and
``FX1_BASE_URL`` for child processes and in-process backends.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from fx1.serve.backends import MOONSHOT_API_URL

MODELS: tuple[str, ...] = ("fx1", "fx1-lite")
DEFAULT_MODEL = "fx1"

ENV_API_KEY = "MOONSHOT_API_KEY"
ENV_BASE_URL = "FX1_BASE_URL"

STORE_NAME = "credentials.json"
_MODE_0600 = stat.S_IRUSR | stat.S_IWUSR


def config_dir() -> Path:
    override = os.environ.get("FX1_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".fx1"


def store_path() -> Path:
    return config_dir() / STORE_NAME


def default_base_url() -> str:
    return os.environ.get(ENV_BASE_URL, MOONSHOT_API_URL)


def fingerprint(api_key: str) -> str:
    """Presence-only fingerprint: the last four characters, never the value."""
    tail = api_key[-4:] if len(api_key) >= 4 else api_key
    return f"...{tail}"


def check_model(model: str) -> str:
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}; choose from {sorted(MODELS)}")
    return model


def host_of(base_url: str) -> str:
    return urlparse(base_url).hostname or base_url


def _normalize(entry: Mapping[str, str]) -> dict[str, str]:
    """Read-compat: accept the pre-base-URL schema (``key``/``env_var``)."""
    api_key = entry.get("api_key") or entry.get("key") or ""
    base_url = entry.get("base_url") or default_base_url()
    return {"api_key": api_key, "base_url": base_url, "set_at": entry.get("set_at", "")}


def _load(path: Path | None = None) -> dict[str, dict[str, str]]:
    path = path or store_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    profiles = raw.get("profiles")
    if isinstance(profiles, dict):
        return cast("dict[str, dict[str, str]]", profiles)
    return {}


def _save(payload: dict[str, dict[str, str]], path: Path | None = None) -> Path:
    path = path or store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".credentials-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"profiles": payload}, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    os.chmod(path, _MODE_0600)
    return path


def set_endpoint(
    model: str,
    api_key: str,
    base_url: str | None = None,
    *,
    path: Path | None = None,
) -> Path:
    check_model(model)
    api_key = api_key.strip()
    if not api_key:
        raise ValueError(f"empty API key for model {model!r}")
    base = (base_url or "").strip() or default_base_url()
    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid base URL {base!r}; expected http(s)://host/...")
    payload = _load(path)
    payload[model] = {
        "api_key": api_key,
        "base_url": base,
        "set_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return _save(payload, path)


def remove_endpoint(model: str, *, path: Path | None = None) -> bool:
    check_model(model)
    payload = _load(path)
    if model not in payload:
        return False
    del payload[model]
    _save(payload, path)
    return True


def list_models(
    *, path: Path | None = None, environ: Mapping[str, str] | None = None
) -> list[dict[str, str]]:
    """Presence-only rows: model, host, status, fingerprint, set_at.

    ``status`` is ``set`` (stored by fxi), ``env`` (only in the environment),
    or ``unset``.
    """
    env = os.environ if environ is None else environ
    payload = _load(path)
    rows: list[dict[str, str]] = []
    for model in MODELS:
        normalized = _normalize(payload[model]) if model in payload else None
        api_key = normalized["api_key"] if normalized else ""
        base_url = normalized["base_url"] if normalized else default_base_url()
        rows.append(
            {
                "model": model,
                "host": host_of(base_url),
                "status": "set" if api_key else ("env" if env.get(ENV_API_KEY) else "unset"),
                "fingerprint": fingerprint(api_key) if api_key else "-",
                "set_at": normalized["set_at"] if (normalized and api_key) else "-",
            }
        )
    return rows


def resolve_endpoint(
    model: str, *, path: Path | None = None, environ: Mapping[str, str] | None = None
) -> tuple[str, str] | None:
    """``(api_key, base_url)`` for *model* — stored wins, else the env."""
    check_model(model)
    env = os.environ if environ is None else environ
    entry = _load(path).get(model)
    if entry:
        normalized = _normalize(entry)
        if normalized["api_key"]:
            return normalized["api_key"], normalized["base_url"]
    api_key = env.get(ENV_API_KEY)
    if api_key:
        return api_key, env.get(ENV_BASE_URL, MOONSHOT_API_URL)
    return None


def apply_profile(
    model: str, *, path: Path | None = None, environ: MutableMapping[str, str] | None = None
) -> list[str]:
    """Inject *model*'s endpoint into *environ* (default os.environ).

    Returns the env var names set. Fail-closed: raises ``RuntimeError``
    naming the model when no key resolves — mirroring ``fx1.serve.backends``.
    """
    resolved = resolve_endpoint(model, path=path)
    if not resolved:
        raise RuntimeError(
            f"no endpoint for model {model!r}; run `dipcatcher` or `fxi setup` "
            f"to enter your API key and base URL, or export {ENV_API_KEY}"
        )
    api_key, base_url = resolved
    target = os.environ if environ is None else environ
    target[ENV_API_KEY] = api_key
    target[ENV_BASE_URL] = base_url
    return [ENV_API_KEY, ENV_BASE_URL]
