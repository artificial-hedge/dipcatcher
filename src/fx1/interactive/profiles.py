"""Model-endpoint store for dipcatcher and ``fxi``.

fx-1 publishes no weights. Each profile here is one chat id's endpoint —
the API key and base URL an operator configured:

- ``fx1``      — the fx1 chat id.
- ``fx1-lite`` — the fx1-lite chat id.

The store is ``$FX1_CONFIG_DIR/credentials.json`` (default
``~/.fx1/credentials.json``), schema version 2, mode 0600. The config
directory is tightened to ``0700`` when this process owns it. A symlink at
the store or its directory is refused and never followed. Writes go through
:func:`fx1.interactive.atomicstore.update_json` (locked, atomic). ``base_url``
is the normalized chat-completions URL. ``service_origin`` is the scheme and
host. Readers still accept version-1 entries (``key`` / ``env_var``, a bare
origin, a ``/v1`` base, or a full route) and normalize them at read time.

Session chat backends receive credentials explicitly. Lab harness
subprocesses inherit a scrubbed environment; optional
:func:`subprocess_env_for_model` adds credentials only to children that
require them.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import unicodedata
import urllib.parse
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import UTC, datetime
from getpass import getuser
from pathlib import Path
from typing import Any, Literal

from fx1.interactive.atomicstore import (
    FileLock,
    StoreCorruptError,
    permission_problem,
    quarantine,
    read_json,
    update_json,
)
from fx1.interactive.endpoint import EndpointError, EndpointSpec, normalize_endpoint
from fx1.interactive.redact import fingerprint as _secret_fingerprint
from fx1.interactive.redact import redact_text, register_secret
from fx1.serve.backends import MOONSHOT_API_URL

MODELS: tuple[str, ...] = ("fx1", "fx1-lite")
DEFAULT_MODEL = "fx1"

ENV_API_KEY = "MOONSHOT_API_KEY"
ENV_BASE_URL = "FX1_BASE_URL"
ENV_FX1_API_KEY = "DIPCATCHER_FX1_API_KEY"
ENV_FX1_BASE_URL = "DIPCATCHER_FX1_BASE_URL"
ENV_FX1_LITE_API_KEY = "DIPCATCHER_FX1_LITE_API_KEY"
ENV_FX1_LITE_BASE_URL = "DIPCATCHER_FX1_LITE_BASE_URL"

CREDENTIAL_ENV_VARS: frozenset[str] = frozenset(
    {
        ENV_API_KEY,
        ENV_BASE_URL,
        "FX1_API_KEY",
        "FX1_BYOK_API_KEY",
        "FX1_LOCAL_API_KEY",
    }
)

STORE_NAME = "credentials.json"
SCHEMA_VERSION = 2
_MODE_0600 = stat.S_IRUSR | stat.S_IWUSR

_LOCK_TIMEOUT_S = 10.0
_ACL_FAILURE = "Windows ACL was not applied to the credentials store"
_NOTES: dict[str, list[str]] = {}
_NOTES_LOCK = threading.Lock()
_ABSENT = object()

SourceName = Literal["stored", "env:model", "env:shared"]
_EnvSource = Literal["env:model", "env:shared"]


@dataclass(frozen=True)
class ResolvedEndpoint:
    """One model's resolved credential. ``api_key`` is omitted from repr and str."""

    api_key: str
    completion_url: str
    host: str
    source: SourceName
    fingerprint: str

    def __repr__(self) -> str:
        return (
            "ResolvedEndpoint("
            f"completion_url={self.completion_url!r}, host={self.host!r}, "
            f"source={self.source!r}, fingerprint={self.fingerprint!r})"
        )

    def __str__(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class _EnvChoice:
    api_key: str
    raw_url: str
    source: _EnvSource
    url_var: str

    def __repr__(self) -> str:
        url = redact_text(self.raw_url)
        return f"_EnvChoice(source={self.source!r}, url_var={self.url_var!r}, raw_url={url!r})"


# The target path is passed through an environment variable, not inserted into
# PowerShell source or command arguments. The API key never enters this child.
_WINDOWS_PRIVATE_ACL = r"""
$ErrorActionPreference = 'Stop'
$target = [Environment]::GetEnvironmentVariable('FX1_ACL_TARGET', 'Process')
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$directory = [Environment]::GetEnvironmentVariable('FX1_ACL_IS_DIR', 'Process') -eq '1'
$verifyOnly = [Environment]::GetEnvironmentVariable('FX1_ACL_VERIFY_ONLY', 'Process') -eq '1'
if (-not $verifyOnly) {
    if ($directory) {
        $acl = [System.Security.AccessControl.DirectorySecurity]::new()
    } else {
        $acl = [System.Security.AccessControl.FileSecurity]::new()
    }
    $acl.SetAccessRuleProtection($true, $false)
    if ($directory) {
        $inherit = [System.Security.AccessControl.InheritanceFlags]::ContainerInherit -bor
            [System.Security.AccessControl.InheritanceFlags]::ObjectInherit
        $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
            $sid, [System.Security.AccessControl.FileSystemRights]::FullControl,
            $inherit, [System.Security.AccessControl.PropagationFlags]::None,
            [System.Security.AccessControl.AccessControlType]::Allow
        )
    } else {
        $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
            $sid, [System.Security.AccessControl.FileSystemRights]::FullControl,
            [System.Security.AccessControl.AccessControlType]::Allow
        )
    }
    $acl.AddAccessRule($rule)
    if ($directory) {
        [System.IO.Directory]::SetAccessControl($target, $acl)
    } else {
        [System.IO.File]::SetAccessControl($target, $acl)
    }
}
if ($directory) {
    $effective = [System.IO.Directory]::GetAccessControl($target)
} else {
    $effective = [System.IO.File]::GetAccessControl($target)
}
$rules = $effective.GetAccessRules(
    $true, $true, [System.Security.Principal.SecurityIdentifier]
)
$owner = $effective.GetOwner([System.Security.Principal.SecurityIdentifier])
$full = [System.Security.AccessControl.FileSystemRights]::FullControl
if (-not $effective.AreAccessRulesProtected -or $rules.Count -ne 1 -or
    $owner.Value -ne $sid.Value -or
    $rules[0].IdentityReference.Value -ne $sid.Value -or
    $rules[0].AccessControlType -ne [System.Security.AccessControl.AccessControlType]::Allow -or
    $rules[0].IsInherited -or
    ($rules[0].FileSystemRights -band $full) -ne $full) {
    throw 'credential ACL verification failed'
}
"""


def _restrict_windows_acl(
    path: Path, *, directory: bool = False, verify_only: bool = False
) -> None:
    """Set or verify a protected, current-user-only DACL before secrets are written."""
    system_root = os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR")
    if not system_root:
        raise RuntimeError("cannot secure Windows credentials: SystemRoot is unavailable")
    powershell = Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    env = {
        "SYSTEMROOT": system_root,
        "WINDIR": system_root,
        "PATH": os.environ.get("PATH", ""),
        "FX1_ACL_TARGET": str(path),
        "FX1_ACL_IS_DIR": "1" if directory else "0",
        "FX1_ACL_VERIFY_ONLY": "1" if verify_only else "0",
    }
    try:
        result = subprocess.run(
            [str(powershell), "-NoProfile", "-NonInteractive", "-Command", _WINDOWS_PRIVATE_ACL],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "cannot secure Windows credentials; use MOONSHOT_API_KEY in the environment"
        ) from exc
    if result.returncode != 0:
        if verify_only:
            raise RuntimeError(
                "existing Windows credential path has an unsafe ACL; "
                "use a new dedicated FX1_CONFIG_DIR or secure it manually"
            )
        raise RuntimeError(
            "cannot verify a private Windows credential ACL; "
            "use MOONSHOT_API_KEY in the environment"
        )


def _windows_config_parent(path: Path) -> Path:
    parent = path.parent.resolve()
    if parent == Path(parent.anchor) or parent == Path.home().resolve():
        raise RuntimeError("store credentials in a dedicated configuration directory")
    return parent


def config_dir() -> Path:
    override = os.environ.get("FX1_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".fx1"


def store_path() -> Path:
    return config_dir() / STORE_NAME


def _resolve_path(path: Path | None) -> Path:
    if path is None:
        return store_path()
    return path


def default_base_url() -> str:
    from fx1.serve.backends import MOONSHOT_API_URL

    return os.environ.get(ENV_BASE_URL, MOONSHOT_API_URL)


def suggest_base_url(environ: Mapping[str, str] | None = None) -> str:
    """Shared base URL from ``FX1_BASE_URL``, else the Moonshot constant.

    Per-model variables are not consulted. No Artificial Hedge hostname
    is invented; the only implicit host is :data:`MOONSHOT_API_URL`.
    """
    chosen = _env_get(_environ(environ), ENV_BASE_URL)
    if chosen:
        return chosen
    return MOONSHOT_API_URL


def fingerprint(api_key: str) -> str:
    """Presence-only fingerprint: the last four characters, never the value."""
    return _secret_fingerprint(api_key)


def check_model(model: str) -> str:
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}; choose from {sorted(MODELS)}")
    return model


def host_of(base_url: str) -> str:
    return urllib.parse.urlparse(base_url).hostname or base_url


def _normalize(entry: Mapping[str, str]) -> dict[str, str]:
    """Read-compat: accept the pre-base-URL schema (``key``/``env_var``)."""
    api_key = entry.get("api_key") or entry.get("key") or ""
    base_url = entry.get("base_url") or default_base_url()
    return {"api_key": api_key, "base_url": base_url, "set_at": entry.get("set_at", "")}


def _prepare_store_directory(path: Path) -> None:
    if os.name != "nt":
        path.parent.mkdir(parents=True, exist_ok=True)
        return
    parent = _windows_config_parent(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        _restrict_windows_acl(parent, directory=True, verify_only=True)
    else:
        _restrict_windows_acl(parent, directory=True)


def _save(payload: dict[str, dict[str, str]], path: Path | None = None) -> Path:
    path = path or store_path()
    _prepare_store_directory(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".credentials-", suffix=".tmp")
    try:
        if os.name == "nt":
            if path.is_file():
                _restrict_windows_acl(path, verify_only=True)
            _restrict_windows_acl(Path(tmp))
        else:
            os.chmod(tmp, _MODE_0600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fd = -1
            json.dump({"profiles": payload}, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
        if os.name == "nt":
            try:
                _restrict_windows_acl(path, verify_only=True)
            except RuntimeError:
                path.unlink(missing_ok=True)
                raise
        else:
            os.chmod(path, _MODE_0600)
    finally:
        if fd >= 0:
            os.close(fd)
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def _note(path: Path, message: str) -> None:
    key = str(path)
    with _NOTES_LOCK:
        bucket = _NOTES.setdefault(key, [])
        if message not in bucket:
            bucket.append(message)


def _notes_for(path: Path) -> list[str]:
    with _NOTES_LOCK:
        return list(_NOTES.get(str(path), ()))


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _is_windows() -> bool:
    return sys.platform == "win32"


def _empty_document() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "profiles": {}}


def _now_stamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _key_has_junk(key: str) -> bool:
    return any(_key_char_junk(ch) for ch in key)


def _key_char_junk(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch) in {"Cc", "Cf"} or ord(ch) > 255


def _clean_api_key(api_key: str, model: str) -> str:
    """Strip surrounding whitespace. Reject internal whitespace or controls."""
    stripped = api_key.strip()
    if not stripped:
        raise ValueError(f"empty API key for model {model!r}")
    if _key_has_junk(stripped):
        raise ValueError(f"API key for model {model!r} contains whitespace or control characters")
    return stripped


def _environ(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    if environ is None:
        return os.environ
    return environ


def _env_get(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "")
    if not isinstance(value, str):
        return ""
    return value.strip()


def _model_env_names(model: str) -> tuple[str, str]:
    if model == "fx1-lite":
        return ENV_FX1_LITE_API_KEY, ENV_FX1_LITE_BASE_URL
    return ENV_FX1_API_KEY, ENV_FX1_BASE_URL


def _shared_base_url(env: Mapping[str, str]) -> str:
    return _env_get(env, ENV_BASE_URL) or MOONSHOT_API_URL


def _model_env_choice(model_key: str, url_var: str, env: Mapping[str, str]) -> _EnvChoice:
    explicit = _env_get(env, url_var)
    if explicit:
        return _EnvChoice(model_key, explicit, "env:model", url_var)
    shared = _env_get(env, ENV_BASE_URL)
    if shared:
        return _EnvChoice(model_key, shared, "env:model", ENV_BASE_URL)
    return _EnvChoice(model_key, MOONSHOT_API_URL, "env:model", "MOONSHOT_API_URL")


def _env_choice(model: str, env: Mapping[str, str]) -> _EnvChoice | None:
    key_var, url_var = _model_env_names(model)
    model_key = _env_get(env, key_var)
    if model_key:
        return _model_env_choice(model_key, url_var, env)
    shared_key = _env_get(env, ENV_API_KEY)
    if not shared_key:
        return None
    explicit = _env_get(env, ENV_BASE_URL)
    raw_url = explicit or MOONSHOT_API_URL
    return _EnvChoice(shared_key, raw_url, "env:shared", ENV_BASE_URL)


def _try_normalize(raw: str) -> tuple[EndpointSpec | None, str]:
    try:
        return normalize_endpoint(raw), ""
    except EndpointError as exc:
        return None, str(exc)


def _entry_key(entry: Mapping[str, Any]) -> str:
    for name in ("api_key", "key"):
        value = entry.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _entry_url(entry: Mapping[str, Any], env: Mapping[str, str]) -> str:
    value = entry.get("base_url")
    if isinstance(value, str) and value.strip():
        return value
    return _shared_base_url(env)


def _set_at(entry: Mapping[str, Any]) -> str:
    value = entry.get("set_at")
    if isinstance(value, str) and value:
        return value
    return "-"


def _safe_host(raw: str) -> str:
    try:
        host = urllib.parse.urlparse(raw).hostname
    except ValueError:
        return "-"
    if not host:
        return "-"
    return host.lower()


def _presence_fingerprint(key: str) -> str:
    stripped = key.strip()
    if not stripped or _key_has_junk(stripped):
        return "-"
    return fingerprint(stripped)


def _extract_profiles(current: Any) -> dict[str, Any] | None:
    if not isinstance(current, dict):
        return None
    if "profiles" not in current:
        return {}
    raw = current["profiles"]
    if not isinstance(raw, dict):
        return None
    return {key: value for key, value in raw.items() if isinstance(key, str)}


def _quarantine_bad_shape(path: Path) -> None:
    if path.is_file():
        quarantine(path)


def _profiles_of(current: Any, path: Path) -> dict[str, Any]:
    extracted = _extract_profiles(current)
    if extracted is None:
        _quarantine_bad_shape(path)
        return {}
    return extracted


def _heal_permissions(path: Path) -> None:
    if os.name != "posix" or path.is_symlink() or not path.is_file():
        return
    try:
        info = path.lstat()
        mode = stat.S_IMODE(info.st_mode)
    except OSError:
        return
    if not stat.S_ISREG(info.st_mode) or info.st_nlink > 1 or mode & 0o077 == 0:
        return
    try:
        os.chmod(path, 0o600)
    except OSError:
        return
    _note(path, f"credentials store permissions were repaired to 0600 (was {mode:04o})")


def _symlink_problem(path: Path) -> str | None:
    if path.is_symlink():
        return "credentials store must not be a symlink"
    if path.parent.is_symlink():
        return "credentials directory must not be a symlink"
    return _special_store(path)


def _special_store(path: Path) -> str | None:
    try:
        if not path.exists():
            return None
        info = path.lstat()
    except OSError:
        return "credentials store is unreadable"
    if not stat.S_ISREG(info.st_mode):
        return "credentials store must be a regular file"
    if info.st_nlink > 1:
        return "credentials store must not be a hard link"
    return None


def _posix_mode(path: Path) -> int | None:
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return None


def _owned_by_us(path: Path) -> bool:
    if os.name != "posix":
        return False
    try:
        return path.stat().st_uid == os.getuid()
    except OSError:
        return False


def _chmod_owned_dir(parent: Path) -> bool:
    if not _owned_by_us(parent) or parent.is_symlink():
        return False
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(parent, flags)
        try:
            os.fchmod(fd, 0o700)
        finally:
            os.close(fd)
    except OSError:
        return False
    return _posix_mode(parent) == 0o700


def _directory_warning(store: Path) -> str | None:
    try:
        _prepare_config_dir(store)
    except ValueError as exc:
        return str(exc)
    return None


def _prepare_config_dir(store: Path) -> None:
    problem = _symlink_problem(store)
    if problem is not None:
        raise ValueError(problem)
    parent = store.parent
    if os.name != "posix" or not parent.is_dir():
        return
    mode = _posix_mode(parent)
    if mode is None or mode & 0o077 == 0:
        return
    if not _chmod_owned_dir(parent):
        raise ValueError("credentials directory is group or world accessible")
    _note(store, "credentials directory permissions were tightened to 0700")


def _load(path: Path | None = None) -> dict[str, Any]:
    target = _resolve_path(path)
    _prepare_config_dir(target)
    if not target.exists():
        return {}
    _heal_permissions(target)
    with FileLock(target, timeout_s=_LOCK_TIMEOUT_S):
        return _load_locked(target)


def _load_locked(target: Path) -> dict[str, Any]:
    if not target.is_file():
        return {}
    try:
        current = read_json(target)
    except StoreCorruptError:
        quarantine(target)
        return {}
    except FileNotFoundError:
        return {}
    return _profiles_of(current, target)


def _quarantine_corrupt(path: Path) -> None:
    with FileLock(path, timeout_s=_LOCK_TIMEOUT_S):
        if not path.is_file():
            return
        try:
            read_json(path)
        except StoreCorruptError:
            quarantine(path)
        except FileNotFoundError:
            return


def _commit(path: Path | None, mutate: Callable[[dict[str, Any]], dict[str, Any]]) -> Path:
    target = _resolve_path(path)
    _prepare_config_dir(target)

    def apply(current: Any) -> dict[str, Any]:
        profiles = mutate(_profiles_of(current, target))
        return {"schema_version": SCHEMA_VERSION, "profiles": profiles}

    try:
        update_json(target, apply, default=_empty_document(), lock_timeout_s=_LOCK_TIMEOUT_S)
    except StoreCorruptError:
        _quarantine_corrupt(target)
        update_json(target, apply, default=_empty_document(), lock_timeout_s=_LOCK_TIMEOUT_S)
    _protect_after_write(target)
    return target


def _protect_after_write(path: Path) -> None:
    if _is_windows():
        protect_store_windows(path)


def _requested_url(base_url: str | None) -> str:
    if base_url is None or base_url.strip() == "":
        return default_base_url()
    return base_url


def _spec_or_value_error(raw_url: str) -> EndpointSpec:
    try:
        return normalize_endpoint(raw_url)
    except EndpointError as exc:
        raise ValueError(f"invalid base URL: {exc}") from None


def _put_profile(
    profiles: dict[str, Any], model: str, api_key: str, spec: EndpointSpec
) -> dict[str, Any]:
    updated = dict(profiles)
    updated[model] = {
        "api_key": api_key,
        "base_url": spec.completion_url,
        "service_origin": spec.origin,
        "set_at": _now_stamp(),
    }
    return updated


def set_endpoint_spec(
    model: str,
    api_key: str,
    base_url: str | None = None,
    *,
    path: Path | None = None,
) -> tuple[Path, EndpointSpec]:
    """Store *model*'s endpoint and return the path plus the resolved route."""
    model = check_model(model)
    key = _clean_api_key(api_key, model)
    spec = _spec_or_value_error(_requested_url(base_url))
    register_secret(key)
    saved = _commit(path, lambda profiles: _put_profile(profiles, model, key, spec))
    return saved, spec


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
    from fx1.serve.backends import hosted_api_url_problem

    base = (base_url or "").strip() or default_base_url()
    problem = hosted_api_url_problem(base)
    if problem is not None:
        raise ValueError(f"invalid base URL: {problem}")
    payload = _load(path)
    payload[model] = {
        "api_key": api_key,
        "base_url": base,
        "set_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return _save(payload, path)


def remove_endpoint(model: str, *, path: Path | None = None) -> bool:
    model = check_model(model)
    target = _resolve_path(path)
    if not target.is_file():
        return False
    if model not in _load(target):
        return False

    def mutate(current: dict[str, Any]) -> dict[str, Any]:
        updated = dict(current)
        updated.pop(model, None)
        return updated

    _commit(target, mutate)
    return True


def _row(model: str, host: str, status: str, source: str, fp: str, set_at: str) -> dict[str, str]:
    return {
        "model": model,
        "host": host,
        "status": status,
        "source": source,
        "fingerprint": fp,
        "set_at": set_at,
    }


def _unset_row(model: str, env: Mapping[str, str]) -> dict[str, str]:
    return _row(model, host_of(_shared_base_url(env)), "unset", "unset", "-", "-")


def _stored_row(model: str, entry: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, str]:
    key = _entry_key(entry)
    raw_url = _entry_url(entry, env)
    spec, _reason = _try_normalize(raw_url)
    fp = _presence_fingerprint(key)
    set_at = _set_at(entry)
    if spec is None:
        return _row(model, _safe_host(raw_url), "invalid-url", "stored", fp, set_at)
    return _row(model, spec.host, "set", "stored", fp, set_at)


def _env_row(model: str, choice: _EnvChoice) -> dict[str, str]:
    spec, _reason = _try_normalize(choice.raw_url)
    if spec is None:
        return _row(model, _safe_host(choice.raw_url), "invalid-url", choice.source, "-", "-")
    return _row(model, spec.host, "env", choice.source, "-", "-")


def _env_or_unset(model: str, env: Mapping[str, str]) -> dict[str, str]:
    choice = _env_choice(model, env)
    if choice is None:
        return _unset_row(model, env)
    return _env_row(model, choice)


def _model_row(model: str, raw_entry: Any, env: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(raw_entry, dict):
        return _row(model, "-", "invalid-url", "stored", "-", "-")
    if not _entry_key(raw_entry):
        return _env_or_unset(model, env)
    return _stored_row(model, raw_entry, env)


def list_models(
    *, path: Path | None = None, environ: Mapping[str, str] | None = None
) -> list[dict[str, str]]:
    """Presence-only rows for both models.

    Columns: model, host, status, source, fingerprint, set_at.
    ``status`` is ``set``, ``env``, ``unset``, or ``invalid-url``.
    ``source`` is ``stored``, ``env:model``, ``env:shared``, or ``unset``.
    Environment rows keep fingerprint ``-`` (the key is not in the store).
    """
    env = _environ(environ)
    payload = _load(path)
    rows: list[dict[str, str]] = []
    for model in MODELS:
        if model not in payload:
            rows.append(_env_or_unset(model, env))
        else:
            rows.append(_model_row(model, payload[model], env))
    return rows


def _resolved(key: str, spec: EndpointSpec, source: SourceName) -> ResolvedEndpoint:
    register_secret(key)
    return ResolvedEndpoint(
        api_key=key,
        completion_url=spec.completion_url,
        host=spec.host,
        source=source,
        fingerprint=fingerprint(key),
    )


def _resolve_stored(
    model: str, entry: Mapping[str, Any], env: Mapping[str, str]
) -> ResolvedEndpoint:
    key = _clean_api_key(_entry_key(entry), model)
    spec, reason = _try_normalize(_entry_url(entry, env))
    if spec is None:
        register_secret(key)
        raise ValueError(f"invalid stored base URL: {reason}")
    return _resolved(key, spec, "stored")


def _resolve_choice(model: str, choice: _EnvChoice) -> ResolvedEndpoint:
    key = _clean_api_key(choice.api_key, model)
    spec, reason = _try_normalize(choice.raw_url)
    if spec is None:
        register_secret(key)
        raise ValueError(f"invalid {choice.url_var}: {reason}")
    return _resolved(key, spec, choice.source)


def _from_stored_entry(model: str, raw: Any, env: Mapping[str, str]) -> ResolvedEndpoint | object:
    if not isinstance(raw, dict):
        raise ValueError(f"invalid stored profile for model {model!r}")
    if not _entry_key(raw):
        return _ABSENT
    return _resolve_stored(model, raw, env)


def _from_env(model: str, env: Mapping[str, str]) -> ResolvedEndpoint | None:
    choice = _env_choice(model, env)
    if choice is None:
        return None
    return _resolve_choice(model, choice)


def resolve_endpoint_info(
    model: str, *, path: Path | None = None, environ: Mapping[str, str] | None = None
) -> ResolvedEndpoint | None:
    """Resolve *model* without reading or writing ``os.environ`` when *environ* is passed.

    Stored profile wins, then the per-model environment, then the shared
    environment. ``None`` when no key is configured. Registers the key with
    the process redactor. Does not mutate ``os.environ``.
    """
    model = check_model(model)
    env = _environ(environ)
    payload = _load(path)
    if model in payload:
        resolved = _from_stored_entry(model, payload[model], env)
        if isinstance(resolved, ResolvedEndpoint):
            return resolved
    return _from_env(model, env)


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
            from fx1.serve.backends import hosted_api_url_problem

            problem = hosted_api_url_problem(normalized["base_url"])
            if problem is not None:
                raise ValueError(f"invalid stored base URL: {problem}")
            return normalized["api_key"], normalized["base_url"]
    api_key = env.get(ENV_API_KEY)
    if api_key:
        from fx1.serve.backends import MOONSHOT_API_URL, hosted_api_url_problem

        base_url = env.get(ENV_BASE_URL, MOONSHOT_API_URL)
        problem = hosted_api_url_problem(base_url)
        if problem is not None:
            raise ValueError(f"invalid {ENV_BASE_URL}: {problem}")
        return api_key, base_url
    return None


def scrub_credentials(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    return {key: value for key, value in source.items() if key not in CREDENTIAL_ENV_VARS}


def subprocess_env_for_model(
    model: str,
    *,
    path: Path | None = None,
    inject_credentials: bool = True,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = scrub_credentials(environ)
    if not inject_credentials:
        return env
    resolved = resolve_endpoint(model, path=path)
    if not resolved:
        raise RuntimeError(
            f"no endpoint for model {model!r}; run `dipcatcher` or `fxi setup` "
            f"to enter your API key and base URL, or export {ENV_API_KEY}"
        )
    api_key, base_url = resolved
    env = dict(env)
    env[ENV_API_KEY] = api_key
    env[ENV_BASE_URL] = base_url
    return env


def apply_profile(
    model: str, *, path: Path | None = None, environ: MutableMapping[str, str] | None = None
) -> list[str]:
    """Inject *model*'s endpoint into *environ* (default ``os.environ``).

    Reads the process environment for fallback, not *environ*. Returns the
    env var names set. Fail-closed: raises ``RuntimeError`` naming the model
    when no key resolves. New concierge paths should use
    :func:`resolve_endpoint_info` instead of injecting the shared variables.
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


def _json_problem(path: Path) -> str | None:
    try:
        payload = read_json(path)
    except StoreCorruptError as exc:
        return exc.reason
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict):
        return "credentials store is not a JSON object"
    profiles = payload.get("profiles", {})
    if not isinstance(profiles, dict):
        return "credentials store profiles field is not an object"
    return None


def _corrupt_warnings(path: Path) -> list[str]:
    warnings: list[str] = []
    if path.is_file():
        reason = _json_problem(path)
        if reason is not None:
            warnings.append(f"corrupt credentials store: {reason}")
    parent = path.parent
    if not parent.is_dir():
        return warnings
    prefix = f"{path.name}.corrupt-"
    for sibling in sorted(parent.glob(prefix + "*")):
        warnings.append(f"quarantined corrupt credentials store: {sibling.name}")
    return warnings


def store_health(path: Path | None = None) -> list[str]:
    """Warnings for a group/world-readable, corrupt, or unprotected store.

    On POSIX, group/world bits are repaired to ``0600`` when ``chmod``
    succeeds; the repair is reported. Windows ACL failures recorded by
    :func:`protect_store_windows` are included. Messages never contain
    secret values.
    """
    target = _resolve_path(path)
    problem = _symlink_problem(target)
    if problem is not None:
        return [problem]
    _heal_permissions(target)
    warnings: list[str] = []
    dir_problem = _directory_warning(target)
    if dir_problem is not None:
        warnings.append(dir_problem)
    if target.is_file():
        problem = permission_problem(target)
        if problem is not None:
            warnings.append(problem)
    warnings.extend(_corrupt_warnings(target))
    warnings.extend(_notes_for(target))
    return _dedupe(warnings)


def _windows_username(username: str | None) -> str:
    if username is None:
        username = os.environ.get("USERNAME", "")
        if not username:
            username = _current_user()
    if not username or any(unicodedata.category(ch) == "Cc" for ch in username):
        return ""
    return username


def _current_user() -> str:
    try:
        return getuser()
    except OSError:
        return ""


def _runner_succeeded(completed: Any) -> bool:
    code = getattr(completed, "returncode", None)
    return isinstance(code, int) and code == 0


def protect_store_windows(
    path: Path,
    *,
    runner: Callable[..., Any] = subprocess.run,
    username: str | None = None,
) -> bool:
    """Restrict *path* to the current user via ``icacls``. Never raises.

    No-op (returns ``False``, no warning) when ``sys.platform`` is not
    ``win32``. On Windows, a failed ``icacls`` is reported by
    :func:`store_health`. *runner* is invoked with an argument list and
    ``shell`` left unset.
    """
    if not _is_windows():
        return False
    user = _windows_username(username)
    if not user:
        _note(path, _ACL_FAILURE)
        return False
    argv = ["icacls", os.fspath(path), "/inheritance:r", "/grant:r", f"{user}:F"]
    try:
        completed = runner(argv, capture_output=True, check=False)
    except OSError:
        _note(path, _ACL_FAILURE)
        return False
    if _runner_succeeded(completed):
        return True
    _note(path, _ACL_FAILURE)
    return False
