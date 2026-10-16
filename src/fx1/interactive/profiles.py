"""Model-endpoint store for ``fxi``.

fx-1 ships two hosted model names; each profile here is one model's
endpoint — the API key and base URL used to talk to it:

- ``fx1``      — the full fx-1 model.
- ``fx1-lite`` — the light fx-1 model.

Values live in ``$FX1_CONFIG_DIR/credentials.json`` (default
``~/.fx1/credentials.json``), with POSIX mode 0600 or a protected Windows
current-user ACL, written atomically. Reporting is presence-only: raw keys
never leave this module except as injected
environment values (mirrors ``fx1.doctor`` and the harness conventions).

Injection: applying a profile sets ``MOONSHOT_API_KEY`` and
``FX1_BASE_URL`` for child processes and in-process backends.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

MODELS: tuple[str, ...] = ("fx1", "fx1-lite")
DEFAULT_MODEL = "fx1"

ENV_API_KEY = "MOONSHOT_API_KEY"
ENV_BASE_URL = "FX1_BASE_URL"

STORE_NAME = "credentials.json"
_MODE_0600 = stat.S_IRUSR | stat.S_IWUSR

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


def default_base_url() -> str:
    from fx1.serve.backends import MOONSHOT_API_URL

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
    if os.name == "nt":
        try:
            mode = path.stat().st_mode
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise RuntimeError("cannot inspect Windows credential path") from exc
        if not stat.S_ISREG(mode):
            raise RuntimeError("Windows credential path is not a regular file")
        _restrict_windows_acl(_windows_config_parent(path), directory=True, verify_only=True)
        _restrict_windows_acl(path, verify_only=True)
    elif not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    profiles = raw.get("profiles")
    if isinstance(profiles, dict):
        return cast("dict[str, dict[str, str]]", profiles)
    return {}


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
            fd = -1  # fdopen owns and closes the descriptor from here onward.
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
