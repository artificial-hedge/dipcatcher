"""Crown-jewel byte pins for the gate-defining files.

The epoch chains cover ``receipts/``, ``verifier/``, ``quality/`` and
``.github/workflows/*.yml`` — but the files that *define* the gates live at
repo root: lint/type/test config, the dependency lock, the Makefile that
names the checks, the pre-commit hooks, the secret-scan allowlist, and
``AGENTS.md`` — the honesty contract itself. A silent edit there weakens
every check downstream of it.

``quality/crown_jewels.json`` pins ``relpath -> sha256`` for exactly
``DEFAULT_JEWELS``. The coverage set is code-defined on purpose: removing a
pin entry fails ``jewel_unpinned``, an entry outside the set fails
``jewel_unexpected`` — the pin cannot quietly shrink or grow. A tampered or
deleted file fails ``jewel_mutated`` / ``jewel_missing``; a pinned path that
is a symlink fails ``jewel_symlink`` (content substitution).

The pin file is itself a member of the ``quality/*.json`` epoch chain, so
its bytes are chain-stamped; the recursion bottoms out at git review of the
pin/epoch diff — the documented trust boundary, same as the epoch head pin.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import hash_bytes

CROWN_JEWELS_SCHEMA = "crown_jewels.v1"
DEFAULT_PIN_PATH = Path("quality/crown_jewels.json")

# The files that define or authorize every gate — byte-pinned exactly. Adding
# a jewel is a code change here (review-visible), not a pin edit.
DEFAULT_JEWELS: tuple[str, ...] = (
    ".gitleaks.toml",  # secret-scan allowlist — removing a rule opens exfil lanes
    ".pre-commit-config.yaml",  # local hook stack
    "AGENTS.md",  # the honesty contract
    "Makefile",  # the gate command definitions
    "conftest.py",  # global pytest fixtures/collection
    "mkdocs.yml",  # docs nav gate
    "pyproject.toml",  # deps, ruff/mypy/pytest config
    "tests/conftest.py",  # suite-level fixtures
    "uv.lock",  # the supply-chain lock
)


def crown_jewel_digests(
    root: Path | str, jewels: tuple[str, ...] = DEFAULT_JEWELS
) -> dict[str, str]:
    """``{relpath: sha256}`` over the pinned set; fails closed on missing."""
    base = Path(root)
    out: dict[str, str] = {}
    for rel in jewels:
        path = base / rel
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"crown jewel not a regular file: {rel}")
        out[rel] = hash_bytes(path.read_bytes())
    return out


def write_crown_jewels_pin(
    root: Path | str = ".",
    pin_path: Path | str = DEFAULT_PIN_PATH,
) -> Path:
    """(Re)write the pin over ``DEFAULT_JEWELS``; overwrite-atomic."""
    from quant_fund.utils.atomicio import atomic_write_text

    payload = {"schema": CROWN_JEWELS_SCHEMA, "files": crown_jewel_digests(root)}
    path = Path(pin_path)
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_crown_jewels_pin(pin_path: Path | str = DEFAULT_PIN_PATH) -> dict[str, str]:
    """Load the pin's files map; raises ValueError on malformed content."""
    path = Path(pin_path)
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema") != CROWN_JEWELS_SCHEMA:
        raise ValueError(f"{path} is not a {CROWN_JEWELS_SCHEMA} pin")
    files = raw.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"{path} has no 'files' map")
    out: dict[str, str] = {}
    for name, sha in files.items():
        if not (
            isinstance(name, str)
            and isinstance(sha, str)
            and len(sha) == 64
            and all(c in "0123456789abcdef" for c in sha)
        ):
            raise ValueError(f"{path} entry {name!r} malformed")
        out[name] = sha
    return out


def crown_jewels_errors(
    root: Path | str = ".",
    pin_path: Path | str = DEFAULT_PIN_PATH,
    *,
    jewels: tuple[str, ...] = DEFAULT_JEWELS,
) -> list[str]:
    """Verify every jewel against the pin.

    Errors are sorted ``jewel_*`` codes. The pinned set must equal
    ``jewels`` exactly — a pin missing a jewel hides a removal
    (``jewel_unpinned``), an extra entry is an undeclared scope change
    (``jewel_unexpected``).
    """
    base = Path(root)
    pin = Path(pin_path)
    if not pin.is_file():
        return ["pin_missing"]
    try:
        pinned = load_crown_jewels_pin(pin)
    except (OSError, ValueError) as exc:
        return [f"pin_malformed:{exc}"]
    errors: list[str] = []
    for name in sorted(set(jewels) - set(pinned)):
        errors.append(f"jewel_unpinned:{name}")
    for name in sorted(set(pinned) - set(jewels)):
        errors.append(f"jewel_unexpected:{name}")
    for name, expected in sorted(pinned.items()):
        path = base / name
        if name not in jewels:
            continue  # already reported as unexpected
        if path.is_symlink():
            errors.append(f"jewel_symlink:{name}")
        elif not path.is_file():
            errors.append(f"jewel_missing:{name}")
        elif hash_bytes(path.read_bytes()) != expected:
            errors.append(f"jewel_mutated:{name}")
    return sorted(errors)
