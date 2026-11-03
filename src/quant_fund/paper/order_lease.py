"""Exclusive flock and a fencing token for one paper state directory.

The lock file is never removed because a recorded pid is dead. A SIGKILL
drops the kernel flock. The next acquirer increments the token on the same
inode. Two processes cannot both hold the flock, and a writer whose token
no longer matches the file must not submit.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
from contextlib import suppress
from pathlib import Path

from quant_fund.observe.order_guard import note_split_brain
from quant_fund.observe.write_halt import DurableWriteError, errno_code, note_write_failure
from quant_fund.paper.runtime_log import forget_runtime

SCHEMA = "paper_order_lease.v1"
LOCK_NAME = ".dipcatcher-order.lock"
_HELD: dict[str, tuple[int, int]] = {}


class SplitBrainError(RuntimeError):
    """This process does not hold the current paper-order fence."""

    def __init__(self) -> None:
        super().__init__("split_brain")


def lock_path(state: Path) -> Path:
    return state / LOCK_NAME


def _key(state: Path) -> str:
    return str(state.resolve())


def held_token(state: Path) -> int | None:
    found = _HELD.get(_key(state))
    if found is None:
        return None
    return found[1]


def read_disk_token(state: Path) -> int:
    """Token recorded in the lock file, or 0 when the file is absent or junk."""
    path = lock_path(state)
    if not path.is_file():
        return 0
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return 0
    return _parse_token(raw.encode("utf-8"))


def release_order_lease(state: Path | None = None) -> None:
    """Drop a flock held in this process. The lock file stays on disk."""
    keys = list(_HELD) if state is None else [_key(state)]
    for key in keys:
        found = _HELD.pop(key, None)
        if found is None:
            continue
        descriptor, _token = found
        with suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        with suppress(OSError):
            os.close(descriptor)


def acquire_order_lease(state: Path) -> int:
    """Take the nonblocking flock, then increment the token. Idempotent here."""
    state.mkdir(parents=True, exist_ok=True)
    key = _key(state)
    found = _HELD.get(key)
    if found is not None:
        return found[1]
    descriptor = _open_lock(state)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(descriptor)
        note_split_brain()
        raise SplitBrainError() from exc
    try:
        token = _bump(descriptor)
    except DurableWriteError:
        os.close(descriptor)
        raise
    except OSError as exc:
        os.close(descriptor)
        note_split_brain()
        raise SplitBrainError() from exc
    _HELD[key] = (descriptor, token)
    forget_runtime(state)
    return token


def assert_fence(state: Path) -> None:
    """Refuse when this process does not hold the token now on disk."""
    found = _HELD.get(_key(state))
    if found is None or read_disk_token(state) != found[1]:
        note_split_brain()
        raise SplitBrainError()


def _open_lock(state: Path) -> int:
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(lock_path(state), flags, 0o644)
    except OSError as exc:
        _disk(exc)
        note_split_brain()
        raise SplitBrainError() from exc


def _disk(exc: OSError) -> None:
    code = errno_code(exc.errno or 0)
    if code == "write_failed":
        return
    note_write_failure(code)
    raise DurableWriteError(code) from exc


def _parse_token(raw: bytes) -> int:
    text = raw.strip()
    if not text:
        return 0
    try:
        payload = json.loads(text.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, dict):
        return 0
    token = payload.get("token")
    if isinstance(token, bool) or not isinstance(token, int) or token < 0:
        return 0
    return token


def _read_token(descriptor: int) -> int:
    os.lseek(descriptor, 0, os.SEEK_SET)
    return _parse_token(os.read(descriptor, 8192))


def _payload(token: int) -> bytes:
    body = {"pid": os.getpid(), "schema": SCHEMA, "token": token}
    encoded = json.dumps(body, separators=(",", ":"), sort_keys=True)
    return (encoded + "\n").encode("utf-8")


def _bump(descriptor: int) -> int:
    token = _read_token(descriptor) + 1
    blob = _payload(token)
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    view = memoryview(blob)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError(errno.EIO, "short write")
        view = view[written:]
    os.fsync(descriptor)
    return token
