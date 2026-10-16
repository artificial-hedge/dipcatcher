"""Checkpoint pin and process launch helpers for local fx-1 engines."""

from __future__ import annotations

import json
import os
import shlex
import string
import subprocess
import sys
from pathlib import Path
from typing import Any, cast


def checkpoint_weights_sha(root: Path) -> str | None:
    """Read the declared weight pin, refusing a manifestless fx-1 checkpoint."""
    manifest_path = root / "weights.manifest.json"
    if not manifest_path.is_file():
        if (root / "weights.safetensors").is_file():
            raise RuntimeError("local fx-1 weights require a SHA-256 manifest")
        return None  # Raw third-party checkpoint layouts have no fx-1 manifest.
    try:
        manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("cannot read local fx-1 weights manifest") from exc
    sha = manifest_obj.get("weights_sha256") if isinstance(manifest_obj, dict) else None
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(char not in string.hexdigits for char in sha)
    ):
        raise RuntimeError("local fx-1 weights manifest has no valid SHA-256 pin")
    return sha.lower()


def _split_windows_command(command: str) -> list[str]:
    """Parse before template expansion so paths with spaces stay one argument."""
    import ctypes
    from ctypes import wintypes

    # Linux typeshed omits these Windows-only ctypes members even inside this
    # runtime-only function.
    ctypes_win = cast(Any, ctypes)
    shell32 = ctypes_win.WinDLL("shell32", use_last_error=True)
    parse = shell32.CommandLineToArgvW
    parse.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    parse.restype = ctypes.POINTER(wintypes.LPWSTR)
    count = ctypes.c_int()
    parsed = parse(command, ctypes.byref(count))
    if not parsed:
        raise OSError(ctypes_win.get_last_error(), "cannot parse local engine command")
    try:
        return [str(parsed[index]) for index in range(count.value)]
    finally:
        kernel32 = ctypes_win.WinDLL("kernel32", use_last_error=True)
        free = kernel32.LocalFree
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_void_p
        free(ctypes.cast(parsed, ctypes.c_void_p))


def spawn_local_engine(serve_cmd: str, root: Path) -> subprocess.Popen[bytes]:
    """Launch an argv-only engine without forwarding the signing secrets."""
    command = serve_cmd.lstrip()
    if not command:
        raise RuntimeError("local fx-1 engine command is empty")
    template_argv = _split_windows_command(command) if os.name == "nt" else shlex.split(command)
    argv = [
        string.Template(arg).substitute(checkpoint_dir=str(root), python=sys.executable)
        for arg in template_argv
    ]
    if not argv or not argv[0]:
        raise RuntimeError("local fx-1 engine command is empty")
    env = dict(os.environ, FX1_CHECKPOINT_DIR=str(root))
    env.pop("FX1_RELEASE_SIGNING_KEY", None)
    env.pop("FX1_SIGNING_KEY", None)
    try:
        return subprocess.Popen(  # noqa: S603 — declared command, no shell  # nosec B603
            argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except OSError as exc:
        raise RuntimeError(f"failed to spawn local fx-1 engine {serve_cmd!r}: {exc}") from exc
