#!/usr/bin/env python3
"""Rotate and prune app, proxy, and audit logs by size and age.

``--root`` is a directory with ``app``, ``proxy``, and ``audit`` children.
Active files rotate when the next byte would pass the size cap. Rotated
copies older than the age cap, or past the file cap, are deleted. The
hash-chained ledger names ``entries.jsonl`` and ``checkpoints.jsonl`` are
left in place, including their numbered copies.

``--docker-project`` prunes rotated Docker json-file logs for that compose
project. The live ``*-json.log`` is not renamed; the daemon rotates it on
size. Stdout is one JSON object. Symlinks are not followed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

STREAMS = ("app", "proxy", "audit")
SEALED_STEMS = ("entries.jsonl", "checkpoints.jsonl")
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_AGE_S = 7 * 24 * 60 * 60
DEFAULT_MAX_FILES = 5
_ROTATED = re.compile(r"^(.+)\.([1-9][0-9]*)$")
_PROJECT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class Policy:
    max_bytes: int
    max_age_s: int
    max_files: int


@dataclass(frozen=True)
class Item:
    name: str
    size: int
    mtime: float
    symlink: bool


@dataclass(frozen=True)
class Action:
    op: str
    name: str
    reason: str
    dest: str = ""

    def as_dict(self, stream: str) -> dict[str, str]:
        payload = {"name": self.name, "op": self.op, "reason": self.reason, "stream": stream}
        if self.dest:
            payload["dest"] = self.dest
        return payload


def _positive(raw: str, default: int) -> int | None:
    text = raw.strip()
    if not text:
        return default
    if not text.isdecimal():
        return None
    value = int(text)
    if value < 1:
        return None
    return value


def policies_from_env(env: Mapping[str, str]) -> dict[str, Policy] | None:
    """Per-stream caps. Shared ``DIPCATCHER_LOG_*`` fills any unset stream."""
    shared_bytes = _positive(env.get("DIPCATCHER_LOG_MAX_BYTES", ""), DEFAULT_MAX_BYTES)
    shared_age = _positive(env.get("DIPCATCHER_LOG_MAX_AGE_S", ""), DEFAULT_MAX_AGE_S)
    shared_files = _positive(env.get("DIPCATCHER_LOG_MAX_FILES", ""), DEFAULT_MAX_FILES)
    if shared_bytes is None or shared_age is None or shared_files is None:
        return None
    found: dict[str, Policy] = {}
    for stream in STREAMS:
        prefix = f"DIPCATCHER_{stream.upper()}_LOG"
        max_bytes = _positive(env.get(f"{prefix}_MAX_BYTES", ""), shared_bytes)
        max_age = _positive(env.get(f"{prefix}_MAX_AGE_S", ""), shared_age)
        max_files = _positive(env.get(f"{prefix}_MAX_FILES", ""), shared_files)
        if max_bytes is None or max_age is None or max_files is None:
            return None
        found[stream] = Policy(max_bytes, max_age, max_files)
    return found


def sealed_name(name: str) -> bool:
    return any(name == stem or name.startswith(stem + ".") for stem in SEALED_STEMS)


def split_rotated(name: str) -> tuple[str, int | None]:
    if sealed_name(name):
        return name, None
    match = _ROTATED.fullmatch(name)
    if match is None:
        return name, None
    return match.group(1), int(match.group(2))


def _safe_name(name: str) -> bool:
    return bool(name) and name not in {".", ".."} and "/" not in name and "\\" not in name


def plan_directory(items: Sequence[Item], policy: Policy, now: float) -> list[Action]:
    """Size-rotate active files, then drop rotated copies by age and count."""
    actions: list[Action] = []
    live = {item.name: item for item in items}
    for item in items:
        if not _safe_name(item.name):
            actions.append(Action("keep", item.name, "name"))
            continue
        if item.symlink:
            actions.append(Action("keep", item.name, "symlink"))
            continue
        if sealed_name(item.name):
            actions.append(Action("keep", item.name, "sealed"))
    rotating = [
        item
        for item in items
        if _safe_name(item.name)
        and not item.symlink
        and not sealed_name(item.name)
        and split_rotated(item.name)[1] is None
        and item.size > policy.max_bytes
    ]
    for item in rotating:
        actions.extend(_rotation_steps(item.name, policy.max_files, live))
    cutoff = now - policy.max_age_s
    names = set(live)
    for name in sorted(names):
        item = live.get(name)
        if item is None or item.symlink or sealed_name(name) or not _safe_name(name):
            continue
        _active, index = split_rotated(name)
        if index is None:
            continue
        if item.mtime < cutoff:
            actions.append(Action("delete", name, "age"))
            live.pop(name, None)
            continue
        if index > policy.max_files:
            actions.append(Action("delete", name, "count"))
            live.pop(name, None)
    return actions


def _blocked(name: str, max_files: int, live: Mapping[str, Item]) -> bool:
    for index in range(1, max_files + 1):
        item = live.get(f"{name}.{index}")
        if item is not None and (item.symlink or sealed_name(item.name)):
            return True
    return False


def _rotation_steps(name: str, max_files: int, live: dict[str, Item]) -> list[Action]:
    if _blocked(name, max_files, live):
        return [Action("keep", name, "symlink")]
    steps: list[Action] = []
    oldest = f"{name}.{max_files}"
    if oldest in live and not live[oldest].symlink and not sealed_name(oldest):
        steps.append(Action("delete", oldest, "count"))
        live.pop(oldest, None)
    index = max_files - 1
    while index >= 1:
        src = f"{name}.{index}"
        dest = f"{name}.{index + 1}"
        item = live.get(src)
        if item is not None and not item.symlink and not sealed_name(src):
            steps.append(Action("rename", src, "shift", dest))
            live.pop(src, None)
            live[dest] = Item(dest, item.size, item.mtime, False)
        index -= 1
    current = live.get(name)
    if current is None or current.symlink or sealed_name(name):
        return steps
    dest = f"{name}.1"
    steps.append(Action("rename", name, "size", dest))
    live.pop(name, None)
    live[dest] = Item(dest, current.size, current.mtime, False)
    return steps


def plan_rotated_only(items: Sequence[Item], policy: Policy, now: float) -> list[Action]:
    """Prune rotated Docker json logs. The live file stays where the daemon has it."""
    actions: list[Action] = []
    cutoff = now - policy.max_age_s
    for item in items:
        if not _safe_name(item.name) or item.symlink or sealed_name(item.name):
            actions.append(Action("keep", item.name, "skipped"))
            continue
        _active, index = split_rotated(item.name)
        if index is None:
            actions.append(Action("keep", item.name, "active"))
            continue
        if item.mtime < cutoff:
            actions.append(Action("delete", item.name, "age"))
        elif index > policy.max_files:
            actions.append(Action("delete", item.name, "count"))
        else:
            actions.append(Action("keep", item.name, "rotated"))
    return actions


def docker_log_dir_ok(path: Path) -> bool:
    """True for a regular Docker json log under ``containers/<id>/``."""
    if path.is_symlink() or not path.is_file():
        return False
    parent = path.parent
    if parent.name != path.name.removesuffix("-json.log") or len(parent.name) < 12:
        return False
    if not re.fullmatch(r"[0-9a-f]+", parent.name):
        return False
    return parent.parent.name == "containers"


def _scan(directory: Path) -> list[Item] | None:
    if directory.is_symlink() or not directory.is_dir():
        return None
    found: list[Item] = []
    for entry in os.scandir(directory):
        name = entry.name
        if not _safe_name(name):
            continue
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        symlink = entry.is_symlink()
        size = 0 if symlink else info.st_size
        found.append(Item(name, size, info.st_mtime, symlink))
    return found


def _apply(directory: Path, actions: Sequence[Action]) -> str | None:
    root = directory.resolve()
    for action in actions:
        if action.op == "keep":
            continue
        if not _safe_name(action.name):
            return "name"
        src = directory / action.name
        try:
            if src.is_symlink() or not src.resolve().is_relative_to(root):
                return "symlink"
        except OSError:
            return "symlink"
        if action.op == "delete":
            src.unlink()
            continue
        if action.op != "rename":
            return "op"
        dest = directory / action.dest
        if not _safe_name(action.dest) or dest.exists() or dest.is_symlink():
            return "dest"
        if not dest.resolve().is_relative_to(root):
            return "symlink"
        src.replace(dest)
    return None


def retain_tree(root: Path, policies: Mapping[str, Policy], now: float) -> dict[str, object]:
    """Apply directory retention. ``ok`` is false when a stream directory is unsafe."""
    actions: list[dict[str, str]] = []
    for stream in STREAMS:
        directory = root / stream
        if not directory.exists():
            try:
                directory.mkdir(parents=True)
            except OSError:
                return {"actions": actions, "error": f"{stream} directory", "ok": False}
        items = _scan(directory)
        if items is None:
            return {"actions": actions, "error": f"{stream} directory", "ok": False}
        planned = plan_directory(items, policies[stream], now)
        failure = _apply(directory, planned)
        if failure is not None:
            return {"actions": actions, "error": failure, "ok": False, "stream": stream}
        actions.extend(item.as_dict(stream) for item in planned if item.op != "keep")
    return {"actions": actions, "ok": True}


def prune_docker_logs(paths: Sequence[str], policy: Policy, now: float) -> dict[str, object]:
    """Delete aged or extra rotated files next to accepted Docker json logs."""
    actions: list[dict[str, str]] = []
    for raw in paths:
        path = Path(raw)
        if not docker_log_dir_ok(path):
            return {"actions": actions, "error": "docker log path", "ok": False}
        items = _scan(path.parent)
        if items is None:
            return {"actions": actions, "error": "docker log directory", "ok": False}
        planned = plan_rotated_only(items, policy, now)
        failure = _apply(path.parent, planned)
        if failure is not None:
            return {"actions": actions, "error": failure, "ok": False}
        actions.extend(item.as_dict("docker") for item in planned if item.op == "delete")
    return {"actions": actions, "ok": True}


def project_ok(name: str) -> bool:
    return bool(_PROJECT.fullmatch(name))


def _docker_prefix() -> list[str]:
    raw = os.environ.get("DIPCATCHER_DOCKER", "").strip()
    if raw:
        return shlex.split(raw)
    sudo = os.environ.get("SUDO")
    if sudo is None:
        sudo = "sudo"
    sudo = sudo.strip()
    if not sudo:
        return ["docker"]
    return [*shlex.split(sudo), "docker"]


def stream_for_container(name: str) -> str:
    """Proxy containers use the proxy cap. Other containers use the app cap."""
    return "proxy" if "proxy" in name.lower() else "app"


def docker_log_paths(project: str) -> list[tuple[str, str]] | None:
    """Return ``(container, log path)`` pairs, or ``None`` when docker fails."""
    if not project_ok(project):
        return None
    prefix = _docker_prefix()
    listed = subprocess.run(
        [*prefix, "ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if listed.returncode != 0:
        return None
    ids = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    if not ids:
        return []
    inspected = subprocess.run(
        [*prefix, "inspect", "-f", "{{.Name}} {{.LogPath}}", *ids],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if inspected.returncode != 0:
        return None
    found: list[tuple[str, str]] = []
    for line in inspected.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        name, sep, path = text.partition(" ")
        if not sep or not path:
            return None
        found.append((name, path))
    return found


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rotate and prune dipcatcher logs")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--docker-project", default="")
    args = parser.parse_args(argv)
    policies = policies_from_env(os.environ)
    if policies is None:
        _emit({"error": "log limits", "ok": False})
        return 1
    if args.root.is_symlink():
        _emit({"error": "symlink", "ok": False})
        return 1
    report = retain_tree(args.root, policies, time.time())
    project = args.docker_project.strip()
    if project and report.get("ok"):
        if not project_ok(project):
            report = {"actions": report["actions"], "error": "project", "ok": False}
        else:
            pairs = docker_log_paths(project)
            if pairs is None:
                report = {"actions": report["actions"], "error": "docker", "ok": False}
            else:
                actions = list(report["actions"])
                ok = True
                error = ""
                now = time.time()
                for name, path in pairs:
                    pruned = prune_docker_logs([path], policies[stream_for_container(name)], now)
                    actions.extend(pruned.get("actions", []))
                    if not pruned.get("ok"):
                        ok = False
                        error = str(pruned.get("error", "docker log"))
                        break
                report = {"actions": actions, "ok": ok}
                if error:
                    report["error"] = error
    _emit(report)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
