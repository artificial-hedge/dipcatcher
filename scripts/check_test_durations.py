#!/usr/bin/env python
"""Verify checked-in ``.test_durations`` covers the pytest-split inventory.

The PR gate and scheduled full suite use the same markers as ``make
test-durations`` (fast ``not network and not slow``, then ``slow and not
network``). Missing node ids skew shard balance under
``--splitting-algorithm least_duration``.

Usage::

    uv run python scripts/check_test_durations.py
    uv run python scripts/check_test_durations.py --durations-path .test_durations
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COLLECT_LINE = re.compile(r"^(tests/.+::.+)$")
# ``pytest --collect-only -q`` file summary lines (matrix inventory format).
FILE_SUMMARY = re.compile(r"^(tests/.+\.py): (\d+)$")

FAST_MARKER = "not network and not slow"
SLOW_MARKER = "slow and not network"


def _pytest_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("HYPOTHESIS_PROFILE", "ci")
    return env


def collect_nodeids(marker: str, *, root: Path) -> set[str]:
    proc = subprocess.run(
        ["uv", "run", "pytest", "--collect-only", "-m", marker],
        cwd=root,
        env=_pytest_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    nodeids: set[str] = set()
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        match = COLLECT_LINE.match(stripped)
        if match:
            nodeids.add(match.group(1))
    return nodeids


def inventory_test_count(marker: str, *, root: Path) -> int:
    proc = subprocess.run(
        ["uv", "run", "pytest", "--collect-only", "-q", "-m", marker],
        cwd=root,
        env=_pytest_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    total = 0
    for line in proc.stdout.splitlines():
        match = FILE_SUMMARY.match(line.strip())
        if match:
            total += int(match.group(2))
    return total


def expected_nodeids(root: Path) -> set[str]:
    return collect_nodeids(FAST_MARKER, root=root) | collect_nodeids(SLOW_MARKER, root=root)


def load_durations(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object, got {type(data).__name__}")
    return {str(k): float(v) for k, v in data.items()}


def check(durations_path: Path, *, root: Path) -> int:
    if not durations_path.is_file():
        print(f"missing durations file: {durations_path}", file=sys.stderr)
        return 1
    durations = load_durations(durations_path)
    collected = expected_nodeids(root)
    missing = sorted(collected - durations.keys())
    inventory = inventory_test_count(FAST_MARKER, root=root) + inventory_test_count(
        SLOW_MARKER, root=root
    )
    if len(collected) != inventory:
        print(
            f"collect inventory mismatch: nodeids={len(collected)} summary_total={inventory}",
            file=sys.stderr,
        )
        return 1
    if missing:
        print(
            f".test_durations stale: {len(missing)} collected tests lack timings "
            f"(have {len(durations)}, need {len(collected)})",
            file=sys.stderr,
        )
        for nodeid in missing[:20]:
            print(f"  missing: {nodeid}", file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more", file=sys.stderr)
        print(
            "Refresh via CI workflow_dispatch (suite=full) job "
            "refresh-test-durations, or `make test-durations` on a large runner.",
            file=sys.stderr,
        )
        return 1
    extra = len(durations) - len(collected)
    if extra:
        print(
            f"note: {extra} duration entries are not in the current collection "
            "(removed or renamed tests; harmless for sharding)",
            file=sys.stderr,
        )
    print(
        f"ok: {len(collected)} collected tests covered by {len(durations)} "
        f"duration entries ({extra} extra historical keys)"
    )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--durations-path",
        type=Path,
        default=REPO_ROOT / ".test_durations",
        help="pytest-split durations JSON (default: repo-root .test_durations)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root (default: parent of scripts/)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return check(args.durations_path.resolve(), root=args.root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
