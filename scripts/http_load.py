#!/usr/bin/env python3
"""Async HTTP load for the lab and research APIs. Stdlib only.

One JSON object on stdout: p50, p95, p99, status counts, and the error rate.
Each worker keeps one connection. The API key is read from an env file and
is not printed. ``--seconds`` is capped at 120 and ``--concurrency`` at 200.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
import urllib.parse
from collections import Counter
from collections.abc import Awaitable

_MAX_SECONDS = 120.0
_MAX_CONCURRENCY = 200
_TIMEOUT_S = 5.0
_READ_LIMIT = 65536


def _load_env_file(path: str) -> dict[str, str]:
    """Read KEY=VALUE lines. Does not mutate the process environment."""
    found: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            found.setdefault(key, value)
    return found


def _percentile(samples: list[float], pct: float) -> float | None:
    if not samples:
        return None
    ordered = sorted(samples)
    rank = math.ceil((pct / 100.0) * len(ordered)) - 1
    rank = min(max(rank, 0), len(ordered) - 1)
    return ordered[rank]


def _request_bytes(host: str, path: str, key: str) -> bytes:
    lines = [
        f"GET {path} HTTP/1.1",
        f"Host: {host}",
        "User-Agent: dipcatcher-http-load",
        "Accept: application/json",
        "Connection: keep-alive",
    ]
    if key:
        lines.append(f"X-API-Key: {key}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode()


def _status_code(header: bytes) -> int | None:
    line = header.split(b"\r\n", 1)[0]
    parts = line.split()
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


def _content_length(header: bytes) -> int | None:
    for line in header.split(b"\r\n")[1:]:
        if not line.lower().startswith(b"content-length:"):
            continue
        raw = line.split(b":", 1)[1].strip()
        if raw.isdigit():
            return int(raw)
        return None
    return None


async def _open(host: str, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
    try:
        return await asyncio.wait_for(asyncio.open_connection(host, port), timeout=_TIMEOUT_S)
    except (TimeoutError, OSError):
        return None


async def _read_response(reader: asyncio.StreamReader) -> tuple[int | None, bool]:
    header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=_TIMEOUT_S)
    status = _status_code(header)
    length = _content_length(header)
    if length is None:
        await asyncio.wait_for(reader.read(_READ_LIMIT), timeout=_TIMEOUT_S)
        return status, False
    if length:
        await asyncio.wait_for(reader.readexactly(length), timeout=_TIMEOUT_S)
    reuse = b"connection: close" not in header.lower()
    return status, reuse


async def _send(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, payload: bytes
) -> tuple[int | None, bool] | None:
    try:
        writer.write(payload)
        await writer.drain()
        return await _read_response(reader)
    except (TimeoutError, OSError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
        return None


async def _close(writer: asyncio.StreamWriter | None) -> None:
    if writer is None:
        return
    writer.close()
    try:
        await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
    except (TimeoutError, OSError):
        return


async def _worker(
    host: str,
    port: int,
    payload: bytes,
    deadline: float,
    statuses: Counter[str],
    latencies: list[float],
) -> None:
    reader: asyncio.StreamReader | None = None
    writer: asyncio.StreamWriter | None = None
    while time.monotonic() < deadline:
        if writer is None or reader is None:
            opened = await _open(host, port)
            if opened is None:
                statuses["transport"] += 1
                continue
            reader, writer = opened
        started = time.perf_counter()
        outcome = await _send(reader, writer, payload)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if outcome is None or outcome[0] is None:
            statuses["transport"] += 1
            await _close(writer)
            reader = writer = None
            continue
        status, reuse = outcome
        statuses[str(status)] += 1
        latencies.append(elapsed_ms)
        if not reuse:
            await _close(writer)
            reader = writer = None
    await _close(writer)


async def _run(
    host: str, port: int, payload: bytes, concurrency: int, seconds: float
) -> dict[str, object]:
    statuses: Counter[str] = Counter()
    latencies: list[float] = []
    deadline = time.monotonic() + seconds
    workers: list[Awaitable[None]] = [
        _worker(host, port, payload, deadline, statuses, latencies) for _ in range(concurrency)
    ]
    await asyncio.gather(*workers)
    attempts = sum(statuses.values())
    http_5xx = sum(count for code, count in statuses.items() if code.startswith("5"))
    ok = sum(count for code, count in statuses.items() if code.startswith("2"))
    errors = attempts - ok
    return {
        "concurrency": concurrency,
        "seconds": seconds,
        "requests": attempts,
        "ok": ok,
        "errors": errors,
        "error_rate": (errors / attempts) if attempts else 1.0,
        "http_429": statuses.get("429", 0),
        "http_5xx": http_5xx,
        "status": dict(sorted(statuses.items())),
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "p99_ms": _percentile(latencies, 99),
        "latency_sample": "completed_http_ms",
    }


def _parse_url(url: str) -> tuple[str, int, str]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise SystemExit("only http URLs are supported")
    port = parsed.port or 80
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return parsed.hostname, port, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Async HTTP load")
    parser.add_argument("--url", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--env-file")
    parser.add_argument("--key-env")
    args = parser.parse_args(argv)
    if args.concurrency < 1 or args.concurrency > _MAX_CONCURRENCY:
        raise SystemExit(f"concurrency must be 1..{_MAX_CONCURRENCY}")
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.seconds > _MAX_SECONDS:
        raise SystemExit(f"seconds must be in (0, {_MAX_SECONDS}]")
    file_env = _load_env_file(args.env_file) if args.env_file else {}
    key = ""
    if args.key_env:
        key = os.environ.get(args.key_env) or file_env.get(args.key_env, "")
    host, port, path = _parse_url(args.url)
    payload = _request_bytes(host, path, key)
    report = asyncio.run(_run(host, port, payload, args.concurrency, args.seconds))
    report["path"] = path
    report["port"] = port
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
