#!/usr/bin/env python3
"""Overlap API, broker, and webhook HMAC secrets, then retire the previous token.

The API and webhook files keep their existing paths. Line 1 is the token a
restarted proxy or Prometheus loads. Line 2 is the previous token. The API
reads the file on each request, so both tokens work without a restart.
Broker overlap uses the existing ``ALPACA_*`` pair as current and ``APCA_*``
as previous. Values are not printed and are not taken from argv.

``drill`` uses a temporary directory and an in-process listener. It does not
read the operator's secret files.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from quant_fund.api.webhook_hmac import sign_alert_webhook, verify_alert_webhook
from quant_fund.live.brokers import broker_overlap_pairs
from quant_fund.utils.secret_tokens import auth_decision, parse_overlap, read_secret_text

_NAMES = (
    ("QUANT_API_KEY", "DIPCATCHER_QUANT_API_KEY_FILE", "quant_api_key"),
    ("RESEARCH_API_KEY", "DIPCATCHER_RESEARCH_API_KEY_FILE", "research_api_key"),
    (
        "DIPCATCHER_ALERT_WEBHOOK_SECRET",
        "DIPCATCHER_ALERT_WEBHOOK_SECRET_FILE",
        "alert_webhook_secret",
    ),
)
_BROKER = (
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
)


def overlap_body(current: str, incoming: str) -> str | None:
    """``current`` on line 1 and ``incoming`` on line 2. ``None`` when either is unusable."""
    if parse_overlap(current) != (current,) or parse_overlap(incoming) != (incoming,):
        return None
    if current == incoming:
        return None
    return current + "\n" + incoming + "\n"


def promote_body(text: str) -> str | None:
    """Swap the two lines so a restarted process loads the incoming token."""
    tokens = parse_overlap(text)
    if tokens is None or len(tokens) != 2:
        return None
    return tokens[1] + "\n" + tokens[0] + "\n"


def retire_body(text: str) -> str | None:
    """Keep line 1 and drop the previous token."""
    tokens = parse_overlap(text)
    if tokens is None:
        return None
    return tokens[0] + "\n"


def write_atomic(path: Path, text: str) -> bool:
    """Replace ``path`` without a symlink and without leaving a partial file."""
    if path.is_symlink():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(temporary, flags, 0o644)
    except OSError:
        return False
    try:
        os.write(descriptor, text.encode("utf-8"))
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
    except OSError:
        os.close(descriptor)
        return False
    os.close(descriptor)
    try:
        os.replace(temporary, path)
    except OSError:
        return False
    return True


def _secret_path(file_var: str, leaf: str) -> Path:
    raw = (
        os.environ.get(file_var, "").strip()
        or os.environ.get(file_var.removeprefix("DIPCATCHER_"), "").strip()
    )
    if raw:
        return Path(raw)
    return Path("/tmp/dipcatcher-ops7-secrets") / leaf


def _apply_phase_body(phase: str) -> dict[str, object]:
    changed: list[str] = []
    for name, file_var, leaf in _NAMES:
        path = _secret_path(file_var, leaf)
        incoming = os.environ.get(name, "").strip()
        if not _phase_applies(phase, name, path, incoming):
            continue
        if phase == "overlap":
            current_text = read_secret_text(str(path))
            current = "" if current_text is None else current_text.split("\n", 1)[0]
            body = overlap_body(current, incoming)
        else:
            current_text = read_secret_text(str(path))
            if current_text is None:
                body = None
            elif phase == "promote":
                body = promote_body(current_text)
            else:
                body = retire_body(current_text)
        if body is None:
            return {"ok": False, "error": name, "changed": changed}
        if not write_atomic(path, body):
            return {"ok": False, "error": "write", "changed": changed}
        changed.append(name)
    pairs = _broker_count()
    return {"ok": True, "changed": changed, "broker_pairs": pairs}


def _phase_applies(phase: str, name: str, path: Path, incoming: str) -> bool:
    """API names are required. The webhook file is required only when it exists."""
    required = name != "DIPCATCHER_ALERT_WEBHOOK_SECRET"
    if phase == "overlap":
        if incoming:
            return True
        if required or path.is_file():
            raise _PhaseSkip(name)
        return False
    if path.is_file():
        return True
    if required:
        raise _PhaseSkip(name)
    return False


class _PhaseSkip(Exception):
    """The named secret could not be rotated. The message is the env name."""


def _apply_phase(phase: str) -> dict[str, object]:
    try:
        return _apply_phase_body(phase)
    except _PhaseSkip as exc:
        return {"ok": False, "error": str(exc), "changed": []}


def _broker_count() -> int:
    try:
        return len(broker_overlap_pairs())
    except Exception:  # noqa: BLE001 — the drill reports the count, not the secret error text
        return -1


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        presented = self.headers.get("X-API-Key")
        code = 200 if auth_decision("QUANT_API_KEY", presented) == "ok" else 401
        self.send_response(code)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def _status(url: str, key: str) -> int:
    request = urllib.request.Request(url, headers={"X-API-Key": key})
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return int(response.status)
    except Exception as exc:  # noqa: BLE001 — HTTPError carries the status
        code = getattr(exc, "code", None)
        return int(code) if isinstance(code, int) else 599


def _hammer(url: str, key: str, stop: threading.Event, seen: list[int]) -> None:
    while not stop.is_set():
        seen.append(_status(url, key))


def _webhook_window(path: Path, old: str, new: str) -> tuple[bool, bool, bool, bool]:
    """Overlap accepts both signatures. After promote and retire, only the new one does."""
    from quant_fund.utils.secret_tokens import accepted_tokens  # noqa: PLC0415

    now = time.time()
    hook_overlap = overlap_body(old, new) or ""
    write_atomic(path, hook_overlap)
    os.environ["DIPCATCHER_ALERT_WEBHOOK_SECRET_FILE"] = str(path)
    tokens = accepted_tokens("DIPCATCHER_ALERT_WEBHOOK_SECRET")
    during_old = _hmac_ok(tokens, old, now)
    during_new = _hmac_ok(tokens, new, now)
    promoted = promote_body(hook_overlap) or ""
    write_atomic(path, retire_body(promoted) or "")
    retired = accepted_tokens("DIPCATCHER_ALERT_WEBHOOK_SECRET")
    return (
        during_old,
        during_new,
        _hmac_ok(retired, old, now),
        _hmac_ok(retired, new, now),
    )


def _hmac_ok(tokens: tuple[str, ...], secret: str, now: float) -> bool:
    body = b"{}"
    stamp = str(int(now))
    signature = sign_alert_webhook(secret, stamp, body)
    return verify_alert_webhook(tokens, stamp, signature, body, now=now)


def run_drill() -> dict[str, object]:
    """Prove both tokens work through the file rewrite, then only the new token."""
    root = Path("/tmp") / f"dipcatcher-ops13-rotate-{secrets.token_hex(4)}"
    root.mkdir(parents=True, mode=0o700)
    old = "old-" + secrets.token_hex(8)
    new = "new-" + secrets.token_hex(8)
    old_broker = ("brk-" + secrets.token_hex(6), "sec-" + secrets.token_hex(6))
    new_broker = ("brk-" + secrets.token_hex(6), "sec-" + secrets.token_hex(6))
    saved = {
        name: os.environ.get(name) for name in ("QUANT_API_KEY", "QUANT_API_KEY_FILE", *_BROKER)
    }
    path = root / "quant_api_key"
    hook = root / "alert_webhook_secret"
    gap = False
    try:
        write_atomic(path, old + "\n")
        write_atomic(hook, old + "\n")
        os.environ["QUANT_API_KEY_FILE"] = str(path)
        os.environ.pop("QUANT_API_KEY", None)
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        port = int(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{port}/health"
        before = _status(url, old)
        stop = threading.Event()
        seen: list[int] = []
        worker = threading.Thread(target=_hammer, args=(url, old, stop, seen), daemon=True)
        worker.start()
        wrote = write_atomic(path, overlap_body(old, new) or "")
        time.sleep(0.05)
        during_old = _status(url, old)
        during_new = _status(url, new)
        during_other = _status(url, "other-" + secrets.token_hex(4))
        stop.set()
        worker.join(timeout=2)
        server.shutdown()
        gap = (not seen) or any(code != 200 for code in seen)
        promoted = promote_body(read_secret_text(str(path)) or "")
        write_atomic(path, promoted or "")
        still_old = auth_decision("QUANT_API_KEY", old)
        still_new = auth_decision("QUANT_API_KEY", new)
        write_atomic(path, retire_body(promoted or "") or "")
        after_old = auth_decision("QUANT_API_KEY", old)
        after_new = auth_decision("QUANT_API_KEY", new)
        hook_old, hook_new, hook_old_after, hook_new_after = _webhook_window(hook, old, new)
        os.environ["ALPACA_API_KEY_ID"] = new_broker[0]
        os.environ["ALPACA_API_SECRET_KEY"] = new_broker[1]
        os.environ["APCA_API_KEY_ID"] = old_broker[0]
        os.environ["APCA_API_SECRET_KEY"] = old_broker[1]
        during_pairs = len(broker_overlap_pairs())
        os.environ.pop("APCA_API_KEY_ID", None)
        os.environ.pop("APCA_API_SECRET_KEY", None)
        after_pairs = len(broker_overlap_pairs())
        ok = (
            wrote
            and before == 200
            and during_old == 200
            and during_new == 200
            and during_other == 401
            and not gap
            and still_old == "ok"
            and still_new == "ok"
            and after_old == "reject"
            and after_new == "ok"
            and hook_old
            and hook_new
            and not hook_old_after
            and hook_new_after
            and during_pairs == 2
            and after_pairs == 1
        )
        return {
            "api_new_after_retire": after_new == "ok",
            "api_new_during_overlap": during_new == 200,
            "api_old_after_retire": after_old == "reject",
            "api_old_during_overlap": during_old == 200,
            "broker_pairs_after_retire": after_pairs,
            "broker_pairs_during_overlap": during_pairs,
            "gap": gap,
            "ok": ok,
            "requests_during_write": len(seen),
            "webhook_new_after_retire": hook_new_after,
            "webhook_new_during_overlap": hook_new,
            "webhook_old_after_retire": hook_old_after,
            "webhook_old_during_overlap": hook_old,
        }
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        os.environ.pop("DIPCATCHER_ALERT_WEBHOOK_SECRET_FILE", None)


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "drill":
        report = run_drill()
        sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
        return 0 if report.get("ok") else 1
    if len(argv) == 2 and argv[1] in {"overlap", "promote", "retire"}:
        report = _apply_phase(argv[1])
        sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
        return 0 if report.get("ok") else 1
    sys.stderr.write("usage: prod_rotate.py overlap|promote|retire|drill\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
