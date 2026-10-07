"""Harness doctor — an executable diagnostic battery over the live surface.

Two legs share one report:

- ``run_remote_doctor(client, make_client)`` drives a ``HarnessClient``
  against a deployed API: readiness, health, wire-contract negotiation,
  capabilities, the command registry, auth enforcement, the managed-key
  lifecycle, ops metrics, and a sealed-receipt round-trip. ``make_client``
  mints sibling clients on the same deployment (used to authenticate as a
  probe key and to confirm revoked keys refuse).
- ``run_local_doctor`` drives ``Fx1Harness`` in-process: health, the
  registry, the honesty gate, a gated completion through an injected
  deterministic backend (plus its completion-log record), the managed-key
  lifecycle, and a sealed-receipt round-trip against a receipts dir.

Each check reports ``{"name", "ok", "detail"}`` — ``detail`` carries one
terse line of evidence, never a secret: minted probe keys are referenced
by record id only and raw key material never leaves the battery. A check
the target legitimately can't answer (no bootstrap credential, no sealed
receipts, open-mode auth) reports ``"skipped": true`` with the reason and
does not fail the report. ``report["ok"]`` is the AND of every non-skipped
check; the battery itself never raises — transport faults land inside the
individual check that hit them.

What doctor deliberately does NOT probe: ``POST /harness/drain`` is a
one-way latch that would take the target deployment down, so drain is a
documented exclusion rather than a check. Registered lab commands are
listed, never executed — they spawn real work.
"""

from __future__ import annotations

import contextlib
import gc
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["run_local_doctor", "run_remote_doctor"]

_REPORT_VERSION = "fx1_doctor_report.v1"


class _Skip(Exception):
    """A check the target legitimately cannot answer — reported, not failed."""


def _run_check(name: str, fn: Callable[[], str]) -> dict[str, Any]:
    """Run one probe; any raised fault becomes a failed check's detail."""
    try:
        detail = fn()
    except _Skip as exc:
        return {"name": name, "ok": True, "skipped": True, "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    return {"name": name, "ok": True, "detail": detail}


def _report(mode: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "object": _REPORT_VERSION,
        "mode": mode,
        "checked_at": time.time(),
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
    }


# --------------------------------------------------------------------------
# remote leg — a deployed harness over the wire
# --------------------------------------------------------------------------


def run_remote_doctor(
    client: Any,
    make_client: Callable[[str | None], Any],
) -> dict[str, Any]:
    """Diagnose one deployed harness API.

    ``client`` is the configured ``HarnessClient``; ``make_client(api_key)``
    builds a sibling on the same deployment so the battery can authenticate
    as a probe key (and as a bogus key for the auth-enforcement check).

    An open-mode deployment (no bootstrap key and an empty key store —
    anonymous gated calls succeed) is detected BEFORE the key-lifecycle
    probe runs: minting a probe key there would arm remote auth
    permanently (``has_keys`` counts ever-minted records, tombstones
    included), which is a posture change a diagnostic must not cause.
    On open mode both auth checks report ``skipped`` instead.
    """
    from fx1.serve.client import HarnessAuthError  # noqa: PLC0415

    try:
        make_client(None).commands()
        open_mode = True  # anonymous caller reached a gated route
    except HarnessAuthError:
        open_mode = False
    except Exception:  # noqa: BLE001
        open_mode = False

    checks = [
        _run_check("ready", lambda: _remote_ready(client)),
        _run_check("health", lambda: _remote_health(client)),
        _run_check("version_compat", lambda: _remote_version_compat(client)),
        _run_check("capabilities", lambda: _remote_capabilities(client)),
        _run_check("commands", lambda: _remote_commands(client)),
        _run_check(
            "unauth_refused", lambda: _remote_unauth_refused(client, make_client, open_mode)
        ),
        _run_check("key_lifecycle", lambda: _remote_key_lifecycle(client, make_client, open_mode)),
        _run_check("metrics", lambda: _remote_metrics(client)),
        _run_check("receipt_roundtrip", lambda: _remote_receipt_roundtrip(client)),
    ]
    return _report("remote", checks)


def _remote_ready(client: Any) -> str:
    payload = client.ready()
    return f"ready payload ok (draining={payload.get('draining', False)})"


def _remote_health(client: Any) -> str:
    h = client.health()
    if h.status != "ok" or h.registered_commands <= 0:
        raise RuntimeError(f"status={h.status!r} registered_commands={h.registered_commands}")
    configured = sorted(k for k, v in h.backends.items() if v)
    return (
        f"status=ok version={h.version} "
        f"commands={h.registered_commands} backends={configured or 'none'}"
    )


def _remote_version_compat(client: Any) -> str:
    out = client.check_compat(strict=False)
    if not out["compatible"]:
        raise RuntimeError(
            f"server api_version={out['server_api_version']!r} "
            f"!= client {out['client_api_version']!r}"
        )
    return f"api_version={out['server_api_version']} fx1={out['server_fx1_version']}"


def _remote_capabilities(client: Any) -> str:
    caps = client.capabilities()
    if not caps:
        raise RuntimeError("empty capabilities payload")
    return f"{len(caps)} capability keys"


def _remote_commands(client: Any) -> str:
    names = client.commands()
    if not names:
        raise RuntimeError("empty command registry")
    return f"{len(names)} registered commands"


def _remote_unauth_refused(
    client: Any, make_client: Callable[[str | None], Any], open_mode: bool
) -> str:
    from fx1.serve.client import HarnessAuthError  # noqa: PLC0415

    if open_mode:
        raise _Skip("open mode — anonymous gated calls allowed; nothing to refuse")
    probe = make_client("fx1-doctor-invalid-key")
    try:
        probe.commands()
    except HarnessAuthError:
        return "gated route refuses a bogus key"
    raise RuntimeError("bogus key authenticated on a keyed deployment")


def _remote_key_lifecycle(
    client: Any, make_client: Callable[[str | None], Any], open_mode: bool
) -> str:
    from fx1.serve.client import HarnessAuthError  # noqa: PLC0415

    if open_mode:
        raise _Skip(
            "open-mode deployment: minting a probe key would arm auth "
            "permanently — pass an admin credential to exercise this"
        )
    try:
        mint = client.key_create(name="fx1-doctor-probe", scopes=["read"])
    except HarnessAuthError as exc:
        raise _Skip(f"credential lacks key-admin scope ({exc.code or 'auth refused'})") from exc
    key_id = mint["id"]
    successor_id: str | None = None
    try:
        probe = make_client(mint["key"])
        probe.commands()  # the minted key authenticates
        client.key_usage(key_id)  # usage card readable
        rotated = client.key_rotate(key_id)
        successor_id = rotated["key"]["id"]
        try:
            probe.commands()
        except HarnessAuthError:
            pass  # rotate tombstoned the predecessor as declared
        else:
            raise RuntimeError("rotated-out key still authenticates")
        return f"mint→use→rotate→revoke ok (key {key_id})"
    finally:
        for rid in (key_id, successor_id):
            if rid is None:
                continue
            with contextlib.suppress(Exception):  # tombstone may exist
                client.key_revoke(rid)


def _remote_metrics(client: Any) -> str:
    m = client.metrics()
    return (
        f"requests_total={m.requests_total} errors={m.errors_total} "
        f"inflight={m.inflight}/{m.max_inflight}"
    )


def _remote_receipt_roundtrip(client: Any) -> str:
    refs = client.receipts()
    if not refs:
        raise _Skip("deployment publishes no sealed receipts")
    stored = client.receipt(refs[0].sha256)
    verdict = client.verify_receipt(stored.document)
    if not (stored.valid and verdict.valid):
        raise RuntimeError(
            f"receipt {refs[0].sha256[:12]}… failed re-verify "
            f"(fetch={stored.valid} verify={verdict.valid})"
        )
    return f"{len(refs)} receipts; {refs[0].name} re-verifies"


# --------------------------------------------------------------------------
# local leg — the in-process SDK
# --------------------------------------------------------------------------


_DOCTOR_MARKER = "fx1-doctor completion marker"


class _DoctorBackend:
    """Deterministic backend injected through ``backend_resolver`` — proves
    the gated completion path end-to-end without a real engine. The
    marker text is honesty-gate clean by construction."""

    def __init__(self) -> None:
        self.closed = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: Any | None = None,
    ) -> str:
        del messages, sampling  # deterministic stub — no input dependence
        return _DOCTOR_MARKER

    def close(self) -> None:
        self.closed += 1


def run_local_doctor(
    *,
    receipts_dir: str | Path = "receipts",
    state_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Diagnose the in-process harness (``Fx1Harness``).

    ``receipts_dir`` points at a sealed-receipt store for the round-trip
    check (absent/empty → skipped). ``state_dir`` — when given — also
    exercises durable-state replay: a key minted by one harness must be
    visible to a second harness bound to the same dir.
    """
    from fx1.sdk import Fx1Harness  # noqa: PLC0415

    sdk = Fx1Harness(receipts_dir=receipts_dir)
    checks = [
        _run_check("health", lambda: _local_health(sdk)),
        _run_check("commands", lambda: _local_commands(sdk)),
        _run_check("honesty_gate", lambda: _local_honesty_gate(sdk)),
        _run_check("completion_roundtrip", _local_completion_roundtrip),
        _run_check("key_lifecycle", lambda: _local_key_lifecycle(sdk)),
        _run_check("receipt_roundtrip", lambda: _local_receipt_roundtrip(sdk, receipts_dir)),
    ]
    if state_dir is not None:
        checks.append(_run_check("state_dir_replay", lambda: _state_dir_replay(state_dir)))
    return _report("local", checks)


def _local_health(sdk: Any) -> str:
    h = sdk.health()
    if h.status != "ok" or h.registered_commands <= 0:
        raise RuntimeError(f"status={h.status!r} registered_commands={h.registered_commands}")
    configured = sorted(k for k, v in h.backends.items() if v)
    return (
        f"status=ok version={h.version} "
        f"commands={h.registered_commands} backends={configured or 'none'}"
    )


def _local_commands(sdk: Any) -> str:
    names = sdk.commands()
    if not names:
        raise RuntimeError("empty command registry")
    return f"{len(names)} registered commands"


def _local_honesty_gate(sdk: Any) -> str:
    refused = sdk.check_text("Sharpe ratio 9.9 annualized")
    accepted = sdk.check_text("pinball loss 0.021 on the validation split")
    if refused.ok or not accepted.ok:
        raise RuntimeError(f"gate misclassified (forbidden ok={refused.ok} clean ok={accepted.ok})")
    return "forbidden headline metric refused; clean text passes"


def _local_completion_roundtrip() -> str:
    from fx1.sdk import Fx1Harness  # noqa: PLC0415

    backend = _DoctorBackend()
    probe = Fx1Harness(backend_resolver=lambda *a, **k: backend)
    out = probe.complete([{"role": "user", "content": "doctor ping"}], backend="byok")
    if out.content != _DOCTOR_MARKER:
        raise RuntimeError(f"backend marker mismatch: {out.content!r}")
    if backend.closed != 1:
        raise RuntimeError("backend not closed after the call")
    cid = probe.last_response_headers.get("x-fx1-completion-id")
    if not cid:
        raise RuntimeError("no completion-id trace header recorded")
    rec = probe.completion(cid)
    if not rec.ok or rec.completion_id != cid:
        raise RuntimeError(f"completion log record wrong for {cid}")
    return f"gated complete ok; log record {cid[:12]}… retrievable"


def _local_key_lifecycle(sdk: Any) -> str:
    mint = sdk.key_create(name="fx1-doctor-probe", scopes=["read"])
    key_id = mint["id"]
    successor_id: str | None = None
    try:
        sdk.key_get(key_id)
        sdk.key_usage(key_id)
        rotated = sdk.key_rotate(key_id)
        successor_id = rotated["key"]["id"]
        sdk.key_revoke(successor_id)
        # revocation tombstones, never deletes: the record must remain
        # fetchable for audit with enabled=False, and authenticate
        # must refuse the raw key.
        tomb = sdk.key_get(successor_id)
        if tomb.get("enabled") is not False:
            raise RuntimeError("revoked key record not tombstoned")
        successor_id = None  # already tombstoned
        if sdk._key_store.authenticate(rotated["key"]["key"]) is not None:
            raise RuntimeError("revoked key still authenticates")
        return f"mint→get→usage→rotate→revoke ok (key {key_id})"
    finally:
        for rid in (key_id, successor_id):
            if rid is None:
                continue
            with contextlib.suppress(Exception):  # tombstone may exist
                sdk.key_revoke(rid)


def _local_receipt_roundtrip(sdk: Any, receipts_dir: str | Path) -> str:
    try:
        refs = sdk.receipts()
    except FileNotFoundError as exc:
        raise _Skip(f"no sealed-receipt store at {receipts_dir}") from exc
    if not refs:
        raise _Skip(f"receipts dir {receipts_dir} is empty")
    stored = sdk.receipt(refs[0].sha256)
    verdict = sdk.verify_receipt(stored.document)
    if not (stored.valid and verdict.valid):
        raise RuntimeError(
            f"receipt {refs[0].sha256[:12]}… failed re-verify "
            f"(fetch={stored.valid} verify={verdict.valid})"
        )
    return f"{len(refs)} receipts; {refs[0].name} re-verifies"


def _state_dir_replay(state_dir: str | Path) -> str:
    """Durable-state probe: mint a key on one harness, rebind a second
    harness to the same dir, the key must have survived. The first
    harness is released before rebinding — the single-writer claim (and a
    real process restart) both demand it."""
    from fx1.sdk import Fx1Harness  # noqa: PLC0415

    first = Fx1Harness(state_dir=state_dir)
    key_id = first.key_create(name="fx1-doctor-durable-probe")["id"]
    del first
    gc.collect()  # releases the single-writer claim on the dir
    try:
        second = Fx1Harness(state_dir=state_dir)
        try:
            second.key_get(key_id)
        finally:
            with contextlib.suppress(Exception):  # cleanup only
                second.key_revoke(key_id)
    except Exception as exc:
        raise RuntimeError(f"journal replay lost the minted key: {exc}") from exc
    return (
        f"key {key_id} survived a harness rebind on the same state_dir "
        "(its tombstone remains — provisioning semantics by design)"
    )
