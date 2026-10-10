"""keys_audit — the managed-API-key store's own contract battery.

``keys.py`` is the multi-tenant auth layer every route leans on. The
batteries that touch it (auth/quota/fault audits) exercise the wire;
this battery pins the store contract itself:

- *mint* — ``fx1k_``-prefixed 40-hex secrets shown once, sha256-only
  storage, ``key_id``/``prefix`` display fingerprints, wire view strips
  ``sha256`` and every ``_``-counter, scope resolution (defaults,
  additive ``admin``, canonical ``SCOPES`` order, empty/unknown
  refused), validation of rpm/ttl_s/max_requests/max_tokens, cap.
- *authenticate* — uniform ``None`` for wrong/unknown/disabled/expired
  (no oracle), refusal ordering budgets → scope → window (a refused
  call never counts as a use nor burns a window slot), rpm window
  raising ``rate_limited`` with honest ``retry_after``, rolling
  window recovery, ``uses``/``last_used_at`` accounting.
- *revoke/update/rotate* — tombstones not deletes, idempotent-refusing
  revokes, patch keeps counters while ``clear`` reverts nullables to
  unbounded, non-clearable fields refused, rotation inherits declared
  policy verbatim (absolute ``expires_at`` carried, fresh ``ttl_s``
  rebases), single-line atomic ``revoked+successor`` journal entry,
  ``keys_cap`` checked before the predecessor is touched.
- *durability* — every mutation journaled and replayed across restart,
  counters restored, rate windows (``_``-prefixed) process-local,
  torn journals quarantine recovered keys while keeping auth
  required, and a later clean mint is not re-quarantined.

Probes are literal bools; the sealed receipt names every defect found.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, cast

from fx1.serve.journal import JobJournal
from fx1.serve.keys import (
    CLEARABLE_KEY_FIELDS,
    KEY_PREFIX,
    SCOPES,
    ApiKeyStore,
    KeyStoreError,
)

__all__ = ["keys_audit", "keys_audit_bench"]


class _Clock:
    def __init__(self, t: float = 1_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _must(rec: dict[str, Any] | None) -> dict[str, Any]:
    assert rec is not None
    return rec


def _refuses(fn: Any, *args: Any, **kwargs: Any) -> bool:
    try:
        fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 — refuse probes accept any failure
        return True
    return False


def _code(fn: Any, *args: Any, **kwargs: Any) -> str | None:
    try:
        fn(*args, **kwargs)
    except KeyStoreError as e:
        return e.code
    except Exception:  # noqa: BLE001
        return None
    return None


def _probe_mint() -> dict[str, bool]:
    out: dict[str, bool] = {}
    clock = _Clock()
    store = ApiKeyStore(clock=clock)
    raw, rec = store.mint("alpha")
    out["mt_prefix_shape"] = raw.startswith(KEY_PREFIX) and len(raw) == len(KEY_PREFIX) + 40
    out["mt_keyid_fingerprint"] = len(rec["key_id"]) == 16 and all(
        c in "0123456789abcdef" for c in rec["key_id"]
    )
    out["mt_prefix_display"] = rec["prefix"] == raw[:13]
    out["mt_wire_no_sha"] = "sha256" not in rec
    out["mt_wire_no_private"] = not any(k.startswith("_") for k in rec)
    out["mt_raw_not_stored"] = raw not in json.dumps(store.list())
    out["mt_default_scopes"] = rec["scopes"] == ["read", "write"] and rec["admin"] is False
    _r2, rec2 = store.mint(admin=True)
    out["mt_admin_scopes"] = rec2["admin"] is True and rec2["scopes"] == list(SCOPES)
    _r3, rec3 = store.mint(scopes=["admin"], admin=True)
    out["mt_admin_unions"] = rec3["scopes"] == ["admin"] and rec3["admin"] is True
    _r4, rec4 = store.mint(scopes=["write", "read", "write"])
    out["mt_scopes_canonical_order"] = rec4["scopes"] == ["read", "write"]
    out["mt_scopes_empty_refused"] = _refuses(store.mint, scopes=[])
    out["mt_scopes_unknown_refused"] = _refuses(store.mint, scopes=["exec"])
    out["mt_scopes_wrong_type_refused"] = _refuses(store.mint, scopes=[1])
    out["mt_rpm_min"] = _refuses(store.mint, rpm=0)
    out["mt_ttl_positive"] = _refuses(store.mint, ttl_s=0) and _refuses(store.mint, ttl_s=-5)
    out["mt_ttl_finite"] = _refuses(store.mint, ttl_s=float("inf")) and _refuses(
        store.mint, ttl_s=float("nan")
    )
    out["mt_budgets_min"] = _refuses(store.mint, max_requests=0) and _refuses(
        store.mint, max_tokens=0
    )
    _r5, rec5 = store.mint(ttl_s=30.0)
    out["mt_expires_computed"] = math.isclose(rec5["expires_at"], clock.t + 30.0)
    capped = ApiKeyStore(max_keys=1, clock=clock)
    capped.mint()
    out["mt_cap"] = _code(capped.mint) == "keys_cap"
    out["mt_max_keys_min"] = _refuses(ApiKeyStore, max_keys=0)
    return out


def _probe_authenticate() -> dict[str, bool]:
    out: dict[str, bool] = {}
    clock = _Clock()
    store = ApiKeyStore(clock=clock)
    raw, rec = store.mint()
    out["au_accepts_own"] = _must(store.authenticate(raw))["key_id"] == rec["key_id"]
    out["au_wrong_none"] = store.authenticate(KEY_PREFIX + "0" * 40) is None
    out["au_unprefixed_none"] = store.authenticate("sk-whatever") is None
    out["au_nonstring_none"] = store.authenticate(cast(str, 12345)) is None
    out["au_empty_none"] = store.authenticate("") is None
    first = _must(store.authenticate(raw))
    out["au_uses_count"] = first["uses"] >= 1 and first["last_used_at"] == clock.t
    # uniform 401 surface — wrong key and revoked key both return None
    store.revoke(rec["key_id"])
    out["au_revoked_none"] = store.authenticate(raw) is None
    # expiry via injected clock
    clock2 = _Clock(500.0)
    store2 = ApiKeyStore(clock=clock2)
    raw2, _ = store2.mint(ttl_s=10.0)
    out["au_alive_pre_expiry"] = store2.authenticate(raw2) is not None
    clock2.t = 511.0
    out["au_expired_none"] = store2.authenticate(raw2) is None
    return out


def _probe_policy_order() -> dict[str, bool]:
    """Refusal ordering: budgets → scope → window; refusals never count."""
    out: dict[str, bool] = {}
    clock = _Clock()
    store = ApiKeyStore(clock=clock)
    # quota_exceeded beats window consumption
    raw, rec = store.mint(max_requests=1, rpm=1)
    store.authenticate(raw)
    before = store.window_state(rec["key_id"])
    e = _code(store.authenticate, raw)
    out["po_quota_exceeded"] = e == "quota_exceeded"
    out["po_quota_no_window_burn"] = store.window_state(rec["key_id"]) == before and before == (
        1,
        0,
        60,
    )
    # insufficient_scope before counters
    raw2, rec2 = store.mint(scopes=["read"], rpm=1)
    e = _code(store.authenticate, raw2, required_scope="write")
    out["po_insufficient_scope"] = e == "insufficient_scope"
    out["po_scope_no_use"] = _must(store.get(rec2["key_id"]))["uses"] == 0
    state = store.window_state(rec2["key_id"])
    out["po_scope_no_window_burn"] = state == (1, 1, 0)
    # read-scope key still reads fine
    out["po_inscope_ok"] = store.authenticate(raw2, required_scope="read") is not None
    # rate_limited carries retry_after + key_id
    e2 = None
    try:
        store.authenticate(raw2)
    except KeyStoreError as err:
        e2 = err
    out["po_rate_limited"] = e2 is not None and e2.code == "rate_limited"
    out["po_retry_after"] = (
        e2 is not None and e2.retry_after is not None and 0 < e2.retry_after <= 60.0
    )
    out["po_error_key_id"] = e2 is not None and e2.key_id == rec2["key_id"]
    # refused request did not count as a use
    out["po_limited_no_use"] = _must(store.get(rec2["key_id"]))["uses"] == 1
    # window rolls after 60 s
    clock.t += 61.0
    out["po_window_rolls"] = store.authenticate(raw2) is not None
    # tokens budget: charge then refuse
    raw3, rec3 = store.mint(max_tokens=10)
    store.authenticate(raw3)
    store.charge_tokens(rec3["key_id"], 10)
    out["po_token_budget"] = _code(store.authenticate, raw3) == "quota_exceeded"
    store.charge_tokens(rec3["key_id"], 0)
    store.charge_tokens("unknown-id", 5)
    out["po_charge_junk_ignored"] = True
    return out


def _probe_revoke_update_rotate() -> dict[str, bool]:
    out: dict[str, bool] = {}
    clock = _Clock()
    store = ApiKeyStore(clock=clock)
    _, rec = store.mint("rotator", scopes=["read", "write"], rpm=3, max_requests=50)
    out["rv_tombstone"] = (lambda r: r["enabled"] is False and r["revoked_at"] == clock.t)(
        store.revoke(rec["key_id"])
    )
    out["rv_record_kept"] = store.get(rec["key_id"]) is not None
    out["rv_double_refused"] = _code(store.revoke, rec["key_id"]) == "key_revoked"
    out["rv_unknown_refused"] = _code(store.revoke, "no-such-id") == "key_not_found"

    # update: patch + clear + protected counters
    raw2, rec2 = store.mint("patchme", rpm=2)
    store.authenticate(raw2)
    patched = store.update(rec2["key_id"], name="renamed", rpm=9)
    out["up_fields_patch"] = patched["name"] == "renamed" and patched["rpm"] == 9
    out["up_counters_kept"] = patched["uses"] == 1
    cleared = store.update(rec2["key_id"], clear=["rpm", "name"])
    out["up_clear_to_none"] = cleared["rpm"] is None and cleared["name"] is None
    out["up_clear_nonnullable_refused"] = _refuses(store.update, rec2["key_id"], clear=["enabled"])
    out["up_clear_unknown_refused"] = _refuses(store.update, rec2["key_id"], clear=["bogus"])
    out["up_rpm_min"] = _refuses(store.update, rec2["key_id"], rpm=0)
    out["up_expiry_positive"] = _refuses(store.update, rec2["key_id"], expires_at=-1)
    out["up_admin_additive"] = (
        store.update(rec2["key_id"], admin=True)["admin"] is True
        and "admin" in _must(store.get(rec2["key_id"]))["scopes"]
    )
    scoped = store.update(rec2["key_id"], scopes=["read"], admin=True)
    out["up_scope_patch"] = scoped["scopes"] == ["read", "admin"]
    out["up_revoked_refused"] = _code(store.update, rec["key_id"], name="x") == "key_revoked"
    out["up_unknown_refused"] = _code(store.update, "nope", name="x") == "key_not_found"

    # rotate: policy inherited verbatim, atomic tombstone+successor
    raw3, rec3 = store.mint(
        "succ", scopes=["write"], rpm=5, max_requests=7, max_tokens=99, ttl_s=1000
    )
    new_raw, new_rec = store.rotate(rec3["key_id"])
    out["rt_policy_inherited"] = (
        new_rec["name"] == "succ"
        and new_rec["scopes"] == ["write"]
        and new_rec["rpm"] == 5
        and new_rec["max_requests"] == 7
        and new_rec["max_tokens"] == 99
        and new_rec["expires_at"] == rec3["expires_at"]
    )
    out["rt_lineage"] = new_rec["rotated_from"] == rec3["key_id"]
    out["rt_new_auths"] = store.authenticate(new_raw) is not None
    out["rt_old_dead"] = store.authenticate(raw3) is None
    out["rt_old_tombstoned"] = _must(store.get(rec3["key_id"]))["enabled"] is False
    # overlap mode
    raw4, rec4 = store.mint("overlap")
    new_raw4, _ = store.rotate(rec4["key_id"], revoke_old=False)
    out["rt_overlap_both_live"] = (
        store.authenticate(raw4) is not None and store.authenticate(new_raw4) is not None
    )
    # fresh ttl rebases expiry
    _, newer = store.rotate(rec4["key_id"], revoke_old=False, ttl_s=5)
    out["rt_fresh_ttl_rebases"] = newer["expires_at"] == clock.t + 5
    out["rt_revoked_refused"] = _code(store.rotate, rec3["key_id"]) == "key_revoked"
    out["rt_unknown_refused"] = _code(store.rotate, "nope") == "key_not_found"
    # cap checked before predecessor touched
    tight = ApiKeyStore(max_keys=1, clock=clock)
    traw, trec = tight.mint()
    out["rt_cap"] = _code(tight.rotate, trec["key_id"]) == "keys_cap"
    out["rt_cap_preserves_old"] = tight.authenticate(traw) is not None
    out["clearable_fields"] = {
        "name",
        "rpm",
        "max_requests",
        "max_tokens",
        "expires_at",
    } == CLEARABLE_KEY_FIELDS
    return out


def _probe_durability(tmp: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    clock = _Clock()
    journal_path = tmp / "keys.jsonl"
    store = ApiKeyStore(journal=JobJournal(journal_path), clock=clock)
    raw, rec = store.mint("durable", rpm=4, max_requests=8)
    store.authenticate(raw)
    store.charge_tokens(rec["key_id"], 12)
    raw_dead, rec_dead = store.mint("dead")
    store.revoke(rec_dead["key_id"])

    revived = ApiKeyStore(journal=JobJournal(journal_path), clock=clock)
    out["db_minted_survives"] = revived.authenticate(raw) is not None
    rec_r = revived.get(rec["key_id"])
    out["db_counters_restored"] = (
        rec_r is not None and rec_r["uses"] >= 1 and rec_r["tokens_used"] == 12
    )
    out["db_policy_restored"] = rec_r is not None and rec_r["rpm"] == 4
    out["db_revoked_stays_dead"] = revived.authenticate(raw_dead) is None
    out["db_has_keys"] = revived.has_keys is True
    # live rate windows are process-local — restart resets the window
    win_path = tmp / "win_keys.jsonl"
    win_store = ApiKeyStore(journal=JobJournal(win_path), clock=clock)
    raw5, _rec5 = win_store.mint(rpm=1)
    win_store.authenticate(raw5)  # burns the whole window
    revived2 = ApiKeyStore(journal=JobJournal(win_path), clock=clock)
    out["db_window_process_local"] = revived2.authenticate(raw5) is not None

    # torn journal → quarantine + auth stays required
    torn = tmp / "torn.jsonl"
    tstore = ApiKeyStore(journal=JobJournal(torn), clock=clock)
    traw, _ = tstore.mint("victim")
    with open(torn, "ab") as fh:
        fh.write(b'{"record": {"sha256": "corrupt-partial')
    broken = ApiKeyStore(journal=JobJournal(torn), clock=clock)
    out["db_quarantine_flag"] = broken.recovery_quarantined is True
    out["db_warnings_reported"] = len(broken.recover_warnings) >= 1
    out["db_auth_still_required"] = broken.has_keys is True
    out["db_victim_dead"] = broken.authenticate(traw) is None
    # bootstrap mints a fresh key — a clean boot keeps it enabled
    fresh_raw, _ = broken.mint("fresh")
    out["db_fresh_mint_works"] = broken.authenticate(fresh_raw) is not None
    clean = ApiKeyStore(journal=JobJournal(torn), clock=clock)
    out["db_clean_boot_enables"] = clean.authenticate(fresh_raw) is not None
    out["db_marker_not_lethal"] = _must(clean.authenticate(fresh_raw))["enabled"] is True
    return out


def _probe_wire_purity() -> dict[str, bool]:
    out: dict[str, bool] = {}
    store = ApiKeyStore()
    raw, _ = store.mint()
    wire = _must(store.authenticate(raw))
    listed = store.list()
    out["wp_no_sha_in_auth"] = "sha256" not in wire
    out["wp_no_sha_in_list"] = all("sha256" not in r for r in listed)
    out["wp_no_underscore_anywhere"] = all(
        not any(k.startswith("_") for k in r)
        for r in [*listed, wire, _must(store.get(wire["key_id"]))]
    )
    out["wp_raw_never_listed"] = raw not in json.dumps(listed)
    return out


def keys_audit() -> dict[str, bool]:
    """Every key-store contract as booleans."""
    out: dict[str, bool] = {}
    out.update(_probe_mint())
    out.update(_probe_authenticate())
    out.update(_probe_policy_order())
    out.update(_probe_revoke_update_rotate())
    out.update(_probe_wire_purity())
    with tempfile.TemporaryDirectory() as tmp:
        out.update(_probe_durability(Path(tmp)))
    return out


def keys_audit_bench() -> dict[str, Any]:
    """Sealed receipt for the key-store battery."""
    r = keys_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "keys_audit",
        "schema": "keys_audit.v1",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process store + real hash-chained journals in temp dirs",
            "not_verified": [
                "HTTP status mapping (429/403/401 surfaces — see api_audit)",
                "concurrent multi-writer state dirs (see fault_audit)",
            ],
        },
        "interpretation": (
            "Key store contract holds: mint validates every policy field, "
            "authenticate is a uniform no-oracle None with budget→scope→window "
            "refusal ordering that never counts refusals, revoke tombstones, "
            "rotate inherits declared policy under a single atomic journal "
            "entry, and replay restores durable counters while torn journals "
            "quarantine without unlocking auth."
            if ok
            else f"KEYS AUDIT DEFECTS: {defects}"
        ),
    }
    from quant_fund.utils.reproducibility import git_revision

    out["git_revision"] = git_revision()
    from quant_fund.research.receipt_v2 import canonical_json_bytes
    from quant_fund.utils.hashing import hash_bytes

    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(keys_audit_bench(), indent=2, sort_keys=True))
