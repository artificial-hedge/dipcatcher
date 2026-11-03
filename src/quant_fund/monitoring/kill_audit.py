"""Kill-switch transition audit — the safety gate's full contract,
enumerated.

Every state x action cell is checked against the declared spec:

- ``ENABLED``: orders allowed; cancel=False; flatten=False.
- ``HALT_NEW_ORDERS``: orders blocked; cancel=False; flatten=False.
- ``CANCEL_OPEN_ORDERS``: orders blocked; cancel=True; flatten=False.
- ``FLATTEN_OPTIONAL``: orders blocked; cancel=True; flatten only with
  ``human_authorized=True`` (and never under ``allow_auto_flatten``).
- Any *unknown* state (typo, future enum, empty): construction raises
  ``KillSwitchActive`` (fail closed). Cancel and flatten are not offered.
- ``set_state`` or attribute assignment of an unknown name must raise,
  never transition.

Sealed ``kill_audit.v1``.
"""

from __future__ import annotations

from typing import Any

from quant_fund.config.models import KillSwitchConfig
from quant_fund.monitoring.kill_switch import (
    CANCEL_OPEN_ORDERS,
    ENABLED,
    FLATTEN_OPTIONAL,
    HALT_NEW_ORDERS,
    KillSwitch,
)
from quant_fund.schemas.errors import KillSwitchActive
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

# spec[state] -> (orders_allowed, cancel, flatten_when_authorized)
_SPEC: dict[str, tuple[bool, bool, bool]] = {
    ENABLED: (True, False, False),
    HALT_NEW_ORDERS: (False, False, False),
    CANCEL_OPEN_ORDERS: (False, True, False),
    FLATTEN_OPTIONAL: (False, True, True),
}
_UNKNOWN = ["BOGUS", "", "enabled", "HALT", "FLATTEN", "0", "None"]


def _mk(state: str, auto_flatten: bool = False) -> KillSwitch:
    return KillSwitch(KillSwitchConfig(state=state, allow_auto_flatten=auto_flatten))


def _record_unknown(rec: Any) -> None:
    """Unknown names are rejected at construction and on ``set_state``."""
    for bad in _UNKNOWN:
        try:
            _mk(bad)
        except KillSwitchActive:
            stored = False
        else:
            stored = True
        rec(
            f"unknown[{bad!r}]:orders", not stored, "" if not stored else "invalid state was stored"
        )
        detail = "construction rejected" if not stored else "invalid state was stored"
        rec(f"unknown[{bad!r}]:cancel+flatten", not stored, detail)
        ks2 = _mk(ENABLED)
        try:
            ks2.set_state(bad)
            rec(f"set_state[{bad!r}]", False, f"transitioned to {ks2.state!r}")
        except KillSwitchActive:
            rec(f"set_state[{bad!r}]", ks2.state == ENABLED)


def kill_switch_audit() -> dict[str, Any]:
    """Enumerate every state x action cell; record spec violations."""
    checks: list[dict[str, Any]] = []

    def rec(label: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": label, "ok": ok, "detail": detail})

    for state, (orders_ok, cancel_ok, flatten_auth) in _SPEC.items():
        for auto in (False, True):
            ks = _mk(state, auto)
            tag = f"{state}|auto={auto}"
            try:
                ks.assert_new_orders_allowed()
                got_orders = True
            except KillSwitchActive:
                got_orders = False
            rec(f"{tag}:orders", got_orders == orders_ok, f"allowed={got_orders}")
            rec(
                f"{tag}:cancel",
                ks.may_cancel_open_orders() == cancel_ok,
                f"may_cancel={ks.may_cancel_open_orders()}",
            )
            for human in (False, True):
                got = ks.may_flatten(human)
                want = flatten_auth and human and not auto
                rec(
                    f"{tag}:flatten(human={human})",
                    got == want,
                    f"may_flatten={got}",
                )

    _record_unknown(rec)

    # valid transitions all reachable
    for state in _SPEC:
        ks = _mk(ENABLED)
        try:
            ks.set_state(state)
            rec(f"transition:ENABLED->{state}", ks.state == state)
        except KillSwitchActive:
            rec(f"transition:ENABLED->{state}", False, "raised")

    n_bad = sum(1 for c in checks if not c["ok"])
    return {
        "n_checks": len(checks),
        "n_violations": n_bad,
        "verdict": "ok" if n_bad == 0 else "violations",
        "violations": [c for c in checks if not c["ok"]],
    }


def kill_audit_bench() -> dict[str, Any]:
    report = kill_switch_audit()
    payload: dict[str, Any] = {
        "kind": "kill_audit",
        "schema": "kill_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "the kill switch implements its declared state matrix; unknown states fail closed on every action",
            "verdict": report["verdict"],
            "n_checks": report["n_checks"],
            "n_violations": report["n_violations"],
        },
        "interpretation": report,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
