"""doctor_audit — adversarial probes on the fx-1 environment doctor.

The doctor's contract is *presence flags only*: the sharp probe is that
real secret values must never appear anywhere in the status dict —
``collect_status`` is exercised with sentinel env values set, and every
value in the mapping is scanned for the sentinel string.

Also pinned: missing data files report ``"missing"`` (never a crash);
a corrupt/absent corpus ledger reports ``unverifiable``/``missing``
rather than raising; the eval-bank count is reported when importable.

Sealed ``doctor_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["doctor_audit", "doctor_audit_bench"]

_SENTINEL = "SK-SENTINEL-DO-NOT-LEAK-7f3b2a1c"


def doctor_audit() -> dict[str, Any]:
    from fx1.doctor import collect_status

    out: dict[str, Any] = {}
    old_moon = os.environ.get("MOONSHOT_API_KEY")
    old_sign = os.environ.get("FX1_SIGNING_KEY")
    os.environ["MOONSHOT_API_KEY"] = _SENTINEL
    os.environ["FX1_SIGNING_KEY"] = _SENTINEL
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = collect_status(root)
            blob = json.dumps(status)
            out["sentinel_absent"] = _SENTINEL not in blob
            out["flags_only"] = status["moonshot_key"] == "set" and status["signing_key"] == "set"
            out["missing_receipts"] = status["receipts"] == "missing"
            out["missing_data"] = status["data_fx1_corpus"] == "missing"
            out["ledger_missing"] = status["corpus_ledger_chain"] == "missing"
            out["bank_count_int"] = isinstance(status["eval_bank_tasks"], int)

            # half-formed data dir: receipts dir exists but empty
            (root / "receipts").mkdir()
            s2 = collect_status(root)
            out["empty_receipts_zero"] = s2["receipts"] == 0

            # corrupt ledger file → "unverifiable" not a crash
            fx = root / "data" / "fx1"
            fx.mkdir(parents=True)
            (fx / "corpus_ledger.jsonl").write_text("not-json\n{garbage")
            s3 = collect_status(root)
            out["corrupt_ledger"] = s3["corpus_ledger_chain"] in (
                "unverifiable",
                "BROKEN",
            )
    finally:
        for name, old in (
            ("MOONSHOT_API_KEY", old_moon),
            ("FX1_SIGNING_KEY", old_sign),
        ):
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old

    # env unset → flags read "unset"
    os.environ.pop("MOONSHOT_API_KEY", None)
    os.environ.pop("FX1_SIGNING_KEY", None)
    s4 = collect_status(Path(tempfile.gettempdir()) / "nonexistent-doctor-root")
    out["unset_flags"] = s4["moonshot_key"] == "unset" and s4["signing_key"] == "unset"
    out["missing_everything"] = s4["receipts"] == "missing"
    return out


def doctor_audit_bench() -> dict[str, Any]:
    r = doctor_audit()
    ok = all(r[k] is True for k in r)
    out: dict[str, Any] = {
        "kind": "doctor_audit",
        "schema": "doctor_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Doctor contract holds: presence flags only (sentinel secret "
            "never appears in output), missing/corrupt inputs degrade to "
            "labels rather than raising, unset envs read 'unset'."
            if ok
            else f"DOCTOR AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
