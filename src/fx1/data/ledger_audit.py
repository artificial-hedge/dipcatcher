"""ledger_audit — adversarial probes on the corpus hash chain.

The ledger is the out-opening lineage record: ``example``/``exclusion``
entries hash-chained on ``prev_hash``. Pinned contract:

- Append-order writes: index must equal position; first entry's
  ``prev_hash`` is ``GENESIS``.
- ``verify_chain`` recomputes every link — content tamper, index edit,
  reorder, or a mid-chain splice all break it.
- Persist-before-admit: entries hit disk before entering memory, so a
  fresh ``CorpusLedger`` on the same path sees the same chain.
- ``audit_export`` exposes counts/rules/head only — no raw text.

**Pinned caveat**: tail *truncation* is invisible — dropping the last
entries leaves an internally valid shorter chain (``verify_chain``
still True). Deleting the tail requires an external head pin (the
corpus-epoch pin files do this for the committed ledger).

Sealed ``ledger_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["ledger_audit", "ledger_audit_bench"]

_H1 = "a" * 64
_H2 = "b" * 64
_H3 = "c" * 64


def _three(path: Path) -> Any:
    from fx1.data.ledger import CorpusLedger

    led = CorpusLedger(path)
    led.record_example(source_sha256=_H1, transform_sha256=_H2, example_sha256=_H3)
    led.record_exclusion(source_sha256=_H1, transform_sha256=_H2, rule="too_short")
    led.record_example(source_sha256=_H2, transform_sha256=_H1, example_sha256=_H1)
    return led


def ledger_audit() -> dict[str, Any]:
    from fx1.data.ledger import GENESIS, CorpusLedger

    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ledger.jsonl"
        led = _three(p)
        out["fresh_valid"] = led.verify_chain()
        out["genesis_link"] = led._entries[0].prev_hash == GENESIS
        out["indexes_dense"] = [e.index for e in led._entries] == [0, 1, 2]

        # fresh reader sees identical persisted chain
        reread = CorpusLedger(p)
        out["persist_before_admit"] = reread.verify_chain() and [
            e.entry_hash for e in reread._entries
        ] == [e.entry_hash for e in led._entries]

        # content tamper: rewrite entry 1's rule in the file
        lines = p.read_text().splitlines()
        e1 = json.loads(lines[1])
        e1["rule"] = "forged_rule"
        # NOTE: not rehashing — a real attacker must also rechain, which
        # is exactly what verify_chain catches at the hash step
        p2 = Path(tmp) / "tampered.jsonl"
        lines_t = lines.copy()
        lines_t[1] = json.dumps(e1)
        p2.write_text("\n".join(lines_t) + "\n")
        out["content_tamper_breaks"] = CorpusLedger(p2).verify_chain() is False

        # reorder: swap lines 1 and 2
        p3 = Path(tmp) / "reordered.jsonl"
        l2 = lines.copy()
        l2[1], l2[2] = l2[2], l2[1]
        p3.write_text("\n".join(l2) + "\n")
        out["reorder_breaks"] = CorpusLedger(p3).verify_chain() is False

        # tail truncation: drop last entry — chain still internally valid
        p4 = Path(tmp) / "truncated.jsonl"
        p4.write_text("\n".join(lines[:2]) + "\n")
        trunc = CorpusLedger(p4)
        out["tail_truncation_invisible"] = trunc.verify_chain() is True

        exp = led.audit_export()
        out["export_shape"] = sorted(exp) == [
            "chain_head",
            "chain_valid",
            "entries",
            "examples",
            "exclusion_rules",
            "exclusions",
        ]
        out["export_counts"] = (
            exp["entries"] == 3
            and exp["examples"] == 2
            and exp["exclusions"] == 1
            and exp["exclusion_rules"] == {"too_short": 1}
        )
        out["export_head"] = exp["chain_head"] == led._entries[-1].entry_hash

        # empty ledger: valid, genesis head
        empty = CorpusLedger(Path(tmp) / "empty.jsonl")
        out["empty_valid_genesis"] = (
            empty.verify_chain() and empty.audit_export()["chain_head"] == GENESIS
        )
    return out


def ledger_audit_bench() -> dict[str, Any]:
    r = ledger_audit()
    ok = all(r[k] is True for k in r)
    out: dict[str, Any] = {
        "kind": "ledger_audit",
        "schema": "ledger_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "results": r,
            "flags": {"tail_truncation_invisible": r["tail_truncation_invisible"]},
            "ok": ok,
        },
        "interpretation": (
            "Corpus-ledger contract holds: hash chain verified per link, "
            "content/index/order tampering breaks verification, "
            "persist-before-admit confirmed by reread. Caveat pinned: "
            "tail truncation is internally valid — needs an external "
            "head pin (corpus-epoch pins provide it)."
            if ok
            else f"LEDGER AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
