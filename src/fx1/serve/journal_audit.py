"""journal_audit — adversarial probes on the durable job journal.

Covers ``fx1.serve.journal.JobJournal`` and its binding into
``_JobStore``: the durability contract that keeps ``/harness/jobs``
survivable across process restarts.

Pinned contract:

- Journal lines are hash-chained (``seq``/``chain``/``sha256``); replay
  returns payloads in order, verifies each link, and stops at the first
  bad line — a torn tail or mid-file edit truncates honestly instead of
  corrupting recovered state (``ReplayResult.truncated_at``/``dropped``).
- ``_JobStore`` with a journal restores terminal records as-was; jobs
  still ``queued``/``running`` at the crash recover as ``failed`` with a
  restart-explaining ``error`` — never silently re-run (the payload is
  not journaled) and never reported as lost-but-pending.
- Idempotency keys survive the restart: ``get_key`` resolves and a
  retried submission would return the recovered record, not a duplicate.
- ``callback_secret`` never reaches disk — ``PrivateAttr`` stays out of
  ``model_dump`` and out of the journal file bytes.
- Cancels and LRU evictions are journaled; boot compacts the journal to
  live records so dead history doesn't accumulate.
- ``journal=None`` keeps the store purely in-memory (unchanged default).
- The same binding on ``EvalStore`` (``evals.jsonl``) recovers eval
  records identically: terminal as-was, in-flight as failed, keys and
  cancels durable, ``callback_secret`` never on disk.

Sealed ``journal_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["journal_audit", "journal_audit_bench"]


def _mk_eval(eval_id: str, status: str = "queued", key: str | None = None):
    from fx1.serve.evals import EvalRecord

    rec = EvalRecord(
        eval_id=eval_id,
        suite="calibration",
        backend="byok",
        seed=0,
        status=status,  # type: ignore[arg-type]
        created_at=1000.0,
        callback_url="https://cb.example/eval" if key else None,
    )
    if key:
        rec._callback_secret = "esecret-" + eval_id  # noqa: SLF001
    return rec


def _mk_job(job_id: str, status: str = "queued", key: str | None = None):
    from fx1.serve.api import JobStatusResponse

    job = JobStatusResponse(
        job_id=job_id,
        status=status,  # type: ignore[arg-type]
        created_at=1000.0,
        finished_at=None,
        result=None,
        error=None,
        callback_url="https://cb.example/hook" if key else None,
    )
    if key:
        job._callback_secret = "secret-" + job_id  # noqa: SLF001 — the probe
        # asserts this private attr never reaches the journal file
    return job


def journal_audit() -> dict[str, Any]:
    from fx1.serve.api import _JobStore
    from fx1.serve.journal import JobJournal

    r: dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "jobs.jsonl"
        j = JobJournal(path)
        for i in range(5):
            j.append({"n": i})
        res = j.replay()
        r["roundtrip_order"] = [p["n"] for p in res.payloads] == list(range(5))

        # torn tail: garbage bytes after the last good line drop cleanly
        with path.open("ab") as fh:
            fh.write(b'{"seq": 5, "chain": "deadbeef", "sh')
        res = j.replay()
        r["torn_tail_truncates"] = (
            len(res.payloads) == 5 and res.truncated_at is not None and res.dropped == 1
        )

        # mid-file edit invalidates from that line on
        with tempfile.TemporaryDirectory() as td2:
            p2 = Path(td2) / "j.jsonl"
            j2 = JobJournal(p2)
            for i in range(6):
                j2.append({"n": i})
            data = bytearray(p2.read_bytes())
            # flip a payload byte inside line 3 (after the 'n' value)
            idx = data.find(b'"n": 3')
            if idx > 0:
                data[idx] = ord("x")
            else:
                idx = len(data) // 2
                data[idx] ^= 0xFF
            p2.write_bytes(bytes(data))
            res = JobJournal(p2).replay()
            r["mid_edit_truncates"] = res.truncated_at is not None and len(res.payloads) <= 5

        # compact keeps only live payloads, in order
        with tempfile.TemporaryDirectory() as td3:
            p3 = Path(td3) / "j.jsonl"
            j3 = JobJournal(p3)
            for i in range(8):
                j3.append({"n": i})
            j3.compact([{"n": 7}, {"n": 8}])
            res = j3.replay()
            r["compact_live_only"] = [p["n"] for p in res.payloads] == [7, 8]

    # --- store binding --------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "jobs.jsonl"

        # crash before terminal: queued job recovers as failed
        store = _JobStore(8, journal=JobJournal(path))
        store.put(_mk_job("j-q", "queued", key="k-q"), "k-q", "fp-q")
        done = _mk_job("j-ok", "succeeded", key="k-ok")
        done.finished_at = 1001.0
        store.put(done, "k-ok", "fp-ok")

        store2 = _JobStore(8, journal=JobJournal(path))
        rec = store2.get("j-q")
        r["recovers_queued_as_failed"] = (
            rec is not None
            and rec.status == "failed"
            and rec.error is not None
            and "restarted" in rec.error
        )
        rec_ok = store2.get("j-ok")
        r["recovers_terminal_as_was"] = rec_ok is not None and rec_ok.status == "succeeded"
        r["idem_key_survives"] = store2.get_key("k-q") == ("fp-q", "j-q")

        # secrets never journaled
        raw = path.read_bytes()
        r["secret_not_journaled"] = b"secret-j-q" not in raw and b"secret-j-ok" not in raw
        # and the recovered record can't deliver (no secret restored)
        recovered = store2.get("j-q")
        r["secret_not_recovered"] = (
            recovered is not None and recovered._callback_secret is None  # noqa: SLF001
        )

        # cancel is journaled
        store2.put(_mk_job("j-c", "queued"), "k-c", "fp-c")
        store2.cancel("j-c")
        store3 = _JobStore(8, journal=JobJournal(path))
        r["cancel_journaled"] = (
            store3.get("j-c") is not None and store3.get("j-c").status == "cancelled"  # type: ignore[union-attr]
        )

        # LRU eviction is journaled: overfill drops the oldest job+key
        with tempfile.TemporaryDirectory() as td4:
            p4 = Path(td4) / "j.jsonl"
            s = _JobStore(2, journal=JobJournal(p4))
            for i in range(4):
                s.put(_mk_job(f"j-{i}", "succeeded"), f"k-{i}", f"fp-{i}")
            s2 = _JobStore(2, journal=JobJournal(p4))
            r["evict_journaled"] = s2.get("j-0") is None and s2.get("j-1") is None
            r["evict_drops_key"] = s2.get_key("k-0") is None
            r["evict_survivors"] = s2.get("j-2") is not None and s2.get("j-3") is not None

        # boot compaction shrinks the file vs the un-compacted log
        with tempfile.TemporaryDirectory() as td5:
            p5 = Path(td5) / "j.jsonl"
            s = _JobStore(4, journal=JobJournal(p5))
            job = _mk_job("j-m", "queued")
            s.put(job, "k-m", "fp-m")
            size_before = p5.stat().st_size
            s.cancel("j-m")
            size_after_cancel = p5.stat().st_size
            _JobStore(4, journal=JobJournal(p5))  # boot = replay + compact
            size_boot = p5.stat().st_size
            r["boot_compacts"] = size_boot < size_after_cancel
            r["journal_grows_on_transition"] = size_after_cancel > size_before

        # in-memory default unchanged
        plain = _JobStore(4)
        plain.put(_mk_job("j-x", "queued"), "k-x", "fp-x")
        r["no_journal_still_works"] = plain.get("j-x") is not None and not hasattr(
            _JobStore(4), "_journal_missing"
        )

    # --- the same binding on EvalStore (evals.jsonl) --------------------
    from fx1.serve.evals import EvalStore

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "evals.jsonl"
        estore = EvalStore(8, journal=JobJournal(path))
        estore.put(_mk_eval("e-q", "queued", key="k-eq"), "k-eq", "efp-q")
        done = _mk_eval("e-ok", "succeeded", key="k-eok")
        done.finished_at = 1001.0
        estore.put(done, "k-eok", "efp-ok")
        running = _mk_eval("e-r", "queued")
        estore.put(running, None, None)
        running.status = "running"
        estore.mark(running)

        estore2 = EvalStore(8, journal=JobJournal(path))
        erec = estore2.get("e-q")
        r["eval_queued_recovers_failed"] = (
            erec is not None
            and erec.status == "failed"
            and erec.error is not None
            and "restarted" in erec.error
        )
        r["eval_terminal_as_was"] = (
            estore2.get("e-ok") is not None and estore2.get("e-ok").status == "succeeded"  # type: ignore[union-attr]
        )
        r["eval_running_recovers_failed"] = (
            estore2.get("e-r") is not None and estore2.get("e-r").status == "failed"  # type: ignore[union-attr]
        )
        r["eval_idem_key_survives"] = estore2.get_key("k-eq") == ("efp-q", "e-q")
        raw = path.read_bytes()
        r["eval_secret_not_journaled"] = b"esecret-e-q" not in raw and b"esecret-e-ok" not in raw
        erecovered = estore2.get("e-q")
        r["eval_secret_not_recovered"] = (
            erecovered is not None and erecovered._callback_secret is None  # noqa: SLF001
        )

        # cancel + eviction journaled on the eval store too
        estore2.put(_mk_eval("e-c", "queued"), "k-ec", "efp-c")
        estore2.cancel("e-c")
        estore3 = EvalStore(8, journal=JobJournal(path))
        r["eval_cancel_journaled"] = (
            estore3.get("e-c") is not None and estore3.get("e-c").status == "cancelled"  # type: ignore[union-attr]
        )
        with tempfile.TemporaryDirectory() as td6:
            p6 = Path(td6) / "e.jsonl"
            es = EvalStore(2, journal=JobJournal(p6))
            for i in range(4):
                es.put(_mk_eval(f"e-{i}", "succeeded"), f"ek-{i}", f"efp-{i}")
            es2 = EvalStore(2, journal=JobJournal(p6))
            r["eval_evict_journaled"] = es2.get("e-0") is None and es2.get("e-3") is not None

    return r


def journal_audit_bench() -> dict[str, Any]:
    r = journal_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "journal_audit",
        "schema": "journal_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "the durable job journal holds: hash-chained replay returns "
            "records in order, torn tails and mid-file edits truncate "
            "honestly, and compaction rewrites only live state. Bound to "
            "the job store, terminal jobs come back as-was, in-flight jobs "
            "recover as failed with an honest restart error, idempotency "
            "keys still resolve, cancels and LRU evictions survive, and "
            "callback secrets never touch disk. The in-memory default is "
            "unchanged when no state dir is configured."
            if ok
            else f"JOURNAL AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(journal_audit_bench(), indent=2, sort_keys=True))
