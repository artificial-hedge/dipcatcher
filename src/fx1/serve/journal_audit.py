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
- ``_BatchStore`` (``batches.jsonl``) recovers batch records the same
  way — mid-flight batches (``validating``/``in_progress``/
  ``finalizing``/``cancelling``) recover ``failed`` since input lines
  are never journaled; output/error file ids and webhook verdicts ride
  the record.
- ``FTJobStore`` (``ft_jobs.jsonl``) restores jobs with their event
  feed, ``ft:`` idempotency keys, and the fine-tuned model registry —
  a card whose producing job was evicted never resolves post-restart.
- ``_FileStore`` (``files.jsonl`` + ``files/*.bin`` blobs) restores
  uploads byte-identical; a journaled record whose blob is missing
  fails closed — boot refuses without rewriting the journal —
  deletes/evictions tombstone, orphans GC on boot.
- ``_IdemStore`` (``idem_*.jsonl``) journals the stored response itself
  so a retried submission replays the recorded answer after a restart.

Sealed ``journal_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from fx1.serve.api import JobStatusResponse, _BatchRecord
    from fx1.serve.evals import EvalRecord
    from fx1.serve.finetune import FTJob

__all__ = ["journal_audit", "journal_audit_bench"]


def _mk_eval(eval_id: str, status: str = "queued", key: str | None = None) -> EvalRecord:
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


def _mk_batch(batch_id: str, status: str = "validating") -> _BatchRecord:
    from fx1.serve.api import _BatchRecord

    rec = _BatchRecord(
        batch_id=batch_id,
        input_file_id="file-input",
        endpoint="/v1/chat/completions",
        completion_window="24h",
        status=status,
        created_at=1000,
        expires_at=99999,
        callback_url="https://cb.example/batch",
    )
    rec._callback_secret = "bsecret-" + batch_id  # noqa: SLF001
    return rec


def _mk_ftjob(job_id: str, status: str = "queued") -> FTJob:
    from fx1.serve.finetune import FTJob

    job = FTJob(
        id=job_id,
        model="local_fx1",
        created_at=1000,
        status=status,  # type: ignore[arg-type]
        training_file="file-train",
        callback_url="https://cb.example/ft",
    )
    job._callback_secret = "fsecret-" + job_id  # noqa: SLF001
    return job


def _mk_job(job_id: str, status: str = "queued", key: str | None = None) -> JobStatusResponse:
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
        edone = _mk_eval("e-ok", "succeeded", key="k-eok")
        edone.finished_at = 1001.0
        estore.put(edone, "k-eok", "efp-ok")
        erunning = _mk_eval("e-r", "queued")
        estore.put(erunning, None, None)
        erunning.status = "running"
        estore.mark(erunning)

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

    # --- _BatchStore (batches.jsonl) ------------------------------------
    from fx1.serve.api import _BatchStore

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "batches.jsonl"
        bstore = _BatchStore(8, journal=JobJournal(path))
        bstore.put(_mk_batch("b-v", "validating"))
        done_b = _mk_batch("b-ok", "completed")
        done_b.completed_at = 1001
        done_b.output_file_id = "file-out"
        bstore.put(done_b)
        live = _mk_batch("b-r", "validating")
        bstore.put(live)
        live.status = "in_progress"
        live.in_progress_at = 1002
        bstore.mark(live)

        bstore2 = _BatchStore(8, journal=JobJournal(path))
        rb = bstore2.get("b-v")
        r["batch_validating_recovers_failed"] = (
            rb is not None
            and rb.status == "failed"
            and rb.errors is not None
            and "restarted" in json.dumps(rb.errors)
        )
        rb_ok = bstore2.get("b-ok")
        r["batch_terminal_as_was"] = (
            rb_ok is not None
            and rb_ok.status == "completed"
            and rb_ok.output_file_id == "file-out"
            and rb_ok.completed_at == 1001
        )
        r["batch_inflight_recovers_failed"] = (
            bstore2.get("b-r") is not None and bstore2.get("b-r").status == "failed"  # type: ignore[union-attr]
        )
        raw = path.read_bytes()
        r["batch_secret_not_journaled"] = b"bsecret-" not in raw
        r["batch_secret_not_recovered"] = (
            rb is not None and rb._callback_secret is None  # noqa: SLF001
        )

        # 'cancelling' is mid-flight — recovers failed, never replays lines
        cx = _mk_batch("b-cx", "validating")
        bstore2.put(cx)
        cx.status = "cancelling"
        cx.cancelling_at = 1003
        bstore2.mark(cx)
        bstore3 = _BatchStore(8, journal=JobJournal(path))
        r["batch_cancelling_recovers_failed"] = (
            bstore3.get("b-cx") is not None and bstore3.get("b-cx").status == "failed"  # type: ignore[union-attr]
        )
        with tempfile.TemporaryDirectory() as td7:
            p7 = Path(td7) / "b.jsonl"
            bs = _BatchStore(2, journal=JobJournal(p7))
            for i in range(4):
                bs.put(_mk_batch(f"b-{i}", "completed"))
            bs2 = _BatchStore(2, journal=JobJournal(p7))
            r["batch_evict_journaled"] = bs2.get("b-0") is None and bs2.get("b-3") is not None

    # --- FTJobStore (ft_jobs.jsonl) -------------------------------------
    from fx1.serve.finetune import FTJobStore

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ft_jobs.jsonl"
        fstore = FTJobStore(8, journal=JobJournal(path))
        fstore.put(_mk_ftjob("f-q", "queued"), "k-fq", "ffp-q")
        succ = _mk_ftjob("f-ok", "succeeded")
        succ.finished_at = 1001
        succ.fine_tuned_model = "ft:local_fx1:x:fok1234abcd"
        fstore.put(succ, "k-fok", "ffp-ok")
        fstore.add_event("f-ok", "info", "job succeeded", {"checkpoint": "ckpt-1"})
        fstore.register_model(
            "ft:local_fx1:x:fok1234abcd",
            job_id="f-ok",
            checkpoint="ckpt-1",
            created=1001,
        )

        fstore2 = FTJobStore(8, journal=JobJournal(path))
        fe = fstore2.get("f-q")
        r["ft_queued_recovers_failed"] = (
            fe is not None
            and fe.job.status == "failed"
            and fe.job.error is not None
            and "restarted" in fe.job.error.message
        )
        fe_ok = fstore2.get("f-ok")
        r["ft_terminal_as_was"] = (
            fe_ok is not None
            and fe_ok.job.status == "succeeded"
            and fe_ok.job.fine_tuned_model == "ft:local_fx1:x:fok1234abcd"
        )
        r["ft_events_survive"] = (
            fe_ok is not None
            and len(fe_ok.events) == 1
            and fe_ok.events[0].message == "job succeeded"
        )
        r["ft_idem_key_survives"] = (
            fstore2.lookup_idem("k-fq") is not None and fstore2.lookup_idem("k-fq").job.id == "f-q"  # type: ignore[union-attr]
        )
        r["ft_model_registry_survives"] = (
            fstore2.checkpoint_for("ft:local_fx1:x:fok1234abcd") == "ckpt-1"
        )
        raw = path.read_bytes()
        r["ft_secret_not_journaled"] = b"fsecret-" not in raw
        r["ft_secret_not_recovered"] = (
            fe is not None and fe.job._callback_secret is None  # noqa: SLF001
        )

        # queued cancel journaled
        fstore2.put(_mk_ftjob("f-c", "queued"), "k-fc", "ffp-c")
        fstore2.request_cancel("f-c")
        fstore3 = FTJobStore(8, journal=JobJournal(path))
        r["ft_cancel_journaled"] = (
            fstore3.get("f-c") is not None and fstore3.get("f-c").job.status == "cancelled"  # type: ignore[union-attr]
        )

        # evicting the producing job drops the model card post-restart
        with tempfile.TemporaryDirectory() as td8:
            p8 = Path(td8) / "f.jsonl"
            fs = FTJobStore(2, journal=JobJournal(p8))
            j0 = _mk_ftjob("f-0", "succeeded")
            j0.fine_tuned_model = "ft:m:a:f0"
            fs.put(j0, None, "")
            fs.register_model("ft:m:a:f0", job_id="f-0", checkpoint="c0", created=1)
            for i in (1, 2, 3):
                fs.put(_mk_ftjob(f"f-{i}", "succeeded"), None, "")
            fs2 = FTJobStore(2, journal=JobJournal(p8))
            r["ft_model_dies_with_job"] = fs2.get_model("ft:m:a:f0") is None

    # --- _FileStore (files.jsonl + blobs) --------------------------------
    from fx1.serve.api import _FileStore

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        filestore = _FileStore(8, 1 << 20, state_dir=root)
        frec = filestore.put(filename="up.jsonl", purpose="batch", content=b'{"a":1}\n')
        gone = filestore.put(filename="gone.jsonl", purpose="batch", content=b"XX")
        filestore.delete(gone.file_id)
        filestore2 = _FileStore(8, 1 << 20, state_dir=root)
        r["file_bytes_survive"] = (
            filestore2.get(frec.file_id) is not None
            and filestore2.get(frec.file_id).content == b'{"a":1}\n'  # type: ignore[union-attr]
        )
        r["file_delete_journaled"] = filestore2.get(gone.file_id) is None
        raw = (root / "files.jsonl").read_bytes()
        r["file_content_not_in_journal"] = b'{"a":1}' not in raw and b"XX" not in raw

        # a journaled record whose blob is missing fails closed — boot
        # refuses rather than drop the record, leaving the journal
        # byte-for-byte intact for operator repair
        with tempfile.TemporaryDirectory() as td8b:
            root8b = Path(td8b)
            fstore8b = _FileStore(8, 1 << 20, state_dir=root8b)
            orphan = fstore8b.put(filename="orphan.jsonl", purpose="batch", content=b"zz")
            (root8b / "files" / f"{orphan.file_id}.bin").unlink()
            journal_before = (root8b / "files.jsonl").read_bytes()
            try:
                _FileStore(8, 1 << 20, state_dir=root8b)
                refused = False
            except RuntimeError:
                refused = True
            r["file_missing_blob_fails_closed"] = (
                refused and (root8b / "files.jsonl").read_bytes() == journal_before
            )

        # orphan blob GC: a blob with no journaled record is unlinked
        stray = root / "files" / "file-stray.bin"
        stray.write_bytes(b"stray")
        _FileStore(8, 1 << 20, state_dir=root)
        r["file_orphan_blob_gcd"] = not stray.exists()

        with tempfile.TemporaryDirectory() as td9:
            root9 = Path(td9)
            fst = _FileStore(2, 1 << 20, state_dir=root9)
            ids = [
                fst.put(filename=f"f{i}.jsonl", purpose="batch", content=b"x").file_id
                for i in range(4)
            ]
            fst2 = _FileStore(2, 1 << 20, state_dir=root9)
            r["file_evict_journaled"] = fst2.get(ids[0]) is None and fst2.get(ids[3]) is not None
            r["file_evicted_blob_gone"] = not (root9 / "files" / f"{ids[0]}.bin").exists()

    # --- _IdemStore (idem_*.jsonl) ---------------------------------------
    from fx1.serve.api import HarnessRunResponse, _IdemStore

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "idem.jsonl"
        idem = _IdemStore[HarnessRunResponse](4, journal=JobJournal(path), model=HarnessRunResponse)
        resp = HarnessRunResponse(
            command="selftest",
            exit_code=0,
            stdout="ok",
            stderr="",
            ok=True,
            timeout_s=30,
            stdout_truncated=False,
            stderr_truncated=False,
        )
        idem.put("key-1", "fp-1", resp)
        idem2 = _IdemStore[HarnessRunResponse](
            4, journal=JobJournal(path), model=HarnessRunResponse
        )
        hit = idem2.get("key-1")
        r["idem_replays_response"] = (
            hit is not None and hit[0] == "fp-1" and hit[1].stdout == "ok" and hit[1].ok is True
        )

        with tempfile.TemporaryDirectory() as td10:
            p10 = Path(td10) / "i.jsonl"
            ist = _IdemStore[HarnessRunResponse](
                2, journal=JobJournal(p10), model=HarnessRunResponse
            )
            for i in range(4):
                ist.put(f"k-{i}", f"fp-{i}", resp)
            ids2 = _IdemStore[HarnessRunResponse](
                2, journal=JobJournal(p10), model=HarnessRunResponse
            )
            r["idem_evict_journaled"] = ids2.get("k-0") is None and ids2.get("k-3") is not None

        r["idem_requires_model"] = False
        try:
            _IdemStore[HarnessRunResponse](2, journal=JobJournal(Path(td) / "x.jsonl"))
        except ValueError:
            r["idem_requires_model"] = True

    # --- report serialization must be journal-safe ----------------------
    # A real defect found while binding the SDK store: dataclass reports
    # carry numpy leaves (calibration's `extracted`), which crashed
    # model_dump(mode="json") inside put/mark — silent journal gaps.
    from dataclasses import dataclass  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from fx1.serve.evals import report_dump  # noqa: PLC0415

    @dataclass
    class _ArrReport:
        bins: list[float]
        extracted: Any

    dump = report_dump(_ArrReport(bins=[0.5, 1.0], extracted=np.arange(4)))
    r["report_dump_json_safe"] = dump["extracted"] == [0, 1, 2, 3] and isinstance(
        json.dumps(dump), str
    )
    try:
        report_dump({"not": "a report"})
        r["report_dump_fail_closed"] = False
    except TypeError:
        r["report_dump_fail_closed"] = True

    # --- SDK stores journal under state_dir ------------------------------
    # Fx1Harness(state_dir=...) binds the same evals/ft_jobs journals a
    # second instance replays — the in-process twin is durable too.
    from fx1.sdk import Fx1Harness  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as td:
        h = Fx1Harness(state_dir=td)
        srec = h.run_eval("calibration", model_fn=lambda msgs: "0.5")
        h2 = Fx1Harness(state_dir=td)
        rec2 = h2.eval_record(srec.eval_id)
        r["sdk_eval_journaled"] = (
            srec.status == "succeeded"
            and rec2.status == "succeeded"
            and rec2.report is not None
            and srec.report is not None
            and rec2.report.get("extracted") == srec.report.get("extracted")
        )
        # a running eval record left by a 'crash' (put without mark)
        crash_rec = _mk_eval("sdk-crash", "queued")
        h2._eval_store.put(crash_rec, None, None)  # noqa: SLF001
        h3 = Fx1Harness(state_dir=td)
        r["sdk_inflight_recovers_failed"] = h3.eval_record("sdk-crash").status == "failed"

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
            "unchanged when no state dir is configured. The same binding "
            "holds on evals, batches (in-flight recovers failed), "
            "fine-tune jobs (events, idem keys, and the model registry — "
            "a card dies with its producing job), files (blob bytes "
            "round-trip, a journaled record whose blob is missing "
            "refuses boot without rewriting the journal, deletes "
            "tombstone, orphans GC), and the "
            "idempotency stores (the recorded response replays after a "
            "restart)."
            if ok
            else f"JOURNAL AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(journal_audit_bench(), indent=2, sort_keys=True))
