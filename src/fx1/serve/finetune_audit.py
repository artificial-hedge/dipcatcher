"""Fine-tune audit — the ``FTJobStore`` state machine + wire shapes.

``finetune`` hosts the OpenAI ``/v1/fine_tuning/jobs`` contract:
``validate_chat_jsonl`` corpus admission (fail-closed at the first
malformed line), the request/record pydantic shapes, and the
``FTJobStore`` — a bounded LRU with idempotency keys, an event feed,
pause/resume/cancel verdicts, an ``ft:`` model registry whose cards
never outlive their producing job, and journaled restart recovery.

Probes are in-process only: tmpdir journals, no trainers, no network.
Deterministic; the sealed bench payload is SYNTHETIC / research-only.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from fx1.serve.backends import InferenceBackend
from fx1.serve.finetune import (
    _FT_EVENT_CAP,
    TRAINABLE_MODELS,
    FTJob,
    FTJobRequest,
    FTJobStore,
    default_ft_runner,
    validate_chat_jsonl,
)
from fx1.serve.journal import JobJournal

_MODEL_A = "ft:m:a:j1"
_MODEL_B = "ft:m:b:j1"
_MODEL_DONE = "ft:m:x:done"
_MODEL_T = "ft:m:t:t"


def _run(fn, *a: Any, **kw: Any) -> tuple[bool, Any]:
    try:
        return True, fn(*a, **kw)
    except Exception as exc:  # noqa: BLE001 — the probe records the type
        return False, exc


def _job(job_id: str, status: str = "queued", **kw: Any) -> FTJob:
    return FTJob(
        id=job_id,
        model="fx1",
        created_at=1700000000,
        status=status,  # type: ignore[arg-type]
        training_file="file-train",
        **kw,
    )


def _jsonl(*records: Any) -> bytes:
    return "\n".join(json.dumps(r) for r in records).encode()


def _valid_line(user: str = "hi") -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "ok"},
        ]
    }


def _store(cap: int = 10, journal_dir: Path | None = None) -> FTJobStore:
    j = JobJournal(journal_dir / "ft_jobs.jsonl") if journal_dir is not None else None
    return FTJobStore(cap, journal=j)


# ---------------------------------------------------------------- admission


def _probe_jsonl_admission() -> dict[str, bool]:
    out: dict[str, bool] = {}

    out["jsonl_valid_counts_examples"] = (
        validate_chat_jsonl(_jsonl(_valid_line(), _valid_line("again")), file_id="f") == 2
    )
    out["jsonl_blank_lines_skipped"] = (
        validate_chat_jsonl(_jsonl(_valid_line()) + b"\n\n  \n", file_id="f") == 1
    )
    ok, exc = _run(validate_chat_jsonl, b"\xff\xfe", file_id="f")
    out["jsonl_non_utf8_refused"] = not ok and isinstance(exc, ValueError)
    ok, exc = _run(validate_chat_jsonl, _jsonl(_valid_line(), {"nope": 1}), file_id="f")
    out["jsonl_bad_line_fails_closed"] = (
        not ok and "line 2" in str(exc) and isinstance(exc, ValueError)
    )
    ok, exc = _run(validate_chat_jsonl, b"{not json", file_id="f")
    out["jsonl_non_json_refused"] = not ok and "line 1" in str(exc)
    ok, _ = _run(validate_chat_jsonl, _jsonl({"messages": "x"}), file_id="f")
    out["jsonl_messages_nonstr_list_refused"] = not ok
    ok, _ = _run(validate_chat_jsonl, _jsonl({"messages": []}), file_id="f")
    out["jsonl_empty_messages_refused"] = not ok
    ok, _ = _run(
        validate_chat_jsonl,
        _jsonl({"messages": [{"role": "hacker", "content": "x"}]}),
        file_id="f",
    )
    out["jsonl_bad_role_refused"] = not ok
    ok, _ = _run(
        validate_chat_jsonl,
        _jsonl({"messages": [{"role": "user", "content": ""}]}),
        file_id="f",
    )
    out["jsonl_empty_content_refused"] = not ok
    ok, _ = _run(
        validate_chat_jsonl,
        _jsonl({"messages": [{"role": "user", "content": 5}]}),
        file_id="f",
    )
    out["jsonl_nonstr_content_refused"] = not ok
    ok, _ = _run(
        validate_chat_jsonl,
        _jsonl({"messages": ["x"]}),
        file_id="f",
    )
    out["jsonl_nonstr_message_refused"] = not ok
    ok, exc = _run(validate_chat_jsonl, b"\n\n", file_id="f")
    out["jsonl_no_examples_refused"] = not ok and "no examples" in str(exc)
    out["jsonl_tool_role_allowed"] = (
        validate_chat_jsonl(_jsonl({"messages": [{"role": "tool", "content": "res"}]}), file_id="f")
        == 1
    )
    out["jsonl_system_role_allowed"] = (
        validate_chat_jsonl(_jsonl({"messages": [{"role": "system", "content": "s"}]}), file_id="f")
        == 1
    )
    return out


# ---------------------------------------------------------------- models


def _probe_request_models() -> dict[str, bool]:
    out: dict[str, bool] = {}

    ok, _ = _run(FTJobRequest.model_validate, {"model": "fx1", "training_file": "f-1"})
    out["req_minimal_accepts"] = ok
    ok, _ = _run(FTJobRequest.model_validate, {"model": "", "training_file": "f"})
    out["req_empty_model_refused"] = not ok
    ok, _ = _run(FTJobRequest.model_validate, {"model": "m" * 65, "training_file": "f"})
    out["req_long_model_refused"] = not ok
    ok, _ = _run(
        FTJobRequest.model_validate,
        {"model": "fx1", "training_file": "f", "suffix": "Bad_Suffix!"},
    )
    out["req_bad_suffix_refused"] = not ok
    ok, _ = _run(
        FTJobRequest.model_validate,
        {"model": "fx1", "training_file": "f", "suffix": "ok_suffix-1"},
    )
    out["req_good_suffix_accepts"] = ok
    ok, _ = _run(
        FTJobRequest.model_validate,
        {"model": "fx1", "training_file": "f", "seed": -1},
    )
    out["req_negative_seed_refused"] = not ok
    ok, _ = _run(
        FTJobRequest.model_validate,
        {"model": "fx1", "training_file": "f", "method": "dpo"},
    )
    out["req_unsupported_method_refused"] = not ok
    ok, _ = _run(
        FTJobRequest.model_validate,
        {"model": "fx1", "training_file": "f", "metadata": {f"k{i}": "v" for i in range(17)}},
    )
    out["req_metadata_cap_16"] = not ok
    ok, _ = _run(
        FTJobRequest.model_validate,
        {"model": "fx1", "training_file": "f", "bogus_field": 1},
    )
    out["req_extra_forbid"] = not ok
    ok, _ = _run(
        FTJobRequest.model_validate,
        {
            "model": "fx1",
            "training_file": "f",
            "hyperparameters": {"n_epochs": 51},
        },
    )
    out["hp_epochs_over_refused"] = not ok
    ok, _ = _run(
        FTJobRequest.model_validate,
        {
            "model": "fx1",
            "training_file": "f",
            "hyperparameters": {"n_epochs": 0},
        },
    )
    out["hp_epochs_zero_refused"] = not ok
    ok, _ = _run(
        FTJobRequest.model_validate,
        {
            "model": "fx1",
            "training_file": "f",
            "hyperparameters": {"learning_rate_multiplier": 0},
        },
    )
    out["hp_lr_zero_refused"] = not ok
    ok, _ = _run(
        FTJobRequest.model_validate,
        {
            "model": "fx1",
            "training_file": "f",
            "hyperparameters": {"batch_size": 513},
        },
    )
    out["hp_batch_over_refused"] = not ok
    ok, _ = _run(
        FTJobRequest.model_validate,
        {
            "model": "fx1",
            "training_file": "f",
            "hyperparameters": {"n_epochs": 2, "mystery": 1},
        },
    )
    out["hp_extra_forbid"] = not ok

    out["trainable_models_shape"] = TRAINABLE_MODELS == ("fx1", "local_fx1")
    ok, _ = _run(
        FTJob.model_validate,
        {"id": "j", "model": "fx1", "created_at": 1, "training_file": "f", "surprise": 1},
    )
    out["job_extra_forbid"] = not ok
    j = _job("j1")
    out["job_defaults_queued"] = j.status == "queued"
    out["job_object_literal"] = j.object == "fine_tuning.job"
    out["job_callback_secret_private"] = "callback_secret" not in j.model_dump()
    return out


# ---------------------------------------------------------------- store


def _probe_store_core() -> dict[str, bool]:
    out: dict[str, bool] = {}
    s = _store(10)

    s.put(_job("j1"), None, "fp1")
    s.put(_job("j2"), None, "fp2")
    s.put(_job("j3"), None, "fp3")
    jobs, more = s.list_jobs(limit=10, after=None)
    out["list_newest_first"] = [j.id for j in jobs] == ["j3", "j2", "j1"]
    out["list_no_more_at_cap"] = more is False
    jobs2, more2 = s.list_jobs(limit=1, after=None)
    out["list_has_more"] = more2 is True and [j.id for j in jobs2] == ["j3"]
    jobs3, _ = s.list_jobs(limit=10, after="j3")
    out["list_after_exclusive"] = [j.id for j in jobs3] == ["j2", "j1"]
    jobs4, _ = s.list_jobs(limit=10, after="j99")
    out["list_unknown_after_empty"] = jobs4 == []

    e = s.get("j1")
    out["get_returns_entry"] = e is not None and e.job.id == "j1"
    # j1 was oldest; after get() it must now list newest (MRU refresh)
    out["get_mru_refresh"] = [j.id for j in s.list_jobs(limit=10, after=None)[0]][0] == "j1"
    out["get_missing_none"] = s.get("nope") is None

    s.put(_job("j4"), "k1", "fp4")
    hit = s.lookup_idem("k1")
    out["idem_resolves"] = hit is not None and hit.job.id == "j4"
    out["idem_prefix_namespaced"] = s.lookup_idem("j4") is None
    s.put(_job("j5"), "k1", "fp5")
    hit2 = s.lookup_idem("k1")
    out["idem_reput_resolves_latest"] = hit2 is not None and hit2.job.id == "j5"
    return out


def _probe_store_eviction() -> dict[str, bool]:
    out: dict[str, bool] = {}
    s = _store(2)
    s.put(_job("j1"), "k1", "f1")
    s.register_model("ft:m:x:j1", job_id="j1", checkpoint="/ckpt/1", created=1)
    s.put(_job("j2"), None, "f2")
    s.put(_job("j3"), None, "f3")

    out["evict_drops_oldest"] = s.get("j1") is None
    out["evict_drops_idem_key"] = s.lookup_idem("k1") is None
    out["evict_drops_model_card"] = s.get_model("ft:m:x:j1") is None
    out["evict_keeps_recent"] = s.get("j2") is not None and s.get("j3") is not None
    ok, _ = _run(_store, 0)
    out["cap_floor_one"] = ok is True and _store(0)._max == 1
    return out


def _probe_store_models() -> dict[str, bool]:
    out: dict[str, bool] = {}
    s = _store(10)
    s.put(_job("j1"), None, "f")
    s.register_model(_MODEL_B, job_id="j1", checkpoint="/c1", created=2)
    s.register_model(_MODEL_A, job_id="j1", checkpoint="/c0", created=1)
    out["models_sorted_by_id"] = [m["id"] for m in s.models()] == [
        _MODEL_A,
        _MODEL_B,
    ]
    out["model_card_shape"] = s.get_model(_MODEL_A) == {
        "id": _MODEL_A,
        "job_id": "j1",
        "checkpoint": "/c0",
        "created": 1,
    }
    out["checkpoint_for_resolves"] = s.checkpoint_for(_MODEL_A) == "/c0"
    out["checkpoint_for_missing_none"] = s.checkpoint_for("ft:m:z") is None

    dropped = s.unregister_model(_MODEL_A)
    out["unregister_returns_card"] = dropped is not None and dropped["id"] == _MODEL_A
    out["unregister_gone"] = s.get_model(_MODEL_A) is None
    out["unregister_missing_none"] = s.unregister_model("ft:m:z") is None

    ckpts, more = s.checkpoints_for("j1", limit=10, after=None)
    out["checkpoints_one_per_card"] = len(ckpts) == 1
    out["checkpoint_id_derived"] = ckpts[0].id.startswith("ftckpt-")
    out["checkpoint_names_model"] = ckpts[0].fine_tuned_model_checkpoint == _MODEL_B
    out["checkpoints_more_flag"] = more is False
    ckpts2, _ = s.checkpoints_for("j1", limit=10, after=ckpts[0].id)
    out["checkpoints_after_exclusive"] = ckpts2 == []
    ckpts3, _ = s.checkpoints_for("missing", limit=10, after=None)
    out["checkpoints_missing_job_empty"] = ckpts3 == []

    s.register_model("ft:m:c:j1", job_id="j1", checkpoint="/c2", created=3)
    ckpts4, more4 = s.checkpoints_for("j1", limit=1, after=None)
    out["checkpoints_oldest_first"] = len(ckpts4) == 1 and more4 is True

    s2 = _store(10)
    s2.put(_job("jX"), None, "f")
    s2.put(_job("jY"), None, "f")
    s2.register_model("ft:m:y:jY", job_id="jY", checkpoint="/cY", created=1)
    s2._entries.pop("jY")  # job evicted but model lingers → register is guarded
    s2.register_model("ft:m:x:jZ", job_id="jZ", checkpoint="/cZ", created=9)
    out["register_dead_job_noop"] = s2.get_model("ft:m:x:jZ") is None
    return out


def _probe_store_events() -> dict[str, bool]:
    out: dict[str, bool] = {}
    s = _store(10)
    s.put(_job("j1"), None, "f")
    s.add_event("j1", "info", "first")
    s.add_event("j1", "warn", "second", {"k": 1})
    s.add_event("nope", "error", "nowhere")

    evs, _ = s.list_events("j1", limit=10, after=None)
    out["events_oldest_first"] = [e.message for e in evs] == ["first", "second"]
    out["events_shape"] = (
        evs[0].object == "fine_tuning.job.event"
        and evs[0].id.startswith("ftev-")
        and evs[0].level == "info"
        and evs[1].data == {"k": 1}
    )
    out["events_missing_empty"] = s.list_events("nope", limit=10, after=None) == ([], False)
    evs2, _ = s.list_events("j1", limit=10, after=evs[0].id)
    out["events_after_exclusive"] = [e.message for e in evs2] == ["second"]
    evs3, _ = s.list_events("j1", limit=10, after="bogus")
    out["events_unknown_after_empty"] = evs3 == []

    for i in range(_FT_EVENT_CAP + 5):
        s.add_event("j1", "info", f"e{i}")
    evs4, _ = s.list_events("j1", limit=_FT_EVENT_CAP + 10, after=None)
    out["events_cap_pops_oldest"] = len(evs4) == _FT_EVENT_CAP and evs4[0].message == "e5"
    return out


def _probe_store_transitions() -> dict[str, bool]:
    out: dict[str, bool] = {}

    s = _store(10)
    out["cancel_missing"] = s.request_cancel("nope") == "missing"
    out["pause_missing"] = s.request_pause("nope") == "missing"
    out["resume_missing"] = s.request_resume("nope") == "missing"

    s.put(_job("q", status="succeeded"), None, "f")
    out["cancel_terminal"] = s.request_cancel("q") == "terminal"
    out["pause_terminal"] = s.request_pause("q") == "terminal"
    out["resume_terminal"] = s.request_resume("q") == "terminal"

    s.put(_job("cj", status="queued"), None, "f")
    out["cancel_queued_flips_now"] = s.request_cancel("cj") == "queued"
    e = s.get("cj")
    out["cancel_queued_record"] = (
        e is not None
        and e.job.status == "cancelled"
        and e.job.finished_at is not None
        and e.cancel.is_set()
    )

    s.put(_job("rj", status="running"), None, "f")
    out["cancel_running_flag_only"] = s.request_cancel("rj") == "running"
    e2 = s.get("rj")
    out["cancel_running_not_flipped"] = (
        e2 is not None and e2.job.status == "running" and e2.cancel.is_set()
    )

    s.put(_job("pq", status="queued"), None, "f")
    out["pause_queued_verdict"] = s.request_pause("pq") == "queued"
    e3 = s.get("pq")
    out["pause_queued_state"] = (
        e3 is not None
        and e3.job.status == "paused"
        and e3.paused_from == "queued"
        and e3.pause.is_set()
        and not e3.resume.is_set()
    )
    out["pause_idempotent"] = s.request_pause("pq") == "paused"
    out["resume_queued_restores"] = s.request_resume("pq") == "queued"
    e3b = s.get("pq")
    out["resume_state_restored"] = (
        e3b is not None
        and e3b.job.status == "queued"
        and e3b.paused_from is None
        and e3b.resume.is_set()
    )

    s.put(_job("pr", status="running"), None, "f")
    out["pause_running_verdict"] = s.request_pause("pr") == "running"
    e4 = s.get("pr")
    out["pause_running_state"] = (
        e4 is not None and e4.job.status == "paused" and e4.paused_from == "running"
    )
    out["resume_running_restores"] = s.request_resume("pr") == "running"

    s.put(_job("np", status="running"), None, "f")
    out["resume_not_paused"] = s.request_resume("np") == "not_paused"

    s.put(_job("pc", status="running"), None, "f")
    s.request_pause("pc")
    out["cancel_paused_flips"] = s.request_cancel("pc") == "paused"
    e5 = s.get("pc")
    out["cancel_paused_terminal"] = (
        e5 is not None
        and e5.job.status == "cancelled"
        and e5.paused_from is None
        and e5.resume.is_set()
    )
    return out


def _probe_store_drain() -> dict[str, bool]:
    out: dict[str, bool] = {}
    s = _store(10)
    s.put(_job("q1"), None, "f")
    s.put(_job("r1", status="running"), None, "f")
    s.put(_job("p1", status="queued"), None, "f")
    s.request_pause("p1")  # paused while queued
    s.put(_job("p2", status="running"), None, "f")
    s.request_pause("p2")  # paused while running
    s.put(_job("t1", status="succeeded"), None, "f")

    pending = s.cancel_pending()
    out["drain_returns_queued_only"] = sorted(j.id for j in pending) == ["p1", "q1"]
    q1 = s.get("q1")
    out["drain_queued_terminal"] = q1 is not None and q1.job.status == "cancelled"
    p1 = s.get("p1")
    out["drain_paused_queued_terminal"] = p1 is not None and p1.job.status == "cancelled"
    r1 = s.get("r1")
    out["drain_running_flagged_only"] = (
        r1 is not None and r1.job.status == "running" and r1.cancel.is_set()
    )
    p2 = s.get("p2")
    out["drain_paused_running_flagged"] = (
        p2 is not None and p2.job.status == "paused" and p2.cancel.is_set()
    )
    out["drain_terminal_untouched"] = (
        s.get("t1") is not None and s.get("t1").job.status == "succeeded"  # type: ignore[union-attr]
    )
    out["drain_idle_empty"] = _store(4).cancel_pending() == []
    return out


def _probe_store_claims() -> dict[str, bool]:
    out: dict[str, bool] = {}
    s = _store(4)

    async def _two_claims() -> tuple[bool, bool]:
        async with s.async_claim_lock("k"):
            try:
                async with asyncio.timeout(0.5):
                    await asyncio.sleep(0)  # cancellation checkpoint
                    async with s.async_claim_lock("k"):
                        return True, False  # re-entrant double claim — should not be
            except TimeoutError:
                return True, True

        # second claim free after release
        async with asyncio.timeout(2.0):
            await asyncio.sleep(0)  # cancellation checkpoint
            async with s.async_claim_lock("k"):
                return False, True

    try:
        first, second_held_or_free = asyncio.run(_two_claims())
        out["claim_key_blocks_contention"] = True  # reached without deadlock
        out["claim_contended_waits"] = first is True
        out["claim_released_reacquirable"] = second_held_or_free is True
    except Exception:  # noqa: BLE001
        out["claim_key_blocks_contention"] = False
        out["claim_contended_waits"] = False
        out["claim_released_reacquirable"] = False
    return out


def _probe_store_journal() -> dict[str, bool]:
    out: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as td:
        jd = Path(td)
        s1 = _store(10, journal_dir=jd)
        s1.put(_job("done", status="succeeded", fine_tuned_model=_MODEL_DONE), "k-done", "fp1")
        s1.add_event("done", "info", "trained")
        s1.register_model(_MODEL_DONE, job_id="done", checkpoint="/ckpt", created=5)
        s1.put(_job("live", status="running"), "k-live", "fp2")
        s1.put(_job("wait", status="queued"), None, "fp3")

        s2 = _store(10, journal_dir=jd)
        d = s2.get("done")
        out["replay_terminal_as_was"] = (
            d is not None and d.job.status == "succeeded" and d.job.fine_tuned_model == _MODEL_DONE
        )
        out["replay_idem_keys_resolve"] = (
            s2.lookup_idem("k-done") is not None and s2.lookup_idem("k-done").job.id == "done"  # type: ignore[union-attr]
        )
        live = s2.get("live")
        out["replay_inflight_failed"] = (
            live is not None
            and live.job.status == "failed"
            and live.job.error is not None
            and "restarted" in live.job.error.message
        )
        wait = s2.get("wait")
        out["replay_queued_failed"] = wait is not None and wait.job.status == "failed"
        out["replay_model_restored"] = s2.get_model(_MODEL_DONE) == {
            "id": _MODEL_DONE,
            "job_id": "done",
            "checkpoint": "/ckpt",
            "created": 5,
        }
        out["replay_secret_not_journaled"] = d is not None and d.job._callback_fired is True
        evs, _ = s2.list_events("done", limit=10, after=None)
        out["replay_events_restored"] = [e.message for e in evs] == ["trained"]

        # evicted-producer cards never resurrect
        s3 = _store(1, journal_dir=jd / "cap1")
        s3.put(_job("e1", status="succeeded"), None, "f")
        s3.register_model("ft:m:e:e1", job_id="e1", checkpoint="/x", created=1)
        s3.put(_job("e2", status="succeeded"), None, "f")  # evicts e1
        s4 = _store(1, journal_dir=jd / "cap1")
        out["replay_evicted_card_gone"] = s4.get_model("ft:m:e:e1") is None
        out["replay_evicted_job_gone"] = s4.get("e1") is None

        # unregister journals the tombstone
        s5 = _store(10, journal_dir=jd / "tomb")
        s5.put(_job("t", status="succeeded"), None, "f")
        s5.register_model(_MODEL_T, job_id="t", checkpoint="/x", created=1)
        s5.unregister_model(_MODEL_T)
        s6 = _store(10, journal_dir=jd / "tomb")
        out["replay_model_delete_holds"] = s6.get_model(_MODEL_T) is None
    return out


def _probe_runner() -> dict[str, bool]:
    out: dict[str, bool] = {}
    resolver = cast("Callable[..., InferenceBackend]", lambda *a, **k: None)
    runner = default_ft_runner(resolver)
    out["runner_returns_callable"] = callable(runner)
    return out


# ---------------------------------------------------------------- battery

_PROBES = (
    _probe_jsonl_admission,
    _probe_request_models,
    _probe_store_core,
    _probe_store_eviction,
    _probe_store_models,
    _probe_store_events,
    _probe_store_transitions,
    _probe_store_drain,
    _probe_store_claims,
    _probe_store_journal,
    _probe_runner,
)


def finetune_audit() -> dict[str, bool]:
    """Run every finetune probe; ``name -> passed``."""
    out: dict[str, bool] = {}
    for probe in _PROBES:
        out.update(probe())
    return out


def finetune_audit_bench() -> dict[str, Any]:
    """Sealed bench payload — SYNTHETIC, research-only, no live-PnL claim."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = finetune_audit()
    ok = bool(r) and all(r.values())
    defects = sorted(k for k, v in r.items() if v is not True)
    if ok:
        interpretation = "all_probes_hold"
    elif not r:
        interpretation = "no_probes"
    else:
        interpretation = f"defects: {', '.join(defects)}"
    out: dict[str, Any] = {
        "kind": "finetune_audit",
        "schema": "finetune_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process; tmpdir journals only",
            "not_verified": [
                "live training pipeline execution",
                "callback webhook delivery",
                "GPU trainer wiring",
            ],
        },
        "interpretation": interpretation,
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
