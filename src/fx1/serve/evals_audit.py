"""evals_audit — adversarial probes on the ``/v1/evals`` deep surface.

The claim under test: the OpenAI-shaped eval API (``POST /v1/evals`` spec
containers, ``POST /v1/evals/{id}/runs`` run submissions) is an honest
submission-and-grading gate over the journaled ``/harness/evals`` engine —
every field that lands on the wire is either exercised end-to-end or
refused inside the ``{error:{message,type,param,code}}`` envelope, the
data-source vocabulary is closed (``type: "custom"`` only; there is no
``stored-completions``/``file_id``/``response_id`` source to leak through
item-schema extras), grading output is the suite's real verdict rows, and
submission semantics (idempotency, scope, drain, durability, metering)
hold under the same contract as the job surface.

This lane goes deeper than ``eval_lifecycle_audit`` (lane 145, run
lifecycle/cursor contract): submission validation boundaries, data-source
integrity, grading-output honesty, and the gate between the declared spec
and the executed run.

Coverage map:

- *Spec validation* — ``data_source_config`` is a closed ``custom``
  vocabulary: wrong type, missing ``item_schema``, unknown item fields,
  bad suite names, negative seeds, and self-repeating fallback chains all
  refuse at 422; ``metadata``/``name``/``testing_criteria`` round-trip and
  bound-check (name ≤255, criteria ≤64, metadata values are strings).
  Criteria carry arbitrary grader kwargs verbatim (``extra=allow``) —
  declarative intent stored on the spec, never silently executed or
  reinterpreted.
- *Spec lifecycle* — get/update/delete/list over the journaled spec
  store: name+metadata edits are durable, the datasource is frozen
  (update attempts 422), deletes tombstone, list is newest-first with
  fail-closed cursors.
- *Data-source integrity* — the run's ``data_source.source`` merges over
  the spec's item_schema and re-validates (unknown keys like ``file_id``
  /``response_id``/``store`` refuse 422 — the closed boundary stands in
  for the absent stored-completions surface); a suite override runs the
  *overridden* suite; zero-task reports (capability's gate shape) serve
  honest empty item pages rather than fabricating rows.
- *Run submit* — the spec-binding cell: a run mints ``evalrun_*`` wired
  to its spec id, the wire keeps the requested ``model`` while the
  record keeps the resolved ``backend`` link (``fx1`` → local_fx1,
  ``ft:<name>`` → the card's checkpoint, unregistered → 404), submit-time
  metadata rides the record + wire + sealed receipt, and idempotency
  dedupes per (credential, key, spec).
- *Run lifecycle* — status projection (queued/running → queued/
  in_progress; succeeded → completed; failed/cancelled → failed/
  canceled), ``result_counts`` only on completed runs and honest
  (``errored: 0`` — the suites never report errored), the ``error``
  block lands on failures with a real message, cancels refuse non-queued
  records with 409, and delete tombstones both the wire and harness
  surfaces.
- *Output items* — per-task verdict rows paged honestly: index-encoded
  cursors refuse foreign/malformed ids 400, items carry the suite's raw
  report row verbatim under ``datasource_item`` plus a suite-named
  verdict under ``results``, and a failed/zero-task run serves an empty
  page, never fabricated rows.
- *Scope/auth* — read covers GETs, write covers every mutation, scopes
  are literal; Bearer works on ``/v1`` while ``/harness`` requires
  ``X-API-Key``; non-admin keys are refused the control plane; the
  workspace is shared — a scoped peer can run, read, and delete another
  key's evals (pinned, matching tenancy_audit's matrix).
- *Idempotency* — replay returns the identical run id; a reused key with
  a different body fails 409; the spec-id namespace means the same key
  under a sibling eval is a distinct run; a 256-char key (the wire
  bound) works; 257 fails 400.
- *Durability* — ``--state-dir`` restart restores specs, run records,
  tombstones, and idem claims; a deleted spec stays deleted; output
  items recompute identically from the journaled report.
- *Concurrency* — parallel runs mint distinct ids under the spec claim
  lock; a same-key burst collapses to one run; concurrent output_item
  reads stay consistent; racing cancels land on one side of the CAS.
- *Drain* — run submissions refuse 503 ``draining`` once latched; spec
  creates (a pure registry write, no worker admission) stay allowed;
  in-flight runs finish and reads keep working under drain.
- *Webhook* — the terminal callback fires once, carries the sealed
  record (never the signing secret), and lands a signed
  ``X-Fx1-Webhook-Signature`` envelope; delivery failures mark
  ``callback_status=failed`` after bounded retries.
- *Metering* — eval ops bill the caller's ``uses``; run model calls
  meter under ``eval:{suite}:{backend}`` in the complete ledger; the
  eval path charges no per-key tokens.
- *SDK parity* — ``eval_spec_*``/``eval_run_*`` twins return the same
  wire shapes, bind the spec, and refuse missing specs/unregistered
  models with the SDK's mapped exceptions.

Defects found while pinning this battery, fixed on this branch:

1. ``POST /v1/evals/{id}/runs`` accepted ``metadata`` and then dropped
   it — the wire object, the journaled record, and the sealed receipt
   all omitted the caller's map. ``EvalRecord.eval_metadata`` now
   carries it end to end (wire ``metadata`` + receipt + SDK twin
   ``eval_run_create(metadata=…)``).
2. The run-submit Idempotency-Key bound was applied to the
   ``{key}:{spec_id}`` concatenation, not the raw header: legal keys of
   ~230–256 chars were refused 400 while the spec-namespacing duplicated
   what ``_idem_scope`` already provides. The spec id now rides the
   scope's ``namespace`` slot, so the full 256-char band works.
3. ``run_wire`` emitted a sparse dict — ``error`` and ``result_counts``
   were absent rather than ``null`` — so the SDK twin's
   ``eval_run_create`` returned a different shape than the
   ``EvalRunObject`` the API serves (both fields default to ``null``
   there). The dict now always carries both keys, nulled when the run
   state doesn't populate them.

Sealed ``evals_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fx1.serve.conv_audit import _RESOURCES, _audit_context, _temporary_directory
from fx1.serve.eval_lifecycle_audit import _Sink

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient

__all__ = ["evals_audit", "evals_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "k3y-material"
_MODEL = "byok"
_N = 8

_SPEC: dict[str, Any] = {
    "name": "audit-tooluse",
    "data_source_config": {
        "type": "custom",
        "item_schema": {"suite": "tooluse", "seed": 0},
    },
    "testing_criteria": [{"name": "suite-verdict", "type": "tooluse_pass"}],
    "metadata": {"lane": "162"},
}


# ---------------------------------------------------------------------------
# Stub backends + client plumbing
# ---------------------------------------------------------------------------


class _StubBackend:
    """Deterministic completion stub returning a bare float — parses for
    every suite grader without pretending to be a real model."""

    def __init__(self, model: str = "evals-stub-0") -> None:
        self._model = model
        self.last_usage = {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        self.calls = 0

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del sampling
        self.calls += 1
        return "0.5"

    def close(self) -> None:
        pass


class _GateBackend(_StubBackend):
    """Blocks ``complete`` on a gate — a parked mid-flight run the probe
    can cancel/drain/meter deterministically."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.entered = threading.Event()

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        self.entered.set()
        self.gate.wait(30)
        return super().complete(messages, sampling=sampling)


def _backends(gate: _GateBackend | None = None) -> dict[str, Any]:
    return {
        "hosted_k3": lambda **k: _StubBackend("hosted-stub"),
        "local_fx1": lambda **k: gate or _StubBackend("ckpt-stub"),
        "byok": lambda **k: gate or _StubBackend("byok-stub"),
    }


@contextmanager
def _api_key_env(api_key: str | None) -> Iterator[None]:
    import os  # noqa: PLC0415

    previous = os.environ.get(_API_KEY_ENV)
    if api_key is None:
        os.environ.pop(_API_KEY_ENV, None)
    else:
        os.environ[_API_KEY_ENV] = api_key
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = previous


def _ok_runner(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
    """Succeeded ft job minting a checkpoint dir for the card."""
    from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

    ckpt = spec.work_dir / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    return FTJobOutcome(
        fine_tuned_model=spec.ft_model_name,
        checkpoint=str(ckpt),
        trained_tokens=7,
    )


def _client(
    backend_map: dict[str, Any] | None = None,
    *,
    api_key: str | None = _ROOT,
    state_dir: Path | None = None,
    ft_runner: Any = None,
    store_max: int | None = None,
    max_inflight: int | None = None,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — isolated env per construction; the env
    root key is set only for the create_app window."""
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    isolated = _temporary_directory()
    receipts = isolated / "receipts"
    receipts.mkdir()
    with _api_key_env(api_key):
        app = api_mod.create_app(
            harness=Harness(runner=lambda argv, timeout_s: (0, "ok", "")),
            backend_resolver=lambda name, *a, **k: (backend_map or _backends())[name](**k),
            state_dir=state_dir if state_dir is not None else isolated / "state",
            receipts_dir=receipts,
            ft_runner=ft_runner if ft_runner is not None else _ok_runner,
            ft_dir=isolated / "fine_tuning",
            store_max=store_max,
            max_inflight=max_inflight,
        )
    resources = _RESOURCES.get()
    resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
    client = TestClient(app, raise_server_exceptions=False)
    resources.callback(client.close)
    resources.enter_context(client)
    return client, api_mod


# ---------------------------------------------------------------------------
# Wire helpers — every helper carries the env root unless told otherwise
# ---------------------------------------------------------------------------


def _h(auth: str | None) -> dict[str, str]:
    return {"X-API-Key": auth} if auth else {}


def _bearer(auth: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {auth}"}


def _mint(client: TestClient, **policy: Any) -> tuple[str, str]:
    r = client.post("/harness/keys", json=policy, headers=_h(_ROOT))
    assert r.status_code == 201, r.text
    return str(r.json()["key"]), str(r.json()["id"])


def _spec(client: TestClient, auth: str | None = _ROOT, **over: Any) -> dict[str, Any]:
    """Create an eval spec; ``over`` merges over the default body."""
    body = json.loads(json.dumps(_SPEC))
    body.update(over)
    r = client.post("/v1/evals", json=body, headers=_h(auth))
    assert r.status_code == 201, r.text
    out: dict[str, Any] = r.json()
    return out


def _spec_id(client: TestClient, auth: str | None = _ROOT, **over: Any) -> str:
    return str(_spec(client, auth, **over)["id"])


def _bare(run_id: str) -> str:
    return run_id.removeprefix("evalrun_")


def _run_submit(
    client: TestClient,
    spec_id: str,
    auth: str | None = _ROOT,
    extra_headers: dict[str, str] | None = None,
    **body: Any,
) -> Any:
    hdrs = {**_h(auth), **(extra_headers or {})}
    payload = {"model": _MODEL, **body}
    return client.post(f"/v1/evals/{spec_id}/runs", json=payload, headers=hdrs)


def _run_get(client: TestClient, spec_id: str, run_id: str, auth: str | None = _ROOT) -> Any:
    return client.get(f"/v1/evals/{spec_id}/runs/{run_id}", headers=_h(auth))


def _run_items(
    client: TestClient, spec_id: str, run_id: str, auth: str | None = _ROOT, **p: Any
) -> Any:
    return client.get(
        f"/v1/evals/{spec_id}/runs/{run_id}/output_items",
        params=p,
        headers=_h(auth),
    )


def _rec(client: TestClient, run_id: str) -> dict[str, Any]:
    """The journaled harness record behind an ``evalrun_`` wire id."""
    r = client.get(f"/harness/evals/{_bare(run_id)}", headers=_h(_ROOT))
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()
    return out


def _wait_rec(client: TestClient, run_id: str, timeout_s: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        rec = _rec(client, run_id)
        if rec["status"] in ("succeeded", "failed", "cancelled"):
            return rec
        assert time.monotonic() < deadline, f"run {run_id} never went terminal"
        time.sleep(0.02)


def _wait_wire(
    client: TestClient, spec_id: str, run_id: str, timeout_s: float = 20.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        r = _run_get(client, spec_id, run_id)
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] in ("completed", "failed", "canceled"):
            return dict(body)
        assert time.monotonic() < deadline, f"run {run_id} never went wire-terminal"
        time.sleep(0.02)


def _wait_running_rec(client: TestClient, run_id: str, timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    while _rec(client, run_id)["status"] != "running":
        assert time.monotonic() < deadline, f"run {run_id} never ran"
        time.sleep(0.01)


def _err(resp: Any) -> dict[str, Any]:
    body = resp.json()
    err = body.get("error")
    return dict(err) if isinstance(err, dict) else {}


def _err_code(resp: Any) -> str:
    return str(_err(resp).get("code") or "")


def _err_type(resp: Any) -> str:
    return str(_err(resp).get("type") or "")


def _items_all(client: TestClient, spec_id: str, run_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    after: str | None = None
    while True:
        page = _run_items(client, spec_id, run_id, limit=100, **({"after": after} if after else {}))
        assert page.status_code == 200, page.text
        body = page.json()
        out.extend(body["data"])
        if not body["has_more"]:
            return out
        after = body["last_id"]


def _wait_for(pred: Any, *, timeout_s: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _raises(fn: Any) -> str:
    """Exception class name; "" when no raise."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 — probe captures the class
        return type(exc).__name__
    return ""


# ---------------------------------------------------------------------------
# Spec create / lifecycle
# ---------------------------------------------------------------------------


def _probe_spec_create(results: dict[str, bool]) -> None:
    client, _api = _client()
    spec = _spec(client)

    results["create_201"] = True  # _spec asserts
    results["create_id_shape"] = bool(re.fullmatch(r"eval_[0-9a-f]{24}", spec["id"]))
    results["create_object_eval"] = spec["object"] == "eval"
    results["create_name_roundtrip"] = spec["name"] == _SPEC["name"]
    results["create_metadata_roundtrip"] = spec["metadata"] == _SPEC["metadata"]
    results["create_criteria_roundtrip"] = spec["testing_criteria"] == _SPEC["testing_criteria"]
    results["create_config_normalized"] = spec["data_source_config"]["type"] == "custom" and {
        "suite",
        "seed",
        "fallbacks",
    } <= set(spec["data_source_config"]["item_schema"])
    results["create_created_at_int"] = isinstance(spec["created_at"], int)

    # Duplicate creates mint distinct spec ids — specs are containers, not
    # content-addressed records.
    other = _spec(client)
    results["create_duplicate_allowed_distinct_id"] = other["id"] != spec["id"]

    # Criteria are declarative: extra grader kwargs store verbatim.
    rich = _spec(
        client,
        testing_criteria=[
            {
                "name": "thresh",
                "type": "score_threshold",
                "threshold": 0.9,
                "labels": ["a", "b"],
                "custom": {"nested": 1},
            }
        ],
    )
    results["criteria_extra_kwargs_stored_verbatim"] = (
        rich["testing_criteria"][0]["threshold"] == 0.9
        and rich["testing_criteria"][0]["labels"] == ["a", "b"]
        and rich["testing_criteria"][0]["custom"] == {"nested": 1}
    )

    # The closed boundary — every malformed body lands a 422 envelope.
    def bad(body: Any) -> Any:
        return client.post("/v1/evals", json=body, headers=_h(_ROOT))

    results["create_empty_body_422"] = bad({}).status_code == 422
    results["create_missing_name_422"] = (
        bad({"data_source_config": _SPEC["data_source_config"]}).status_code == 422
    )
    results["create_empty_name_422"] = bad({**_SPEC, "name": ""}).status_code == 422
    results["create_long_name_422"] = bad({**_SPEC, "name": "x" * 256}).status_code == 422
    results["create_name_ok_at_255"] = bad({**_SPEC, "name": "x" * 255}).status_code == 201
    r_noncustom = bad(
        {
            **_SPEC,
            "data_source_config": {
                "type": "stored_completions",
                "item_schema": _SPEC["data_source_config"]["item_schema"],
            },
        }
    )
    results["create_noncustom_config_422"] = r_noncustom.status_code == 422
    results["create_noncustom_config_envelope"] = (
        _err_code(r_noncustom) == "validation" and _err_type(r_noncustom) == "invalid_request_error"
    )
    results["create_missing_item_schema_422"] = (
        bad({**_SPEC, "data_source_config": {"type": "custom"}}).status_code == 422
    )
    results["create_item_schema_extra_forbid_422"] = (
        bad(
            {
                **_SPEC,
                "data_source_config": {
                    "type": "custom",
                    "item_schema": {"suite": "tooluse", "file_id": "file-x"},
                },
            }
        ).status_code
        == 422
    )
    results["create_item_schema_byok_field_422"] = (
        bad(
            {
                **_SPEC,
                "data_source_config": {
                    "type": "custom",
                    "item_schema": {
                        "suite": "tooluse",
                        "byok": {"base_url": "http://x", "api_key": "k", "model": "m"},
                    },
                },
            }
        ).status_code
        == 422
    )
    results["create_bad_suite_422"] = (
        bad(
            {
                **_SPEC,
                "data_source_config": {
                    "type": "custom",
                    "item_schema": {"suite": "ghost"},
                },
            }
        ).status_code
        == 422
    )
    results["create_negative_seed_422"] = (
        bad(
            {
                **_SPEC,
                "data_source_config": {
                    "type": "custom",
                    "item_schema": {"suite": "tooluse", "seed": -1},
                },
            }
        ).status_code
        == 422
    )
    results["create_fallback_repeats_backend_422"] = (
        bad(
            {
                **_SPEC,
                "data_source_config": {
                    "type": "custom",
                    "item_schema": {
                        "suite": "tooluse",
                        "backend": "byok",
                        "fallbacks": ["byok"],
                    },
                },
            }
        ).status_code
        == 422
    )
    results["create_criteria_over_64_422"] = (
        bad({**_SPEC, "testing_criteria": [{"name": f"c{i}"} for i in range(65)]}).status_code
        == 422
    )
    results["create_criterion_nameless_422"] = (
        bad({**_SPEC, "testing_criteria": [{"type": "x"}]}).status_code == 422
    )
    results["create_criterion_unknown_type_stored"] = (
        bad({**_SPEC, "testing_criteria": [{"name": "c", "type": "bogus-grader"}]}).status_code
        == 201
    )
    results["create_metadata_non_str_422"] = bad({**_SPEC, "metadata": {"n": 5}}).status_code == 422
    results["create_extra_field_422"] = bad({**_SPEC, "bogus": 1}).status_code == 422
    # judge_backend is bound to the judge suites at spec-validate — a
    # non-judge suite carrying one refuses 422, not at run submit.
    r_judge_bad = bad(
        {
            **_SPEC,
            "data_source_config": {
                "type": "custom",
                "item_schema": {"suite": "tooluse", "judge_backend": "byok"},
            },
        }
    )
    results["create_judge_on_nonjudge_suite_422"] = r_judge_bad.status_code == 422
    results["create_judge_on_nonjudge_suite_envelope"] = _err_code(
        r_judge_bad
    ) == "validation" and "judge" in str(_err(r_judge_bad).get("message", ""))
    results["create_judge_on_judge_suite_ok"] = (
        bad(
            {
                **_SPEC,
                "data_source_config": {
                    "type": "custom",
                    "item_schema": {"suite": "capability", "judge_backend": "byok"},
                },
            }
        ).status_code
        == 201
    )


def _probe_spec_lifecycle(results: dict[str, bool]) -> None:
    client, _api = _client()
    a = _spec(client, name="spec-a")
    b = _spec(client, name="spec-b")

    g = client.get(f"/v1/evals/{a['id']}", headers=_h(_ROOT))
    results["get_200_shape"] = g.status_code == 200 and g.json()["name"] == "spec-a"
    results["get_missing_404"] = (
        client.get("/v1/evals/eval_ghost", headers=_h(_ROOT)).status_code == 404
        and _err_code(client.get("/v1/evals/eval_ghost", headers=_h(_ROOT))) == "eval_not_found"
    )

    upd = client.post(f"/v1/evals/{a['id']}", json={"name": "spec-a2"}, headers=_h(_ROOT))
    results["update_name_200"] = upd.status_code == 200 and upd.json()["name"] == "spec-a2"
    upd2 = client.post(
        f"/v1/evals/{a['id']}", json={"metadata": {"only": "new"}}, headers=_h(_ROOT)
    )
    results["update_metadata_replaces"] = upd2.json()["metadata"] == {"only": "new"}
    results["update_keeps_datasource"] = upd2.json()["data_source_config"]["type"] == "custom"
    results["update_keeps_criteria"] = upd2.json()["testing_criteria"] == _SPEC["testing_criteria"]
    results["update_requires_field_422"] = (
        client.post(f"/v1/evals/{a['id']}", json={}, headers=_h(_ROOT)).status_code == 422
    )
    results["update_datasource_frozen_422"] = (
        client.post(
            f"/v1/evals/{a['id']}",
            json={"data_source_config": _SPEC["data_source_config"]},
            headers=_h(_ROOT),
        ).status_code
        == 422
    )
    results["update_criteria_frozen_422"] = (
        client.post(
            f"/v1/evals/{a['id']}",
            json={"testing_criteria": [{"name": "x"}]},
            headers=_h(_ROOT),
        ).status_code
        == 422
    )
    results["update_missing_404"] = (
        client.post("/v1/evals/eval_ghost", json={"name": "x"}, headers=_h(_ROOT)).status_code
        == 404
    )

    # List: newest-first, fail-closed cursors, bounded page size.
    page1 = client.get("/v1/evals", params={"limit": 1}, headers=_h(_ROOT))
    body1 = page1.json()
    results["list_200"] = page1.status_code == 200
    results["list_newest_first"] = body1["data"][0]["id"] == b["id"]
    results["list_page_shape"] = body1["object"] == "list" and body1["has_more"] is True
    page2 = client.get("/v1/evals", params={"limit": 1, "after": b["id"]}, headers=_h(_ROOT))
    results["list_cursor_chains"] = page2.json()["data"][0]["id"] == a["id"]
    results["list_limit_0_400"] = (
        client.get("/v1/evals", params={"limit": 0}, headers=_h(_ROOT)).status_code == 400
    )
    results["list_limit_101_400"] = (
        client.get("/v1/evals", params={"limit": 101}, headers=_h(_ROOT)).status_code == 400
    )
    r_badcur = client.get("/v1/evals", params={"after": "eval_bogus"}, headers=_h(_ROOT))
    results["list_bad_cursor_400"] = (
        r_badcur.status_code == 400 and _err_code(r_badcur) == "invalid_cursor"
    )

    # Delete: tombstone the spec; subsequent verbs 404.
    d = client.delete(f"/v1/evals/{b['id']}", headers=_h(_ROOT))
    results["delete_200_shape"] = (
        d.status_code == 200
        and d.json()["object"] == "eval.deleted"
        and d.json()["deleted"] is True
        and d.json()["id"] == b["id"]
    )
    results["delete_then_get_404"] = (
        client.get(f"/v1/evals/{b['id']}", headers=_h(_ROOT)).status_code == 404
    )
    results["delete_twice_404"] = (
        client.delete(f"/v1/evals/{b['id']}", headers=_h(_ROOT)).status_code == 404
    )
    results["list_excludes_deleted"] = all(
        s["id"] != b["id"] for s in client.get("/v1/evals", headers=_h(_ROOT)).json()["data"]
    )
    results["update_deleted_404"] = (
        client.post(f"/v1/evals/{b['id']}", json={"name": "x"}, headers=_h(_ROOT)).status_code
        == 404
    )


# ---------------------------------------------------------------------------
# Run submit
# ---------------------------------------------------------------------------


def _probe_run_submit(results: dict[str, bool]) -> None:
    client, _api = _client()
    spec_id = _spec_id(client)

    r = _run_submit(client, spec_id)
    run = r.json()
    results["run_create_201"] = r.status_code == 201
    results["run_id_shape"] = bool(re.fullmatch(r"evalrun_[0-9a-f]{32}", run["id"]))
    results["run_object_eval_run"] = run["object"] == "eval.run"
    results["run_bound_to_spec"] = run["eval_id"] == spec_id
    results["run_model_recorded"] = run["model"] == _MODEL
    results["run_backend_resolved"] = run["backend"] == _MODEL
    results["run_suite_seed_from_spec"] = run["suite"] == "tooluse" and run["seed"] == 0
    results["run_metadata_empty_default"] = run["metadata"] == {}
    results["run_receipt_url_points_at_record"] = (
        run["receipt_url"] == f"/harness/evals/{_bare(run['id'])}/receipt"
    )
    loc = str(r.headers.get("location", ""))
    loc_get = client.get(loc, headers=_h(_ROOT))
    results["run_location_header_resolves"] = (
        loc.startswith(f"/v1/evals/{spec_id}/runs/")
        and loc_get.status_code == 200
        and loc_get.json()["id"] == run["id"]
    )
    results["run_ptcr_empty"] = run["per_testing_criteria_results"] == []

    # The record twin carries the binding.
    rec = _rec(client, run["id"])
    results["record_binds_spec"] = rec["eval_spec"] == spec_id
    results["record_binds_model"] = rec["eval_model"] == _MODEL
    results["record_metadata_absent_when_unsent"] = rec["eval_metadata"] is None

    # model mapping: fx1 → local_fx1 (no checkpoint configured → the run
    # fails closed, async and honestly).
    r_fx1 = _run_submit(client, spec_id, model="fx1")
    run_fx1 = r_fx1.json()
    results["model_fx1_maps_local_fx1"] = (
        r_fx1.status_code == 201 and run_fx1["model"] == "fx1" and run_fx1["backend"] == "local_fx1"
    )
    rec_fx1 = _wait_rec(client, run_fx1["id"])
    w_fx1 = _wait_wire(client, spec_id, run_fx1["id"])
    results["model_fx1_fails_closed_no_ckpt"] = (
        rec_fx1["status"] == "failed"
        and "checkpoint_dir" in str(rec_fx1.get("error", ""))
        and w_fx1["status"] == "failed"
        and w_fx1["error"]["code"] == "eval_run_failed"
        and w_fx1["result_counts"] is None
    )
    r_ft = _run_submit(client, spec_id, model="ft:ghost")
    results["model_ft_unregistered_404"] = (
        r_ft.status_code == 404 and _err_code(r_ft) == "model_not_found"
    )
    r_unk = _run_submit(client, spec_id, model="gpt-9")
    results["model_unknown_400"] = (
        r_unk.status_code == 400 and _err_code(r_unk) == "invalid_request"
    )
    results["model_empty_422"] = _run_submit(client, spec_id, model="").status_code == 422

    # Eval-scope binding: spec B cannot reach spec A's run in any verb.
    other_id = _spec_id(client)
    r_cross = _run_get(client, other_id, run["id"])
    results["run_get_cross_spec_404"] = (
        r_cross.status_code == 404 and _err_code(r_cross) == "run_not_found"
    )
    results["run_items_cross_spec_404"] = _run_items(client, other_id, run["id"]).status_code == 404
    results["run_cancel_cross_spec_404"] = (
        client.post(f"/v1/evals/{other_id}/runs/{run['id']}/cancel", headers=_h(_ROOT)).status_code
        == 404
    )
    results["run_delete_cross_spec_404"] = (
        client.delete(f"/v1/evals/{other_id}/runs/{run['id']}", headers=_h(_ROOT)).status_code
        == 404
    )
    results["runs_list_scoped_to_spec"] = all(
        rw["eval_id"] == spec_id
        for rw in client.get(
            f"/v1/evals/{spec_id}/runs", params={"limit": 100}, headers=_h(_ROOT)
        ).json()["data"]
    ) and all(
        rw["id"] != run["id"]
        for rw in client.get(
            f"/v1/evals/{other_id}/runs", params={"limit": 100}, headers=_h(_ROOT)
        ).json()["data"]
    )
    r_missing = _run_submit(client, "eval_ghost")
    results["run_submit_missing_spec_404"] = (
        r_missing.status_code == 404 and _err_code(r_missing) == "eval_not_found"
    )
    results["run_submit_deleted_spec_404"] = _probe_deleted_spec_submit(client)

    # data_source.source merges over the spec's item_schema and
    # re-validates — overrides change what the run executes.
    r_seed = _run_submit(client, spec_id, data_source={"type": "custom", "source": {"seed": 42}})
    results["datasource_seed_override_wired"] = (
        r_seed.status_code == 201
        and r_seed.json()["seed"] == 42
        and _rec(client, r_seed.json()["id"])["seed"] == 42
    )
    results["datasource_merge_keeps_spec_fields"] = (
        r_seed.status_code == 201 and r_seed.json()["suite"] == "tooluse"
    )
    r_suite = _run_submit(
        client, spec_id, data_source={"type": "custom", "source": {"suite": "capability"}}
    )
    rec_suite = _wait_rec(client, r_suite.json()["id"], timeout_s=60)
    results["datasource_suite_override_executes"] = (
        r_suite.status_code == 201
        and r_suite.json()["suite"] == "capability"
        and rec_suite.get("suite") == "capability"
        and rec_suite["status"] == "succeeded"
    )
    results["datasource_null_source_passthrough"] = (
        _run_submit(client, spec_id, data_source={"type": "custom"}).status_code == 201
    )
    results["datasource_non_custom_type_422"] = (
        _run_submit(
            client,
            spec_id,
            data_source={"type": "stored_completions", "source": {"seed": 1}},
        ).status_code
        == 422
    )
    # Stored-completion refs (file_id / response_id / store:false) hit the
    # item_schema's extra-forbid — the closed boundary refuses them since
    # no such data source exists.
    for ref in ("file_id", "response_id", "store"):
        results[f"datasource_{ref}_refused_422"] = (
            _run_submit(
                client,
                spec_id,
                data_source={"type": "custom", "source": {ref: "file-x"}},
            ).status_code
            == 422
        )
    results["datasource_bad_value_422"] = (
        _run_submit(
            client, spec_id, data_source={"type": "custom", "source": {"seed": -1}}
        ).status_code
        == 422
    )

    # Run-body validation layers.
    results["judge_byok_without_judge_422"] = (
        _run_submit(
            client,
            spec_id,
            judge_byok={"base_url": "http://x", "api_key": "k", "model": "m"},
        ).status_code
        == 422
    )
    results["callback_secret_without_url_422"] = (
        _run_submit(client, spec_id, callback_secret="s").status_code == 422
    )
    results["callback_url_non_http_422"] = (
        _run_submit(client, spec_id, callback_url="ftp://x").status_code == 422
    )
    results["run_metadata_non_str_422"] = (
        _run_submit(client, spec_id, metadata={"n": 5}).status_code == 422
    )

    # metadata flows end to end (fixed defect): wire create + get, the
    # journaled record, and the sealed receipt.
    meta = {"lane": "162", "purpose": "probe"}
    r_meta = _run_submit(client, spec_id, metadata=meta)
    mid = r_meta.json()["id"]
    wire_meta = _wait_wire(client, spec_id, mid)
    rec_meta = _rec(client, mid)
    results["run_metadata_roundtrips_create"] = r_meta.json()["metadata"] == meta
    results["run_metadata_roundtrips_get"] = wire_meta["metadata"] == meta
    results["run_metadata_on_record"] = rec_meta["eval_metadata"] == meta
    receipt = client.get(f"/harness/evals/{_bare(mid)}/receipt", headers=_h(_ROOT))
    doc = receipt.json()
    results["run_metadata_in_sealed_receipt"] = (
        receipt.status_code == 200
        and doc.get("schema") == "fx1_eval_record.v1"
        and doc.get("record", {}).get("eval_metadata") == meta
    )


def _probe_deleted_spec_submit(client: TestClient) -> bool:
    sid = _spec_id(client)
    client.delete(f"/v1/evals/{sid}", headers=_h(_ROOT))
    r = _run_submit(client, sid)
    return r.status_code == 404 and _err_code(r) == "eval_not_found"


def _probe_idempotency(results: dict[str, bool]) -> None:
    client, _api = _client()
    spec_id = _spec_id(client)

    # Same key + same body → the same run, not a duplicate.
    key = "idem-run-1"
    a = _run_submit(client, spec_id, extra_headers={"Idempotency-Key": key})
    b = _run_submit(client, spec_id, extra_headers={"Idempotency-Key": key})
    results["idem_replay_same_run_id"] = (
        a.status_code == 201 and b.status_code == 201 and a.json()["id"] == b.json()["id"]
    )
    page = client.get(f"/v1/evals/{spec_id}/runs", params={"limit": 100}, headers=_h(_ROOT)).json()
    results["idem_replay_still_one_record"] = (
        len([r for r in page["data"] if r["id"] == a.json()["id"]]) == 1
    )

    # Same key + a different body → 409 idempotency_conflict.
    diff = _run_submit(
        client,
        spec_id,
        data_source={"type": "custom", "source": {"seed": 9}},
        extra_headers={"Idempotency-Key": key},
    )
    results["idem_conflict_409"] = (
        diff.status_code == 409 and _err_code(diff) == "idempotency_conflict"
    )

    # The spec is the dedupe namespace: same key under a sibling spec mints
    # a distinct run.
    other = _spec_id(client)
    c = _run_submit(client, other, extra_headers={"Idempotency-Key": key})
    results["idem_scoped_per_spec"] = c.json()["id"] != a.json()["id"]

    # The bound is on the raw header (fixed defect): 256 works, 257 fails.
    k256 = "k" * 256
    r256 = _run_submit(client, spec_id, extra_headers={"Idempotency-Key": k256})
    results["idem_key_256_accepted"] = r256.status_code == 201
    r256b = _run_submit(client, spec_id, extra_headers={"Idempotency-Key": k256})
    results["idem_key_256_replays"] = r256b.json()["id"] == r256.json()["id"]
    r257 = _run_submit(client, spec_id, extra_headers={"Idempotency-Key": "k" * 257})
    results["idem_key_257_400"] = r257.status_code == 400 and _err_code(r257) == "bad_request"

    # Metadata-only difference still dedupes (the fingerprint covers the
    # harness submit body, not the wire metadata).
    meta_diff = _run_submit(
        client, spec_id, metadata={"x": "y"}, extra_headers={"Idempotency-Key": key}
    )
    results["idem_ignores_run_metadata"] = meta_diff.json()["id"] == a.json()["id"]


def _probe_run_lifecycle(results: dict[str, bool]) -> None:
    gate = _GateBackend()
    client, _api = _client(backend_map=_backends(gate))
    spec_id = _spec_id(client)

    # queued -> in_progress wire projection while the run is parked.
    r = _run_submit(client, spec_id)
    run_id = r.json()["id"]
    assert gate.entered.wait(15), "run never entered the backend"
    mid = _run_get(client, spec_id, run_id).json()
    results["status_in_progress_while_running"] = mid["status"] == "in_progress"
    results["midflight_result_counts_null"] = mid["result_counts"] is None
    results["midflight_items_empty"] = _run_items(client, spec_id, run_id).json()["data"] == []
    # Cancel on a running record is a pinned refusal, not a race gamble.
    cx = client.post(f"/v1/evals/{spec_id}/runs/{run_id}/cancel", headers=_h(_ROOT))
    results["cancel_running_409_conflict"] = cx.status_code == 409 and _err_code(cx) == "conflict"
    gate.gate.set()
    final = _wait_wire(client, spec_id, run_id)
    results["terminal_completed"] = final["status"] == "completed"
    results["completed_result_counts_shape"] = (
        set(final["result_counts"]) == {"total", "passed", "failed", "errored"}
        and int(final["result_counts"]["total"]) > 0
        and int(final["result_counts"]["errored"]) == 0
    )
    results["completed_error_null"] = final["error"] is None
    results["completed_metadata_holds"] = final["metadata"] == {}
    results["completed_items_populated"] = (
        len(_run_items(client, spec_id, run_id).json()["data"]) == final["result_counts"]["total"]
    )
    results["completed_receipt_exports"] = (
        client.get(f"/harness/evals/{_bare(run_id)}/receipt", headers=_h(_ROOT)).status_code == 200
    )
    # Re-cancel on terminal is a second pinned refusal.
    cx2 = client.post(f"/v1/evals/{spec_id}/runs/{run_id}/cancel", headers=_h(_ROOT))
    results["cancel_terminal_409_conflict"] = (
        cx2.status_code == 409 and _err_code(cx2) == "conflict"
    )
    results["cancel_missing_run_404"] = (
        client.post(f"/v1/evals/{spec_id}/runs/evalrun_ghost/cancel", headers=_h(_ROOT)).status_code
        == 404
    )

    # Delete: non-terminal refuses; terminal tombstones both surfaces.
    # Re-arm the gate so the next run is parked mid-flight when the
    # delete lands — deterministic non-terminal window.
    gate.entered.clear()
    gate.gate.clear()
    r2 = _run_submit(client, spec_id)
    rid2 = r2.json()["id"]
    assert gate.entered.wait(15), "second run never parked"
    dr_early = client.delete(f"/v1/evals/{spec_id}/runs/{rid2}", headers=_h(_ROOT))
    results["delete_nonterminal_409"] = dr_early.status_code == 409
    gate.gate.set()
    _wait_wire(client, spec_id, rid2)
    d = client.delete(f"/v1/evals/{spec_id}/runs/{rid2}", headers=_h(_ROOT))
    results["delete_terminal_200_shape"] = (
        d.status_code == 200
        and d.json()["object"] == "eval.run.deleted"
        and d.json()["deleted"] is True
        and d.json()["id"] == rid2
    )
    results["run_get_deleted_404"] = _run_get(client, spec_id, rid2).status_code == 404
    results["run_items_deleted_404"] = _run_items(client, spec_id, rid2).status_code == 404
    results["run_list_excludes_deleted"] = all(
        rw["id"] != rid2
        for rw in client.get(
            f"/v1/evals/{spec_id}/runs", params={"limit": 100}, headers=_h(_ROOT)
        ).json()["data"]
    )
    results["harness_record_tombstoned"] = (
        client.get(f"/harness/evals/{_bare(rid2)}", headers=_h(_ROOT)).status_code == 404
    )
    results["run_delete_twice_404"] = (
        client.delete(f"/v1/evals/{spec_id}/runs/{rid2}", headers=_h(_ROOT)).status_code == 404
    )

    # Spec delete tombstones the wire surface but never the evidence record.
    spec_id2 = _spec_id(client)
    r3 = _run_submit(client, spec_id2)
    rid3 = r3.json()["id"]
    _wait_wire(client, spec_id2, rid3)
    client.delete(f"/v1/evals/{spec_id2}", headers=_h(_ROOT))
    results["spec_delete_404s_run_get"] = _run_get(client, spec_id2, rid3).status_code == 404
    results["spec_delete_404s_run_list"] = (
        client.get(f"/v1/evals/{spec_id2}/runs", headers=_h(_ROOT)).status_code == 404
    )
    results["spec_delete_404s_items"] = _run_items(client, spec_id2, rid3).status_code == 404
    results["spec_delete_404s_run_submit"] = _probe_deleted_spec_submit(client)
    h = client.get(f"/harness/evals/{_bare(rid3)}", headers=_h(_ROOT))
    results["harness_record_survives_spec_delete"] = (
        h.status_code == 200 and h.json()["eval_spec"] == spec_id2
    )
    results["harness_receipt_survives_spec_delete"] = (
        client.get(f"/harness/evals/{_bare(rid3)}/receipt", headers=_h(_ROOT)).status_code == 200
    )

    # Wire status mapping covers the failed projection too.
    r_fail = _run_submit(client, spec_id, model="fx1")
    w_fail = _wait_wire(client, spec_id, r_fail.json()["id"])
    results["failed_status_mapped"] = w_fail["status"] == "failed"
    results["failed_error_block"] = (
        w_fail["error"]["code"] == "eval_run_failed"
        and "checkpoint_dir" in w_fail["error"]["message"]
    )
    results["failed_run_no_result_counts"] = w_fail["result_counts"] is None
    results["failed_run_items_empty"] = (
        _run_items(client, spec_id, r_fail.json()["id"]).json()["data"] == []
    )

    # Runs list: newest-first, cursor + limit bounds, bare ids accepted.
    page = client.get(f"/v1/evals/{spec_id}/runs", params={"limit": 2}, headers=_h(_ROOT))
    body = page.json()
    results["runs_list_newest_first"] = (
        len(body["data"]) == 2 and body["data"][0]["created_at"] >= body["data"][1]["created_at"]
    )
    results["runs_list_limit_0_400"] = (
        client.get(f"/v1/evals/{spec_id}/runs", params={"limit": 0}, headers=_h(_ROOT)).status_code
        == 400
    )
    results["runs_list_limit_101_400"] = (
        client.get(
            f"/v1/evals/{spec_id}/runs", params={"limit": 101}, headers=_h(_ROOT)
        ).status_code
        == 400
    )
    cur = client.get(
        f"/v1/evals/{spec_id}/runs",
        params={"after": body["data"][0]["id"], "limit": 1},
        headers=_h(_ROOT),
    ).json()
    results["runs_cursor_chains_prefixed_id"] = cur["data"][0]["id"] == body["data"][1]["id"]
    cur_bare = client.get(
        f"/v1/evals/{spec_id}/runs",
        params={"after": _bare(body["data"][0]["id"]), "limit": 1},
        headers=_h(_ROOT),
    ).json()
    results["runs_cursor_accepts_bare_id"] = cur_bare["data"][0]["id"] == body["data"][1]["id"]
    r_bad = client.get(
        f"/v1/evals/{spec_id}/runs", params={"after": "evalrun_bogus"}, headers=_h(_ROOT)
    )
    results["runs_bad_cursor_400"] = (
        r_bad.status_code == 400 and _err_code(r_bad) == "invalid_cursor"
    )
    r_for = client.get(f"/v1/evals/{spec_id}/runs", params={"after": rid3}, headers=_h(_ROOT))
    results["runs_foreign_cursor_400"] = (
        r_for.status_code == 400 and _err_code(r_for) == "invalid_cursor"
    )


# ---------------------------------------------------------------------------
# Output items (grading output)
# ---------------------------------------------------------------------------


def _probe_output_items(results: dict[str, bool]) -> None:
    client, _api = _client()
    spec_id = _spec_id(client)

    r = _run_submit(client, spec_id)
    run_id = r.json()["id"]
    wire = _wait_wire(client, spec_id, run_id)
    assert wire["status"] == "completed"
    rec = _rec(client, run_id)

    page = _run_items(client, spec_id, run_id, limit=100)
    body = page.json()
    items = body["data"]
    results["items_200"] = page.status_code == 200
    results["items_page_shape"] = (
        body["object"] == "list"
        and isinstance(body["has_more"], bool)
        and "first_id" in body
        and "last_id" in body
    )
    results["items_populated_after_success"] = len(items) == wire["result_counts"]["total"]
    first = items[0]
    results["item_id_shape"] = bool(re.fullmatch(r"evalrun_[0-9a-f]+-\d+", first["id"]))
    results["item_object_shape"] = first["object"] == "eval.run.output_item"
    results["item_binds_run_and_spec"] = first["run_id"] == run_id and first["eval_id"] == spec_id
    results["item_status_is_verdict"] = first["status"] in ("pass", "fail")
    results["item_datasource_item_verbatim_row"] = (
        first["datasource_item_id"] == first["datasource_item"]["task_id"]
    )
    results["item_results_named_by_suite"] = first["results"] == [
        {"name": "tooluse", "passed": first["status"] == "pass"}
    ]
    results["item_created_at_terminal_stamp"] = first["created_at"] == int(
        rec["finished_at"] or rec["created_at"]
    )

    # The verdict rows are the suite's report verbatim — cross-check
    # against the journaled record's report.
    report = rec["report"]
    rows = report.get("outcomes") or report.get("results") or []
    verdict_by_name = {}
    for row in rows:
        name = row.get("task_id") or row.get("task") or row.get("question_id")
        flag = row.get("completed", row.get("passed", row.get("correct")))
        if isinstance(name, str) and isinstance(flag, bool):
            verdict_by_name[name] = flag
    items_by_name = {it["datasource_item_id"]: it["status"] == "pass" for it in items}
    results["items_verdicts_match_report"] = items_by_name == verdict_by_name

    # Honest paging: index-encoded cursor, foreign/malformed refuse.
    p1 = _run_items(client, spec_id, run_id, limit=4).json()
    p2 = _run_items(client, spec_id, run_id, limit=4, after=p1["last_id"]).json()
    results["items_cursor_chains"] = (
        len(p1["data"]) == 4
        and len(p2["data"]) == 4
        and p2["data"][0]["id"] == items[4]["id"]
        and p1["has_more"] is True
    )
    p3 = _run_items(client, spec_id, run_id, limit=1).json()
    results["items_limit_1_one_row"] = len(p3["data"]) == 1
    results["items_limit_0_400"] = _run_items(client, spec_id, run_id, limit=0).status_code == 400
    results["items_limit_101_400"] = (
        _run_items(client, spec_id, run_id, limit=101).status_code == 400
    )
    r_bad = _run_items(client, spec_id, run_id, after="nonsense")
    results["items_bad_cursor_400"] = (
        r_bad.status_code == 400 and _err_code(r_bad) == "invalid_cursor"
    )
    foreign = _run_items(client, spec_id, run_id, after="evalrun_ffffffff-0")
    results["items_foreign_cursor_400"] = (
        foreign.status_code == 400 and _err_code(foreign) == "invalid_cursor"
    )
    past_end = _run_items(client, spec_id, run_id, after=items[-1]["id"])
    results["items_past_end_empty"] = past_end.json()["data"] == []

    # A zero-task suite serves an honest empty page (capability's gate
    # report has no per-task verdict array — never fabricated rows).
    cap_id = _spec_id(
        client,
        data_source_config={
            "type": "custom",
            "item_schema": {"suite": "capability", "seed": 0},
        },
        name="cap-spec",
    )
    rc = _run_submit(client, cap_id)
    w_cap = _wait_wire(client, cap_id, rc.json()["id"], timeout_s=90)
    assert w_cap["status"] == "completed", w_cap
    cap_items = _run_items(client, cap_id, rc.json()["id"]).json()
    results["zero_task_items_empty"] = cap_items["data"] == [] and cap_items["has_more"] is False
    results["zero_task_result_counts_zero"] = w_cap["result_counts"] == {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "errored": 0,
    }
    results["zero_task_first_last_null"] = (
        cap_items["first_id"] is None and cap_items["last_id"] is None
    )

    # Criteria are declarative: the suite's verdict is the grading output —
    # per-criteria results stay empty rather than pretending to measure.
    results["criteria_stored_not_executed"] = wire["per_testing_criteria_results"] == []
    results["criteria_do_not_gate_runs"] = wire["status"] == "completed"


# ---------------------------------------------------------------------------
# Scope/auth + shared workspace
# ---------------------------------------------------------------------------


def _probe_scope_auth(results: dict[str, bool]) -> None:
    client, _api = _client()
    spec_id = _spec_id(client)

    # No credential → 401 envelope; wrong Bearer → 401; Bearer works on /v1.
    r_no = client.get("/v1/evals", headers={})
    results["no_key_401"] = r_no.status_code == 401
    results["no_key_envelope_shape"] = (
        _err_type(r_no) == "authentication_error" and _err_code(r_no) == "unauthorized"
    )
    r_bearer_bad = client.post("/v1/evals", json=_SPEC, headers=_bearer("wrong"))
    results["bearer_wrong_401"] = r_bearer_bad.status_code == 401
    r_bearer = client.post("/v1/evals", json=_SPEC, headers=_bearer(_ROOT))
    results["bearer_env_root_works"] = r_bearer.status_code == 201
    results["bearer_env_root_can_read"] = (
        client.get("/v1/evals", headers=_bearer(_ROOT)).status_code == 200
    )
    mixed = client.post(
        "/v1/evals",
        json=_SPEC,
        headers={**_bearer(_ROOT), "X-API-Key": "fx1k_forged"},
    )
    results["mixed_auth_400"] = mixed.status_code == 400

    # Literal scopes: write without read cannot GET; read without write
    # cannot POST.
    key_r, _idr = _mint(client, name="r", scopes=["read"])
    key_w, _idw = _mint(client, name="w", scopes=["write"])
    key_rw, _idrw = _mint(client, name="rw", scopes=["read", "write"])
    results["read_key_cannot_create_spec"] = (
        client.post("/v1/evals", json=_SPEC, headers=_h(key_r)).status_code == 403
    )
    results["write_key_cannot_get_spec"] = (
        client.get(f"/v1/evals/{spec_id}", headers=_h(key_w)).status_code == 403
    )
    results["write_key_cannot_list_specs"] = (
        client.get("/v1/evals", headers=_h(key_w)).status_code == 403
    )
    results["write_key_cannot_list_runs"] = (
        client.get(f"/v1/evals/{spec_id}/runs", headers=_h(key_w)).status_code == 403
    )
    results["read_key_cannot_submit_run"] = (
        _run_submit(client, spec_id, auth=key_r).status_code == 403
    )
    results["read_key_cannot_delete_spec"] = (
        client.delete(f"/v1/evals/{spec_id}", headers=_h(key_r)).status_code == 403
    )
    rid_read = _run_submit(client, spec_id).json()["id"]
    results["read_key_can_read_items"] = (
        _run_items(client, spec_id, rid_read, auth=key_r).status_code == 200
    )
    results["insufficient_scope_envelope"] = (
        _err_code(client.get("/v1/evals", headers=_h(key_w))) == "insufficient_scope"
    )

    # Shared workspace: a scoped peer can read, mutate, run, and delete
    # another principal's evals — the declared contract, pinned.
    results["peer_reads_foreign_spec"] = (
        client.get(f"/v1/evals/{spec_id}", headers=_h(key_rw)).status_code == 200
    )
    results["peer_updates_foreign_spec"] = (
        client.post(
            f"/v1/evals/{spec_id}",
            json={"name": "peer-renamed"},
            headers=_h(key_rw),
        ).status_code
        == 200
    )
    r_peer = _run_submit(client, spec_id, auth=key_rw)
    results["peer_runs_on_foreign_spec"] = r_peer.status_code == 201
    rid = r_peer.json()["id"]
    results["peer_reads_foreign_run"] = (
        _run_get(client, spec_id, rid, auth=key_rw).status_code == 200
    )
    results["peer_reads_foreign_items"] = (
        _run_items(client, spec_id, rid, auth=key_rw).status_code == 200
    )
    results["peer_deletes_foreign_run"] = (
        client.delete(f"/v1/evals/{spec_id}/runs/{rid}", headers=_h(key_rw)).status_code == 200
    )
    results["peer_deletes_foreign_spec"] = (
        client.delete(f"/v1/evals/{spec_id}", headers=_h(key_rw)).status_code == 200
    )

    # Non-admin keys are refused the control plane.
    results["non_admin_key_roster_403"] = (
        client.get("/harness/keys", headers=_h(key_rw)).status_code == 403
    )
    results["non_admin_key_mint_403"] = (
        client.post("/harness/keys", json={}, headers=_h(key_rw)).status_code == 403
    )
    results["non_admin_key_drain_403"] = (
        client.post("/harness/drain", headers=_h(key_rw)).status_code == 403
    )

    # Same idem key under two credentials → independent claims (no leak).
    k_a, _ = _mint(client, name="a2", scopes=["write"])
    k_b, _ = _mint(client, name="b2", scopes=["write"])
    sp2 = _spec_id(client)
    ra = _run_submit(client, sp2, auth=k_a, extra_headers={"Idempotency-Key": "same"})
    rb = _run_submit(client, sp2, auth=k_b, extra_headers={"Idempotency-Key": "same"})
    results["idem_isolated_per_credential"] = ra.json()["id"] != rb.json()["id"]


# ---------------------------------------------------------------------------
# Metering
# ---------------------------------------------------------------------------


def _probe_metering(results: dict[str, bool]) -> None:
    client, _api = _client()
    key, kid = _mint(client, name="metered", scopes=["read", "write"])

    def uses() -> int:
        c = client.get(f"/harness/keys/{kid}/usage", headers=_h(_ROOT))
        return int(c.json()["uses"])

    base = uses()
    client.get("/v1/evals", headers=_h(key))
    client.get("/v1/evals", headers=_h(key))
    results["eval_list_bills_caller"] = uses() - base == 2

    sp = _spec(client, auth=key)
    results["spec_create_bills_caller"] = uses() - base == 3

    r = _run_submit(client, sp["id"], auth=key)
    results["run_submit_bills_caller"] = uses() - base == 4
    _wait_wire(client, sp["id"], r.json()["id"])
    card = client.get(f"/harness/keys/{kid}/usage", headers=_h(_ROOT)).json()
    results["eval_path_charges_no_tokens"] = int(card["tokens_used"]) == 0

    # The model call meters under the eval ledger, globally.
    metrics = client.get("/metrics", headers=_h(_ROOT)).json()
    complete = metrics.get("complete", {})
    results["eval_model_meters_eval_ledger"] = (
        "eval:tooluse:byok" in complete and int(complete["eval:tooluse:byok"]["ok"]) >= 1
    )
    results["eval_ledger_records_usage"] = (
        int(complete.get("eval:tooluse:byok", {}).get("usage_calls", 0)) >= 1
    )

    # A peer's run bills the acting key, not the spec's minter.
    k_b, kb_id = _mint(client, name="peer", scopes=["read", "write"])

    def uses_b() -> int:
        return int(client.get(f"/harness/keys/{kb_id}/usage", headers=_h(_ROOT)).json()["uses"])

    before_a, before_b = uses(), uses_b()
    _run_submit(client, sp["id"], auth=k_b)
    results["peer_run_bills_actor"] = uses_b() - before_b == 1 and uses() == before_a


# ---------------------------------------------------------------------------
# Drain
# ---------------------------------------------------------------------------


def _probe_drain(results: dict[str, bool]) -> None:
    gate = _GateBackend()
    client, _api = _client(backend_map=_backends(gate))
    spec_id = _spec_id(client)

    # An in-flight run keeps its claim when the latch falls.
    r_live = _run_submit(client, spec_id)
    rid_live = r_live.json()["id"]
    assert gate.entered.wait(15), "run never entered the backend"
    assert client.post("/harness/drain", headers=_h(_ROOT)).status_code == 200

    r_drain = _run_submit(client, spec_id)
    results["run_submit_draining_503"] = r_drain.status_code == 503
    results["run_submit_draining_code"] = _err_code(r_drain) == "draining"
    results["run_submit_draining_envelope"] = _err_type(r_drain) == "service_unavailable"

    # Spec create is a registry write — drain gates worker admission, not
    # the container surface.
    results["spec_create_under_drain_allowed"] = (
        client.post("/v1/evals", json=_SPEC, headers=_h(_ROOT)).status_code == 201
    )
    results["reads_under_drain_200"] = (
        client.get("/v1/evals", headers=_h(_ROOT)).status_code == 200
        and _run_get(client, spec_id, rid_live).status_code == 200
    )

    gate.gate.set()
    final = _wait_wire(client, spec_id, rid_live)
    results["inflight_run_finishes_under_drain"] = final["status"] == "completed"
    results["inflight_record_terminal_under_drain"] = (
        _rec(client, rid_live)["status"] == "succeeded"
    )


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------


def _probe_webhook(results: dict[str, bool]) -> None:
    from fx1.serve.webhooks import verify_webhook  # noqa: PLC0415

    sink = _Sink()
    _RESOURCES.get().callback(sink.close)
    client, _api = _client()
    spec_id = _spec_id(client)

    secret = "whsec_probe"
    r = _run_submit(client, spec_id, callback_url=sink.url("/hook"), callback_secret=secret)
    run_id = r.json()["id"]
    _wait_wire(client, spec_id, run_id)
    assert _wait_for(lambda: len(sink.hits) >= 1, timeout_s=10), "callback never landed"
    hit = sink.hits[0]
    results["callback_fires_on_terminal"] = len(sink.hits) == 1
    results["callback_path_honored"] = hit.path == "/hook"
    body = json.loads(hit.body)
    results["callback_body_is_record"] = body.get("eval_id") == _bare(run_id)
    results["callback_body_binds_spec"] = body.get("eval_spec") == spec_id
    results["callback_body_binds_model"] = body.get("eval_model") == _MODEL
    results["callback_never_echoes_secret"] = secret not in hit.body.decode()
    sig = hit.headers.get("X-Fx1-Webhook-Signature", "")
    ts = hit.headers.get("X-Fx1-Webhook-Timestamp", "")
    results["callback_signed_envelope"] = (
        str(sig).startswith("sha256=")
        and verify_webhook(secret, str(ts), str(sig), hit.body)
        and str(ts).isdigit()
    )
    rec = _rec(client, run_id)
    results["callback_status_recorded"] = rec["callback_status"] == "delivered"
    results["callback_attempts_recorded"] = int(rec["callback_attempts"]) >= 1

    # A failing endpoint marks failed after bounded retries; never raises
    # into the run itself.
    r2 = _run_submit(client, spec_id, callback_url=sink.url("/fail"))
    rid2 = r2.json()["id"]
    _wait_wire(client, spec_id, rid2)
    assert _wait_for(
        lambda: (
            (
                client.get(f"/harness/evals/{_bare(rid2)}", headers=_h(_ROOT))
                .json()
                .get("callback_status")
            )
            == "failed"
        ),
        timeout_s=15,
    ), "failed delivery never sealed"
    rec2 = _rec(client, rid2)
    results["callback_failure_recorded"] = rec2["callback_status"] == "failed"
    results["callback_failure_bounded_retries"] = int(rec2["callback_attempts"]) == 3
    results["callback_failure_run_still_succeeds"] = rec2["status"] == "succeeded"

    # No callback_url → no delivery machinery.
    r3 = _run_submit(client, spec_id)
    rid3 = r3.json()["id"]
    _wait_wire(client, spec_id, rid3)
    time.sleep(0.1)
    rec3 = _rec(client, rid3)
    results["no_callback_no_delivery"] = (
        rec3["callback_status"] is None and int(rec3["callback_attempts"]) == 0
    )


# ---------------------------------------------------------------------------
# Durability (state-dir restart)
# ---------------------------------------------------------------------------


def _probe_durability(results: dict[str, bool]) -> None:
    base = _temporary_directory()
    state = base / "state"

    c1, _api = _client(state_dir=state)
    spec = _spec(c1, name="durable")
    spec_id = spec["id"]
    r = _run_submit(c1, spec_id, metadata={"epoch": "1"})
    run_id = r.json()["id"]
    _wait_wire(c1, spec_id, run_id)
    pre = _run_get(c1, spec_id, run_id).json()
    pre_items = _items_all(c1, spec_id, run_id)
    dead = _spec_id(c1, name="dead-spec")
    c1.delete(f"/v1/evals/{dead}", headers=_h(_ROOT))
    drun = _run_submit(c1, spec_id, extra_headers={"Idempotency-Key": "durable-key"})
    drun_id = drun.json()["id"]
    _wait_wire(c1, spec_id, drun_id)
    c1.close()

    c2, _api2 = _client(state_dir=state)
    g = c2.get(f"/v1/evals/{spec_id}", headers=_h(_ROOT))
    results["spec_survives_restart"] = g.status_code == 200 and g.json()["name"] == "durable"
    g2 = c2.get(f"/v1/evals/{spec_id}/runs/{run_id}", headers=_h(_ROOT))
    results["run_survives_restart"] = g2.status_code == 200
    post = g2.json()
    results["run_wire_identical_after_restart"] = all(
        post[k] == pre[k]
        for k in (
            "id",
            "eval_id",
            "model",
            "status",
            "suite",
            "seed",
            "backend",
            "metadata",
        )
    )
    results["run_metadata_survives_restart"] = post["metadata"] == {"epoch": "1"}
    results["items_recompute_identically"] = [
        (i["id"], i["status"], i["datasource_item_id"]) for i in _items_all(c2, spec_id, run_id)
    ] == [(i["id"], i["status"], i["datasource_item_id"]) for i in pre_items]
    results["deleted_spec_stays_deleted"] = (
        c2.get(f"/v1/evals/{dead}", headers=_h(_ROOT)).status_code == 404
    )
    results["deleted_spec_runs_404"] = (
        c2.get(f"/v1/evals/{dead}/runs", headers=_h(_ROOT)).status_code == 404
    )
    # The idem claim rides the journal: a retried submit returns the
    # restored record rather than duplicating the run.
    r_re = _run_submit(c2, spec_id, extra_headers={"Idempotency-Key": "durable-key"})
    results["idem_replay_survives_restart"] = r_re.json()["id"] == drun_id


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def _probe_concurrency(results: dict[str, bool]) -> None:
    client, _api = _client()
    spec_id = _spec_id(client)

    # Parallel submits under the spec claim lock mint distinct ids.
    with ThreadPoolExecutor(max_workers=_N) as pool:
        futs = [pool.submit(_run_submit, client, spec_id) for _ in range(_N)]
        subs = [f.result() for f in futs]
    ids = [s.json()["id"] for s in subs if s.status_code == 201]
    results["parallel_runs_all_201"] = len(ids) == _N
    results["parallel_runs_distinct_ids"] = len(set(ids)) == _N
    for rid in ids:
        _wait_wire(client, spec_id, rid)
    runs_page = client.get(
        f"/v1/evals/{spec_id}/runs", params={"limit": 100}, headers=_h(_ROOT)
    ).json()
    listed = {rw["id"] for rw in runs_page["data"]}
    results["parallel_runs_all_listed"] = all(i in listed for i in ids)

    # Same idem key across a parallel burst: one run, N-1 replays.
    burst = "burst-key-1"

    def submit_burst() -> Any:
        return _run_submit(client, spec_id, extra_headers={"Idempotency-Key": burst})

    with ThreadPoolExecutor(max_workers=_N) as pool:
        burst_subs = [f.result() for f in [pool.submit(submit_burst) for _ in range(_N)]]
    burst_ids = {s.json()["id"] for s in burst_subs if s.status_code == 201}
    results["parallel_same_idem_one_run"] = len(burst_ids) == 1 and all(
        s.status_code == 201 for s in burst_subs
    )

    # Concurrent output_item reads never tear.
    rid = next(iter(burst_ids))
    _wait_wire(client, spec_id, rid)
    pages: list[dict[str, Any]] = []
    lock = threading.Lock()

    def reader() -> None:
        for _ in range(6):
            body = _run_items(client, spec_id, rid, limit=100).json()
            with lock:
                pages.append(body)

    with ThreadPoolExecutor(max_workers=4) as pool:
        threads = [pool.submit(reader) for _ in range(4)]
        for t in threads:
            t.result()
    sigs = {tuple((i["id"], i["status"]) for i in p["data"]) for p in pages}
    results["concurrent_item_reads_consistent"] = len(sigs) == 1

    # Cancel racing the running claim: every running record refuses 409 —
    # the CAS window is one atomic queued→running claim, and a terminal
    # record refuses again after. The race can't split the difference.
    gate = _GateBackend()
    c2, _api2 = _client(backend_map=_backends(gate))
    sp2 = _spec_id(c2)
    gate_rids = [_run_submit(c2, sp2).json()["id"] for _ in range(3)]
    assert _wait_for(lambda: gate.entered.is_set(), timeout_s=10)

    def try_cancel(rid_: str) -> int:
        return c2.post(f"/v1/evals/{sp2}/runs/{rid_}/cancel", headers=_h(_ROOT)).status_code

    with ThreadPoolExecutor(max_workers=3) as pool:
        codes = list(pool.map(try_cancel, gate_rids))
    results["cancel_running_always_409"] = all(code == 409 for code in codes)
    gate.gate.set()
    for rid_ in gate_rids:
        _wait_wire(c2, sp2, rid_)
    finals = [_rec(c2, r_)["status"] for r_ in gate_rids]
    results["post_race_runs_terminal_clean"] = all(s == "succeeded" for s in finals)


# ---------------------------------------------------------------------------
# Over-capacity + refusal envelopes
# ---------------------------------------------------------------------------


def _probe_capacity_and_envelope(results: dict[str, bool]) -> None:
    gate = _GateBackend()
    client, _api = _client(backend_map=_backends(gate), max_inflight=1)
    spec_id = _spec_id(client)
    r1 = _run_submit(client, spec_id)
    assert gate.entered.wait(15)
    r2 = _run_submit(client, spec_id)
    results["over_capacity_503"] = r2.status_code == 503
    results["over_capacity_code"] = _err_code(r2) == "over_capacity"
    results["over_capacity_retry_after"] = r2.headers.get("retry-after") is not None
    gate.gate.set()
    _wait_wire(client, spec_id, r1.json()["id"])

    # Every refusal class lands in the {error:{...}} envelope, never bare
    # {detail} — spot-check the distinct classes hit in this battery.
    checks = [
        client.post("/v1/evals", json={}, headers=_h(_ROOT)),
        client.get("/v1/evals/eval_ghost", headers=_h(_ROOT)),
        _run_submit(client, "eval_ghost"),
        _run_submit(client, spec_id, model="gpt-9"),
        client.get("/v1/evals", params={"limit": 0}, headers=_h(_ROOT)),
        client.get("/v1/evals", headers={}),
    ]
    results["all_refusals_enveloped"] = all(
        isinstance(c.json().get("error"), dict) and "message" in c.json()["error"] for c in checks
    )

    # Metering distinguishes authenticated traffic: an in-scope GET that
    # 404s still bills the caller's uses, while a scope-refused POST
    # short-circuits before the ledger — and neither charges tokens.
    key_r, kid_r = _mint(client, name="refuser", scopes=["read"])

    def uses_r() -> int:
        return int(client.get(f"/harness/keys/{kid_r}/usage", headers=_h(_ROOT)).json()["uses"])

    before = uses_r()
    client.get("/v1/evals/eval_ghost", headers=_h(key_r))  # 404, authorized
    after_get = uses_r()
    results["authorized_404_bills_uses"] = after_get == before + 1
    client.post("/v1/evals", json=_SPEC, headers=_h(key_r))  # 403 scope refusal
    after_post = uses_r()
    results["scope_refusal_bills_no_uses"] = after_post == after_get
    card = client.get(f"/harness/keys/{kid_r}/usage", headers=_h(_ROOT)).json()
    results["refused_requests_charge_no_tokens"] = int(card["tokens_used"]) == 0


# ---------------------------------------------------------------------------
# SDK parity
# ---------------------------------------------------------------------------


def _probe_sdk_parity(results: dict[str, bool]) -> None:
    from fx1.harness import Harness
    from fx1.sdk import Fx1Harness

    sdk = Fx1Harness(
        harness=Harness(runner=lambda argv, timeout_s: (0, "ok", "")),
        backend_resolver=lambda name, *a, **k: _StubBackend("sdk-stub"),
    )
    spec = sdk.eval_spec_create(
        "sdk-spec",
        suite="tooluse",
        seed=0,
        testing_criteria=[{"name": "c"}],
        metadata={"via": "sdk"},
    )
    results["sdk_spec_create_shape"] = spec["object"] == "eval" and spec["id"].startswith("eval_")
    results["sdk_spec_get_roundtrip"] = sdk.eval_spec_get(spec["id"])["name"] == "sdk-spec"
    results["sdk_spec_update_metadata"] = sdk.eval_spec_update(spec["id"], metadata={"v": "2"})[
        "metadata"
    ] == {"v": "2"}
    run = sdk.eval_run_create(spec["id"], model="byok", metadata={"origin": "sdk"})
    results["sdk_run_terminal_completed"] = run["status"] == "completed"
    results["sdk_run_wire_fields"] = {
        "id",
        "object",
        "eval_id",
        "model",
        "status",
        "created_at",
        "suite",
        "seed",
        "backend",
        "metadata",
        "result_counts",
        "per_testing_criteria_results",
        "error",
        "receipt_url",
    } <= set(run)
    results["sdk_run_metadata_flows"] = run["metadata"] == {"origin": "sdk"}
    results["sdk_run_bound_to_spec"] = run["eval_id"] == spec["id"]
    results["sdk_runs_list"] = any(r["id"] == run["id"] for r in sdk.eval_runs(spec["id"]))
    items = sdk.eval_run_items(spec["id"], run["id"])
    results["sdk_items_populated"] = len(items) == run["result_counts"]["total"]
    results["sdk_item_shape_matches_wire"] = items[0]["object"] == "eval.run.output_item" and items[
        0
    ]["results"] == [{"name": "tooluse", "passed": items[0]["status"] == "pass"}]

    # Error parity: missing spec → KeyError; unregistered ft → KeyError;
    # unknown model → ValueError; empty update → ValueError.
    results["sdk_missing_spec_keyerror"] = (
        _raises(lambda: sdk.eval_run_create("eval_ghost", model="byok")) == "KeyError"
    )
    results["sdk_unregistered_ft_keyerror"] = (
        _raises(lambda: sdk.eval_run_create(spec["id"], model="ft:ghost")) == "KeyError"
    )
    results["sdk_unknown_model_valueerror"] = (
        _raises(lambda: sdk.eval_run_create(spec["id"], model="gpt-9")) == "ValueError"
    )
    results["sdk_update_no_field_valueerror"] = (
        _raises(lambda: sdk.eval_spec_update(spec["id"])) == "ValueError"
    )
    results["sdk_spec_delete_keyerror"] = (
        _raises(lambda: sdk.eval_spec_delete("eval_ghost")) == "KeyError"
    )
    sdk.eval_spec_delete(spec["id"])
    results["sdk_spec_deleted_get_keyerror"] = (
        _raises(lambda: sdk.eval_spec_get(spec["id"])) == "KeyError"
    )


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


def evals_audit() -> dict[str, bool]:
    """Every probe, measured end-to-end against a live app."""
    results: dict[str, bool] = {}
    with _audit_context():
        _probe_spec_create(results)
        _probe_spec_lifecycle(results)
        _probe_run_submit(results)
        _probe_idempotency(results)
        _probe_run_lifecycle(results)
        _probe_output_items(results)
        _probe_scope_auth(results)
        _probe_metering(results)
        _probe_drain(results)
        _probe_webhook(results)
        _probe_durability(results)
        _probe_concurrency(results)
        _probe_capacity_and_envelope(results)
        _probe_sdk_parity(results)
    return results


def evals_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = evals_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "evals_audit",
        "schema": "evals_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "Starlette TestClient (in-process ASGI) with stub backends",
            "not_verified": [
                "multi-process writers on one state_dir (single-process lock)",
                "hosted_K3/BYOK upstream truth (stub backends never dial out)",
                "real grader execution inside testing_criteria (criteria are "
                "declarative spec metadata — the suite's verdict row is the "
                "grading output, pinned under datasource_item/results)",
                "stored-completions/file_id data sources (the vocabulary is "
                "closed — their refusal IS the pinned contract)",
                "webhook delivery past loopback (signed envelope verified; "
                "remote-network retries/TLS are out of scope)",
            ],
        },
        "interpretation": (
            "The /v1/evals surface is an honest gate: spec validation is a "
            "closed custom vocabulary that refuses every malformed or "
            "stored-completions-shaped body at 422; run submits bind "
            "eval_spec+eval_model at construction so cross-spec reads, "
            "lists, and deletes 404; the run's model is recorded verbatim "
            "while the record carries the resolved backend chain; submit "
            "metadata rides the record, the wire, and the sealed receipt; "
            "idempotency dedupes per (credential, key, spec) with the raw "
            "header's 256-char bound; grading output is the suite's real "
            "per-task verdict rows paged honestly with index cursors; "
            "cancels refuse non-queued records 409 and deletes tombstone "
            "both surfaces; scopes are literal while the workspace is "
            "shared as declared; drain gates submissions but not reads or "
            "registry writes; the terminal webhook fires once, signed, and "
            "never leaks the secret; --state-dir restart restores specs, "
            "runs, tombstones, and idem claims; concurrency mints distinct "
            "ids under the claim lock; metering bills the caller's uses and "
            "the eval ledger without phantom per-key token charges; and "
            "the SDK twin returns identical wire shapes with mapped "
            "exceptions."
            if ok
            else f"EVALS AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(evals_audit_bench(), indent=1))
