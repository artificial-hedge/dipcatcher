"""Surface-parity audit — the SDK and the HTTP API are one contract.

``fx1.sdk.Fx1Harness`` and ``fx1.serve.api`` expose the same harness over
two transports. This audit drives the SAME injected backend through both
and asserts byte-identical results — a drift between what an in-process
caller and what a network caller get is a product defect, not a transport
detail.

- *Complete parity* — ``Fx1Harness.complete`` output equals
  ``POST /harness/complete``: identical content (evidence footer included),
  identical ``model``/``backend``/``receipt_hashes`` envelope fields.
- *Batch parity* — ``complete_many`` results equal
  ``POST /harness/complete/batch`` items in submission order, one shared
  backend per batch on both surfaces (close counts prove it).
- *Stream parity* — SSE ``token`` frames concatenate to exactly the
  ``complete`` content; ``stream_complete`` returns the identical chunk
  list; the ``final`` envelope matches the completion envelope.
- *Error parity* — the same fault surfaces with the same class on both:
  gate refusal (SDK raises ``Fx1HonestyError`` / API 502 JSON with the
  refusal text), non-streaming backend (NotImplementedError / 501),
  checkpoint_dir on a non-local backend (ValueError / 422), unknown
  backend (KeyError / 422 literal rejection), malformed receipts.
- *Verifier parity* — ``verify_receipt`` and ``POST /receipts/verify``
  agree on valid/errors/warnings/schema for a sealed receipt, a tampered
  digest, and a non-receipt dict.
- *Registry/health parity* — command list and backend presence booleans
  are identical over both surfaces.

Sealed ``parity_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

__all__ = ["parity_audit", "parity_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_BYOK_ENVS = ("FX1_BYOK_BASE_URL", "FX1_BYOK_API_KEY", "FX1_BYOK_MODEL")
_LOCAL_ENVS = (
    "FX1_LOCAL_SERVE_URL",
    "FX1_LOCAL_SERVE_CMD",
    "FX1_LOCAL_MODEL",
    "FX1_LOCAL_API_KEY",
    "FX1_CHECKPOINT_DIR",
)


class _ParityBackend:
    """Deterministic streaming backend — one instance per resolution."""

    _closed_total = 0

    def __init__(self) -> None:
        self._model = "parity-v0"

    def complete(self, messages: list[dict[str, str]]) -> str:
        return f"echo:{messages[-1]['content']}"

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        text = self.complete(messages)
        yield text[:3]
        yield text[3:]

    def close(self) -> None:
        _ParityBackend._closed_total += 1


class _DirtyBackend(_ParityBackend):
    def complete(self, messages: list[dict[str, str]]) -> str:
        return "total Sharpe 4.2 on NAV"  # forbidden headline

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield "total Sharpe 4.2 on NAV"


class _NonStreamingBackend:
    def __init__(self) -> None:
        self._model = "ns-v0"

    def complete(self, messages: list[dict[str, str]]) -> str:
        return "ans"

    def close(self) -> None:
        return None


def _raises(fn: Any) -> tuple[str, str]:
    """(exception class name, str(exc)) — ("", "") when no raise."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 — probe captures the class
        return type(exc).__name__, str(exc)
    return "", ""


def _surfaces(
    backend: Any,
) -> tuple[Any, TestClient]:
    """(Fx1Harness, TestClient) wired to one resolver returning `backend`'s class."""
    from fastapi.testclient import TestClient as _TC

    import fx1.serve.api as api_mod
    from fx1.harness import Harness
    from fx1.sdk import Fx1Harness

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, f"ran:{' '.join(argv)}", ""

    def resolver(name: str, **kw: Any) -> Any:
        if name not in {"hosted_k3", "local_fx1", "byok"}:
            raise KeyError(name)
        return backend()

    sdk = Fx1Harness(harness=Harness(runner=fake_runner), backend_resolver=resolver)
    app = api_mod.create_app(harness=Harness(runner=fake_runner), backend_resolver=resolver)
    return sdk, _TC(app)


def parity_audit() -> dict[str, bool]:
    out: dict[str, bool] = {}
    saved = {
        k: os.environ.get(k) for k in (_API_KEY_ENV, *_BYOK_ENVS, *_LOCAL_ENVS, "MOONSHOT_API_KEY")
    }
    try:
        os.environ.pop(_API_KEY_ENV, None)

        sdk, client = _surfaces(_ParityBackend)
        msg = [{"role": "user", "content": "ping"}]
        receipt = "a" * 64
        sdk_out = sdk.complete(msg, backend="byok", receipt_hashes=[receipt])
        api_out = client.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": msg,
                "receipt_hashes": [receipt],
            },
        )
        out["complete_status_200"] = api_out.status_code == 200
        api_json = api_out.json()
        out["complete_content_identical"] = api_json["content"] == sdk_out.content
        out["complete_envelope_identical"] = (
            api_json["backend"] == sdk_out.backend
            and api_json["model"] == sdk_out.model
            and tuple(api_json["receipt_hashes"]) == sdk_out.receipt_hashes
        )
        out["complete_footer_identical"] = (
            sdk_out.content.endswith(
                f"Evidence: `{receipt[:16]}…` — verify with `dipcatcher verify-research`."
            )
            and api_json["content"] == sdk_out.content
        )

        # --- batch parity -----------------------------------------------------
        _ParityBackend._closed_total = 0
        batch = [
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ]
        sdk_many = sdk.complete_many(batch, backend="byok", receipt_hashes=[receipt], max_workers=2)
        api_many = client.post(
            "/harness/complete/batch",
            json={
                "backend": "byok",
                "batch": batch,
                "receipt_hashes": [receipt],
                "max_workers": 2,
            },
        )
        out["batch_status_200"] = api_many.status_code == 200
        api_items = api_many.json()
        out["batch_content_identical"] = [r.content for r in sdk_many] == [
            i["content"] for i in api_items["results"]
        ] and all(i["ok"] for i in api_items["results"])
        out["batch_envelope_identical"] = (
            api_items["backend"] == sdk_many[0].backend
            and api_items["model"] == sdk_many[0].model
            and tuple(api_items["receipt_hashes"]) == sdk_many[0].receipt_hashes
        )
        out["batch_one_backend_each_side"] = _ParityBackend._closed_total == 2

        # --- stream parity ----------------------------------------------------
        sdk_chunks = sdk.stream_complete(msg, backend="byok", receipt_hashes=[receipt])
        api_stream = client.post(
            "/harness/complete/stream",
            json={
                "backend": "byok",
                "messages": msg,
                "receipt_hashes": [receipt],
            },
        )
        out["stream_status_200_sse"] = api_stream.status_code == 200 and api_stream.headers[
            "content-type"
        ].startswith("text/event-stream")
        frames = [
            ln[len("data: ") :] for ln in api_stream.text.splitlines() if ln.startswith("data: ")
        ]
        payloads = [json.loads(f) for f in frames if f.strip() != "[DONE]"]
        token_chunks = [p["content"] for p in payloads if p["type"] == "token"]
        final = [p for p in payloads if p["type"] == "final"]
        out["stream_chunks_identical"] = token_chunks == sdk_chunks
        out["stream_joined_eq_complete"] = (
            "".join(token_chunks) == sdk_out.content and "".join(sdk_chunks) == sdk_out.content
        )
        out["stream_done_terminal"] = frames[-1].strip() == "[DONE]"
        out["stream_final_envelope"] = (
            len(final) == 1
            and final[0]["model"] == sdk_out.model
            and final[0]["receipt_hashes"] == [receipt]
        )

        # --- error parity -----------------------------------------------------
        dirty_sdk, dirty_api = _surfaces(_DirtyBackend)
        sdk_err_cls, sdk_err_txt = _raises(lambda: dirty_sdk.complete(msg, backend="byok"))
        api_err = dirty_api.post("/harness/complete", json={"backend": "byok", "messages": msg})
        out["gate_sdk_raises_api_502"] = (
            sdk_err_cls == "Fx1HonestyError"
            and api_err.status_code == 502
            and "honesty gate" in api_err.json()["detail"]
        )
        # the API surfaces the SDK's exception text verbatim after the prefix
        out["gate_error_text_shared"] = (
            api_err.status_code == 502
            and api_err.json()["detail"] == f"honesty gate refused model output: {sdk_err_txt}"
        )
        sdk_serr, _ = _raises(lambda: dirty_sdk.stream_complete(msg, backend="byok"))
        api_serr = dirty_api.post(
            "/harness/complete/stream", json={"backend": "byok", "messages": msg}
        )
        out["stream_gate_parity"] = (
            sdk_serr == "Fx1HonestyError"
            and api_serr.status_code == 502
            and "total Sharpe" not in api_serr.text
        )

        ns_sdk, ns_api = _surfaces(_NonStreamingBackend)
        out["stream_unsupported_parity"] = (
            _raises(lambda: ns_sdk.stream_complete(msg, backend="byok"))[0] == "NotImplementedError"
            and ns_api.post(
                "/harness/complete/stream",
                json={"backend": "byok", "messages": msg},
            ).status_code
            == 501
        )
        out["checkpoint_dir_non_local_parity"] = (
            _raises(lambda: sdk.complete(msg, backend="byok", checkpoint_dir="/x"))[0]
            == "ValueError"
            and client.post(
                "/harness/complete",
                json={
                    "backend": "byok",
                    "messages": msg,
                    "checkpoint_dir": "/x",
                },
            ).status_code
            == 422
        )
        # Resolver KeyError is unreachable over the wire: CompleteRequest's
        # backend Literal rejects unknown names with 422 before resolution;
        # the SDK raises KeyError on the same name. Both fail closed.
        out["unknown_backend_fail_closed"] = (
            _raises(lambda: sdk.complete(msg, backend="bogus"))[0] == "KeyError"
            and client.post(
                "/harness/complete",
                json={"backend": "bogus", "messages": msg},
            ).status_code
            == 422
        )
        out["empty_batch_contract"] = (
            sdk.complete_many([], backend="byok") == []
            and client.post(
                "/harness/complete/batch",
                json={"backend": "byok", "batch": []},
            ).status_code
            == 422
        )

        # --- verifier parity ----------------------------------------------------
        from fx1.serve.byok_audit import byok_audit_bench

        good = byok_audit_bench()
        sdk_v = sdk.verify_receipt(good)
        api_v = client.post("/receipts/verify", json={"receipt": good}).json()
        out["verify_valid_parity"] = (
            sdk_v.valid is True
            and api_v["valid"] is True
            and api_v["schema_tag"] == sdk_v.schema_tag
            and api_v["kind"] == sdk_v.kind
            and api_v["verdict"] == sdk_v.verdict
            and tuple(api_v["errors"]) == sdk_v.errors
            and tuple(api_v["warnings"]) == sdk_v.warnings
        )
        tampered = dict(good)
        tampered["receipt_sha256"] = "0" * 64
        sdk_tv = sdk.verify_receipt(tampered)
        api_tv = client.post("/receipts/verify", json={"receipt": tampered}).json()
        out["verify_tamper_parity"] = (
            sdk_tv.valid is False
            and api_tv["valid"] is False
            and tuple(api_tv["errors"]) == sdk_tv.errors
        )
        sdk_mv = sdk.verify_receipt({"not": "a receipt"})
        api_mv = client.post("/receipts/verify", json={"receipt": {"not": "a receipt"}}).json()
        out["verify_malformed_parity"] = (
            sdk_mv.valid is False
            and api_mv["valid"] is False
            and tuple(api_mv["errors"]) == sdk_mv.errors
        )

        # --- registry / run / health parity -----------------------------------
        api_cmds = client.get("/harness/commands").json()
        out["commands_parity"] = sorted(sdk.commands()) == sorted(
            c["name"] for c in api_cmds["items"]
        ) and api_cmds["total"] == len(api_cmds["items"])
        name = sorted(sdk.commands())[0]
        sdk_run = sdk.run(name)
        api_run = client.post("/harness/runs", json={"command": name}).json()
        out["run_parity"] = (
            api_run["exit_code"] == sdk_run.exit_code
            and api_run["stdout"] == sdk_run.stdout
            and api_run["ok"] == sdk_run.ok
        )
        api_health = client.get("/health").json()
        sdk_health = sdk.health()
        out["health_parity"] = (
            api_health["backends"] == sdk_health.backends
            and api_health["version"] == sdk_health.version
            and api_health["registered_commands"] == sdk_health.registered_commands
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def parity_audit_bench() -> dict[str, Any]:
    r = parity_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "parity_audit",
        "schema": "parity_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "surface parity holds: SDK and HTTP API return byte-identical "
            "content, envelopes, chunk lists, error taxonomy, verifier "
            "verdicts, registry, and health over the same injected backend — "
            "no transport-dependent behavior drift. Flags: unknown backend "
            "names surface as KeyError in-process vs 422 literal rejection "
            "over the wire (request validation runs before resolution); "
            "empty batches are [] in-process vs 422 over the wire."
            if ok
            else f"PARITY AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
