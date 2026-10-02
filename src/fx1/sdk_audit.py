"""sdk_audit — adversarial probes on ``fx1.sdk.Fx1Harness``.

The SDK is the in-process twin of ``fx1.serve.api``: fx-1 drives the same
registry, the same three backends, the same honesty gate, and the same
receipt verifier without a socket. If the two surfaces drift — a command
listed only over HTTP, an error class that changes meaning in-process —
the harness contract forks and receipts stop meaning the same thing on
both paths. These probes pin the parity.

Pinned contract:

- *Registry parity* — ``commands()`` returns exactly ``HARNESS_REGISTRY``;
  ``role=`` filters by ``HarnessRole``; unknown names raise ``KeyError``
  on ``run`` and ``command`` (the 404-class).
- *Run path* — the injectable runner sees ``dipcatcher``-less argv from
  the registry; ``config`` and ``extra_args --config`` are contained to
  ``configs/``; the result carries exit_code/stdout/stderr/ok.
- *Completion path* — ``local_fx1`` requires a checkpoint (``ValueError``
  — the 422-class); ``checkpoint_dir`` is rejected for other backends;
  ``BackendNotConfiguredError`` propagates untransformed (the 503-class);
  the honesty gate runs before the result returns (``Fx1HonestyError`` —
  the 502-class); ``backend.close()`` is always invoked, even on gate
  failure — spawned engines never leak; ``receipt_hashes`` append the
  evidence footer; ``model``/``backend`` echo what the backend resolved.
- *Receipt surface* — ``verify_receipt`` returns the full verifier shape;
  a sealed bench receipt verifies, a tampered digest fails closed.
- *Health* — presence booleans only: no env values, no paths leak; the
  ``local_fx1`` flag requires a card'd checkpoint AND a serve env.

Sealed ``sdk_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["sdk_audit", "sdk_audit_bench"]


class _FakeBackend:
    """Records close() and returns a canned, honesty-clean completion."""

    def __init__(self, content: str = "clean answer") -> None:
        self._model = "fake-0"
        self.closed = 0
        self._content = content
        self.seen_messages: list[dict[str, str]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.seen_messages = list(messages)
        return self._content

    def close(self) -> None:
        self.closed += 1


def sdk_audit() -> dict[str, bool]:
    """Probe the SDK contract; every key must be True."""
    from fx1.harness import HARNESS_REGISTRY, Harness, HarnessRole
    from fx1.sdk import Fx1Harness
    from fx1.serve.backends import BackendNotConfiguredError

    out: dict[str, bool] = {}

    def _raises(fn: Any) -> str:
        try:
            fn()
            return "no-raise"
        except Exception as e:  # noqa: BLE001 - probe records the class
            return type(e).__name__

    # ---- registry parity -------------------------------------------------
    sdk = Fx1Harness()
    out["registry_full"] = sdk.commands() == [c.name for c in HARNESS_REGISTRY]
    out["registry_role_filter"] = sdk.commands(HarnessRole.VERIFICATION) == [
        c.name for c in HARNESS_REGISTRY if c.role == HarnessRole.VERIFICATION
    ]
    out["registry_nonempty"] = len(sdk.commands()) > 0
    cmd = sdk.command("doctor")
    out["command_shape"] = cmd.argv == ["doctor"] and cmd.role == HarnessRole.VERIFICATION
    out["unknown_command_404"] = _raises(lambda: sdk.command("not-a-command")) == "KeyError"
    out["unknown_run_404"] = _raises(lambda: sdk.run("not-a-command")) == "KeyError"

    # ---- run path ----------------------------------------------------------
    seen: dict[str, Any] = {}

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        seen["argv"] = list(argv)
        seen["timeout_s"] = timeout_s
        return 7, "out-text", "err-text"

    sdk2 = Fx1Harness(harness=Harness(runner=fake_runner))
    res = sdk2.run("doctor")
    out["run_executes_registry_argv"] = seen["argv"] == ["doctor"]
    out["run_result_shape"] = (
        res.exit_code == 7 and res.stdout == "out-text" and res.stderr == "err-text"
    )
    out["run_ok_flag"] = res.ok is False
    out["run_timeout_bound"] = seen["timeout_s"] == sdk2.command("doctor").timeout_s
    out["run_extra_args"] = sdk2.run("doctor", ["--foo", "1"]).exit_code == 7 and seen["argv"] == [
        "doctor",
        "--foo",
        "1",
    ]

    with tempfile.TemporaryDirectory() as td:
        configs = Path(td) / "configs"
        configs.mkdir()
        inside = configs / "ok.yml"
        inside.write_text("x: 1\n")
        cwd = Path.cwd()
        os.chdir(td)
        try:
            out["config_inside_ok"] = sdk2.run("doctor", config=inside).exit_code == 7
            outside = Path(td) / "evil.yml"
            outside.write_text("x: 1\n")
            out["config_escape_422"] = (
                _raises(lambda: sdk2.run("doctor", config=outside)) == "ValueError"
            )
            out["config_extra_args_escape_422"] = (
                _raises(lambda: sdk2.run("doctor", ["--config", str(outside)])) == "ValueError"
            )
            out["config_eq_escape_422"] = (
                _raises(lambda: sdk2.run("doctor", [f"--config={outside}"])) == "ValueError"
            )
        finally:
            os.chdir(cwd)

    # ---- completion path ---------------------------------------------------
    fake = _FakeBackend()
    sdk3 = Fx1Harness(backend_resolver=lambda *a, **k: fake)
    result = sdk3.complete([{"role": "user", "content": "hi"}], backend="byok")
    out["complete_runs_backend"] = result.content == "clean answer"
    out["complete_result_fields"] = result.backend == "byok" and result.model == "fake-0"
    out["complete_backend_closed"] = fake.closed == 1
    out["complete_messages_passthrough"] = fake.seen_messages == [{"role": "user", "content": "hi"}]
    cited = sdk3.complete(
        [{"role": "user", "content": "hi"}],
        backend="byok",
        receipt_hashes=["a" * 64],
    )
    out["cited_footer"] = "Evidence:" in cited.content and "`aaaaaaaaaaaaaaaa…`" in cited.content
    out["cited_hashes_echoed"] = cited.receipt_hashes == ("a" * 64,)
    out["checkpoint_rejected_nonlocal"] = (
        _raises(
            lambda: sdk3.complete(
                [{"role": "u", "content": "x"}],
                backend="byok",
                checkpoint_dir="some-checkpoint",
            )
        )
        == "ValueError"
    )
    env_ckpt = os.environ.pop("FX1_CHECKPOINT_DIR", None)
    try:
        out["local_needs_checkpoint"] = (
            _raises(lambda: sdk3.complete([{"role": "u", "content": "x"}], backend="local_fx1"))
            == "ValueError"
        )
    finally:
        if env_ckpt is not None:
            os.environ["FX1_CHECKPOINT_DIR"] = env_ckpt

    def _unconfigured(*a: Any, **k: Any) -> Any:
        raise BackendNotConfiguredError("no creds")

    sdk4 = Fx1Harness(backend_resolver=_unconfigured)
    out["unconfigured_503_class"] = (
        _raises(lambda: sdk4.complete([{"role": "u", "content": "x"}], backend="byok"))
        == "BackendNotConfiguredError"
    )

    def _unknown(*a: Any, **k: Any) -> Any:
        raise KeyError("bogus-backend")

    sdk5 = Fx1Harness(backend_resolver=_unknown)
    out["unknown_backend_404"] = (
        _raises(lambda: sdk5.complete([{"role": "u", "content": "x"}], backend="byok"))
        == "KeyError"
    )

    dirty = _FakeBackend("total Sharpe 4.2 on NAV")  # forbidden headline
    sdk6 = Fx1Harness(backend_resolver=lambda *a, **k: dirty)
    out["honesty_gate_502"] = (
        _raises(lambda: sdk6.complete([{"role": "u", "content": "x"}], backend="byok"))
        == "Fx1HonestyError"
    )
    out["closed_on_gate_fail"] = dirty.closed == 1

    # ---- receipts ------------------------------------------------------------
    from fx1.serve.byok_audit import byok_audit_bench

    good = byok_audit_bench()
    verdict = sdk.verify_receipt(good)
    out["verify_valid_receipt"] = verdict.valid is True
    out["verify_shape"] = verdict.schema_tag == "byok_audit.v1" and isinstance(
        verdict.errors, tuple
    )
    tampered = dict(good)
    tampered["receipt_sha256"] = "0" * 64
    out["verify_tamper_detected"] = sdk.verify_receipt(tampered).valid is False
    out["verify_malformed_fails_closed"] = sdk.verify_receipt({"not": "receipt"}).valid is False

    # ---- health ----------------------------------------------------------------
    h = sdk.health()
    out["health_booleans_only"] = all(isinstance(v, bool) for v in h.backends.values()) and set(
        h.backends
    ) == {"hosted_k3", "byok", "local_fx1"}
    out["health_no_values"] = all(not isinstance(v, str) for v in h.backends.values())
    out["health_commands_count"] = h.registered_commands == len(sdk.commands())
    out["health_version"] = h.version != ""

    # local_fx1 presence needs card'd checkpoint AND serve env
    envs = ["FX1_CHECKPOINT_DIR", "FX1_LOCAL_SERVE_URL", "FX1_LOCAL_SERVE_CMD"]
    saved = {k: os.environ.pop(k, None) for k in envs}
    try:
        with tempfile.TemporaryDirectory() as td2:
            os.environ["FX1_CHECKPOINT_DIR"] = td2
            out["health_localfx1_needs_card"] = sdk.health().backends["local_fx1"] is False
            (Path(td2) / "modelcard.json").write_text("{}")
            out["health_localfx1_needs_serve_env"] = sdk.health().backends["local_fx1"] is False
            os.environ["FX1_LOCAL_SERVE_URL"] = "http://127.0.0.1:9"
            out["health_localfx1_configured"] = sdk.health().backends["local_fx1"] is True
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    return out


def sdk_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under sdk_audit.v1."""
    r = sdk_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "sdk_audit",
        "schema": "sdk_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Fx1Harness is the in-process twin of the harness HTTP API: "
            "identical registry, identical backend set and error taxonomy, "
            "honesty gate enforced before results return, spawned engines "
            "always closed, receipt verification parity with the wire "
            "surface, and health that leaks booleans only."
            if ok
            else f"SDK AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
