"""serve_audit — adversarial probes on the ``fx1.serve`` layer.

Covers ``backends`` (hosted K3 + local checkpoint gating), ``signing``
(HMAC release chain), and ``chat`` (cited_complete honesty + footer).

Pinned contract:

- ``HostedK3Backend`` refuses to construct without ``MOONSHOT_API_KEY``;
  the request body pins ``temperature: 0.0`` (deterministic evals) and the
  Authorization header carries the key (env-only, never argv).
- ``LocalFx1Backend`` is fail-closed: no ``modelcard.json`` →
  FileNotFoundError; ``ship_eligible=False`` card → RuntimeError;
  ``require_signature=True`` with no/invalid release → RuntimeError.
  ``complete`` delegates to a local OpenAI-compatible engine — attach via
  ``FX1_LOCAL_SERVE_URL``/``serve_url=`` or spawn via
  ``FX1_LOCAL_SERVE_CMD``/``serve_cmd=`` (``$checkpoint_dir``/``$python``
  substituted); unconfigured → ``BackendNotConfiguredError``, a dead or
  unready engine → RuntimeError, ``close()`` terminates the spawn.
- ``get_backend`` fails closed on unknown kinds and lists the known set.
- Signing: sign→verify roundtrip; artifact tamper, missing sig/manifest,
  wrong key, empty dir, missing dir all fail closed; manifest excludes
  its own sig/manifest files.
- ``cited_complete`` runs the honesty gate on the model output and
  appends a receipt-hash footer only when hashes are provided.

Flagged warts (documented, not fixed):

- corrupt manifest fails closed via the signature check before parsing
  (``corrupt_manifest_fails``); a *validly signed* corrupt manifest would
  still raise out of ``model_validate_json`` — unreachable without the key.
- ``flag_unsigned_default`` — without ``FX1_SIGNING_KEY`` the signature
  gate defaults off, so an unsigned-but-eligible checkpoint serves.
- ``flag_footer_unverified`` — the citation footer prints whatever hashes
  the caller supplies; no check that a receipt exists.
- ``flag_ship_eligible_not_honesty`` — a card with
  ``honesty_gate_candidate=False`` is ship-ineligible only via the
  composite; the backend refuses before reading it, which is correct.

Sealed ``serve_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from fx1.serve.backends import SamplingParams

__all__ = ["serve_audit", "serve_audit_bench"]


def serve_audit() -> dict[str, Any]:
    import json
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from fx1.modelcard import EvalDelta, ModelCard
    from fx1.serve.backends import (
        HostedK3Backend,
        LocalFx1Backend,
        OpenAICompatBackend,
        StreamingBackend,
        get_backend,
    )
    from fx1.serve.chat import cited_complete
    from fx1.serve.signing import (
        MANIFEST_FILENAME,
        SIGNATURE_FILENAME,
        build_manifest,
        sign_release,
        verify_release,
    )

    out: dict[str, Any] = {}

    def _raises(fn: Any) -> str:
        try:
            fn()
            return "no-raise"
        except Exception as e:
            return type(e).__name__

    # ---------------- HostedK3Backend -----------------------------
    saved = os.environ.pop("MOONSHOT_API_KEY", None)
    try:
        out["hosted_requires_key"] = _raises(lambda: HostedK3Backend()) == "RuntimeError"
        os.environ["MOONSHOT_API_KEY"] = "audit-key"
        backend = HostedK3Backend()
        captured: dict[str, Any] = {}

        class _Resp:
            def read(self) -> bytes:
                return json.dumps({"choices": [{"message": {"content": "answer"}}]}).encode()

            def close(self) -> None:
                return None

            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *a: Any) -> None:
                return None

        def fake_urlopen(req: Any, **kw: Any) -> _Resp:
            captured["body"] = json.loads(req.data.decode())
            captured["auth"] = req.headers.get("Authorization")
            return _Resp()

        import urllib.request  # noqa: PLC0415
        from types import SimpleNamespace as _SN  # noqa: PLC0415

        # The wire seam is ``build_opener(...).open`` — stub the opener, not
        # ``urlopen`` (a dead seam since the redirect-hardening refactor).
        with patch.object(urllib.request, "build_opener", lambda *a, **k: _SN(open=fake_urlopen)):
            text = backend.complete([{"role": "user", "content": "hi"}])
        out["hosted_temperature_zero"] = (
            text == "answer"
            and captured["body"].get("temperature") == 0.0
            and captured["auth"] == "Bearer audit-key"
        )
    finally:
        if saved is None:
            os.environ.pop("MOONSHOT_API_KEY", None)
        else:
            os.environ["MOONSHOT_API_KEY"] = saved

    # ---------------- LocalFx1Backend ------------------------------
    def _card(eligible: bool, honesty: bool = True) -> ModelCard:
        return ModelCard(
            version="fx-1.v0.1",
            corpus_sha256="a" * 64,
            corpus_receipt_range="aa..bb",
            training_manifest_sha256="b" * 64,
            eval_delta=EvalDelta(
                domain_pass_rate_base=0.5,
                domain_pass_rate_candidate=0.6 if eligible else 0.4,
                general_pass_rate_base=0.5,
                general_pass_rate_candidate=0.5,
                honesty_gate_candidate=honesty,
            ),
        )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ckpt"
        root.mkdir()
        out["local_no_card"] = _raises(lambda: LocalFx1Backend(root)) == "FileNotFoundError"
        _card(eligible=False).save(root / "modelcard.json")
        out["local_ship_gate"] = _raises(lambda: LocalFx1Backend(root)) == "RuntimeError"
        out["flag_ship_eligible_not_honesty"] = not _card(
            eligible=True, honesty=False
        ).eval_delta.ship_eligible
        _card(eligible=True).save(root / "modelcard.json")
        (root / "weights.bin").write_bytes(b"w")
        out["flag_unsigned_default"] = isinstance(LocalFx1Backend(root), LocalFx1Backend)
        out["local_unsigned_refused"] = (
            _raises(lambda: LocalFx1Backend(root, require_signature=True)) == "RuntimeError"
        )
        os.environ["FX1_SIGNING_KEY"] = "serve-audit-key"
        sign_release(root)
        ok_backend = LocalFx1Backend(root, require_signature=True)
        out["local_signed_serves"] = isinstance(ok_backend, LocalFx1Backend)

        # --- local weights-direct completion (attach/spawn engine) ---
        serve_env = [
            "FX1_LOCAL_SERVE_URL",
            "FX1_LOCAL_SERVE_CMD",
            "FX1_LOCAL_MODEL",
            "FX1_LOCAL_API_KEY",
            "FX1_LOCAL_TIMEOUT_S",
            "FX1_LOCAL_START_TIMEOUT_S",
        ]
        saved_env = {k: os.environ.pop(k, None) for k in serve_env}
        try:
            bare = LocalFx1Backend(root, require_signature=True)
            out["local_unconfigured_fails_closed"] = (
                _raises(lambda: bare.complete([])) == "BackendNotConfiguredError"
            )
            out["local_bad_url_refused"] = (
                _raises(
                    lambda: LocalFx1Backend(root, require_signature=True, serve_url="not-a-url")
                )
                == "RuntimeError"
            )
            import urllib.error  # noqa: PLC0415

            captured2: dict[str, Any] = {}

            class _Resp2:
                def read(self) -> bytes:
                    return json.dumps({"choices": [{"message": {"content": "answer"}}]}).encode()

                def close(self) -> None:
                    return None

                def __enter__(self) -> _Resp2:
                    return self

                def __exit__(self, *a: Any) -> None:
                    return None

            def fake_urlopen2(req: Any, **kw: Any) -> _Resp2:
                captured2.setdefault("urls", []).append(
                    req.full_url if hasattr(req, "full_url") else str(req)
                )
                if hasattr(req, "data") and req.data:
                    captured2["body"] = json.loads(req.data.decode())
                    captured2["auth"] = req.headers.get("Authorization")
                return _Resp2()

            import urllib.request  # noqa: PLC0415

            attached = LocalFx1Backend(
                root,
                require_signature=True,
                serve_url="http://127.0.0.1:8011/v1",
                api_key="local-key",
            )
            with patch.object(
                urllib.request, "build_opener", lambda *a, **k: _SN(open=fake_urlopen2)
            ):
                text = attached.complete([{"role": "user", "content": "hi"}])
            out["local_attach_complete"] = text == "answer"
            out["local_temperature_zero"] = captured2["body"].get("temperature") == 0.0
            out["local_auth_header"] = captured2["auth"] == "Bearer local-key"
            out["local_model_defaults_card"] = captured2["body"].get("model") == "fx-1.v0.1"
            out["local_url_normalized"] = any(
                u.endswith("/v1/chat/completions") for u in captured2["urls"]
            )
            os.environ["FX1_LOCAL_SERVE_URL"] = "http://127.0.0.1:8012"
            env_be = LocalFx1Backend(root, require_signature=True)
            with patch.object(
                urllib.request, "build_opener", lambda *a, **k: _SN(open=fake_urlopen2)
            ):
                env_be.complete([{"role": "user", "content": "hi"}])
            out["local_env_url_honored"] = any("127.0.0.1:8012" in u for u in captured2["urls"])

            # --- SSE streaming ------------------------------------------------
            class _StreamResp:
                def __init__(self, lines: list[bytes]) -> None:
                    self._lines = lines

                def __iter__(self) -> Any:
                    return iter(self._lines)

                def close(self) -> None:
                    return None

                def __enter__(self) -> _StreamResp:
                    return self

                def __exit__(self, *a: Any) -> None:
                    return None

            def fake_stream(req: Any, **kw: Any) -> _StreamResp:
                captured2["stream_body"] = json.loads(req.data.decode())
                captured2["accept"] = req.headers.get("Accept")
                return _StreamResp(
                    [
                        b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n',
                        b'data: {"choices": [{"delta": {"content": "hel"}}]}\n\n',
                        b'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n',
                        b"data: [DONE]\n\n",
                        # frames after [DONE] must never reach the consumer
                        b'data: {"choices": [{"delta": {"content": "NEVER"}}]}\n\n',
                    ]
                )

            with patch.object(
                urllib.request, "build_opener", lambda *a, **k: _SN(open=fake_stream)
            ):
                deltas = list(attached.stream([{"role": "user", "content": "hi"}]))
            out["local_stream_deltas"] = deltas == ["hel", "lo"]
            out["local_stream_wire_flag"] = captured2["stream_body"].get("stream") is True
            out["local_stream_temp_pin"] = captured2["stream_body"].get("temperature") == 0.0
            out["local_stream_accept_sse"] = captured2["accept"] == "text/event-stream"

            def fake_bad_stream(req: Any, **kw: Any) -> _StreamResp:
                return _StreamResp([b"data: {not json\n\n"])

            with patch.object(
                urllib.request, "build_opener", lambda *a, **k: _SN(open=fake_bad_stream)
            ):
                out["local_stream_malformed_fails"] = (
                    _raises(lambda: list(attached.stream([{"role": "u", "content": "x"}])))
                    == "RuntimeError"
                )
            out["local_is_streaming_backend"] = isinstance(attached, StreamingBackend)
            spawned = LocalFx1Backend(
                root,
                require_signature=True,
                serve_url="http://127.0.0.1:8013/v1",
                serve_cmd='${python} -c "import time; time.sleep(30)"',
            )
            captured2["urls"] = []

            def fake_urlopen_cold(req: Any, **kw: Any) -> _Resp2:
                # The engine reads as down only until a proc exists — the
                # spawn template must actually run (a concurrent-ready fake
                # would let the lock's re-check skip spawning entirely).
                url = req.full_url if hasattr(req, "full_url") else str(req)
                if url.endswith("/v1/models") and spawned._proc is None:
                    raise urllib.error.URLError("connection refused")
                return fake_urlopen2(req, **kw)

            # Cold-start fakes patch both seams: ``urlopen`` covers the
            # ``/v1/models`` readiness probe, ``build_opener`` the wire call.
            with (
                patch.object(urllib.request, "urlopen", fake_urlopen_cold),
                patch.object(
                    urllib.request, "build_opener", lambda *a, **k: _SN(open=fake_urlopen_cold)
                ),
            ):
                spawned.complete([{"role": "user", "content": "hi"}])
            out["local_spawn_template_ran"] = spawned._proc is not None
            spawned.close()
            out["local_close_terminates"] = spawned._proc is None
            # Real urlopen here: 127.0.0.1:9 refuses, so the spawn template
            # must run and its instant exit must surface as RuntimeError.
            dead = LocalFx1Backend(
                root,
                require_signature=True,
                serve_url="http://127.0.0.1:9/v1",
                serve_cmd='${python} -c "raise SystemExit(1)"',
                start_timeout_s=2.0,
            )
            out["local_dead_engine_fails"] = (
                _raises(lambda: dead.complete([{"role": "u", "content": "x"}])) == "RuntimeError"
            )
            dead.close()
            out["local_close_idempotent"] = True

            # Concurrency: N threads sharing one backend must spawn exactly
            # once — the engine lock makes the check-spawn sequence atomic.
            import subprocess  # noqa: PLC0415
            from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

            real_popen = subprocess.Popen
            spawn_count = [0]

            def counting_popen(*a: Any, **kw: Any) -> Any:
                spawn_count[0] += 1
                return real_popen(*a, **kw)

            shared = LocalFx1Backend(
                root,
                require_signature=True,
                serve_url="http://127.0.0.1:8014/v1",
                serve_cmd='${python} -c "import time; time.sleep(30)"',
            )

            def fake_urlopen_cold_locked(req: Any, **kw: Any) -> _Resp2:
                # The engine reads as down only until a spawn has occurred —
                # so whichever thread loses the readiness race must be the
                # one that spawns, and the lock re-check must short-circuit
                # the rest.
                url = req.full_url if hasattr(req, "full_url") else str(req)
                if url.endswith("/v1/models") and spawn_count[0] == 0:
                    raise urllib.error.URLError("connection refused")
                return fake_urlopen2(req, **kw)

            try:
                with (
                    patch.object(subprocess, "Popen", counting_popen),
                    patch.object(urllib.request, "urlopen", fake_urlopen_cold_locked),
                    patch.object(
                        urllib.request,
                        "build_opener",
                        lambda *a, **k: _SN(open=fake_urlopen_cold_locked),
                    ),
                    ThreadPoolExecutor(max_workers=4) as pool,
                ):
                    list(
                        pool.map(
                            lambda _i: shared.complete([{"role": "u", "content": "x"}]),
                            range(6),
                        )
                    )
                out["local_spawn_once_concurrent"] = spawn_count[0] == 1
            finally:
                shared.close()
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        # tamper after signing
        (root / "weights.bin").write_bytes(b"tampered")
        out["tamper_refuses"] = (
            _raises(lambda: LocalFx1Backend(root, require_signature=True)) == "RuntimeError"
        )
        # unlisted artifact added post-sign fails closed — verify_release
        # requires exact inventory, not just manifest-listed hashes
        sign_release(root)
        (root / "extra.bin").write_bytes(b"rogue")
        out["unlisted_artifact_refused"] = not verify_release(root)
        (root / "extra.bin").unlink()
        # wrong key
        os.environ["FX1_SIGNING_KEY"] = "other-key"
        out["wrong_key_fails"] = not verify_release(root)
        os.environ["FX1_SIGNING_KEY"] = "serve-audit-key"
        out["right_key_verifies"] = verify_release(root)
        # missing pieces
        (root / SIGNATURE_FILENAME).unlink()
        out["missing_sig_fails"] = not verify_release(root)
        sign_release(root)
        (root / MANIFEST_FILENAME).unlink()
        out["missing_manifest_fails"] = not verify_release(root)
        sign_release(root)
        # corrupt manifest: signature no longer matches → fail closed
        (root / MANIFEST_FILENAME).write_text("{corrupt", encoding="utf-8")
        out["corrupt_manifest_fails"] = not verify_release(root)
        (root / MANIFEST_FILENAME).unlink()
        # manifest excludes itself
        sign_release(root)
        man = build_manifest(root)
        out["manifest_excludes_self"] = SIGNATURE_FILENAME not in man.artifacts and (
            MANIFEST_FILENAME not in man.artifacts
        )
        del os.environ["FX1_SIGNING_KEY"]

        empty = Path(td) / "empty"
        empty.mkdir()
        out["empty_dir_raises"] = _raises(lambda: build_manifest(empty)) == "ValueError"
        out["missing_dir_raises"] = (
            _raises(lambda: build_manifest(Path(td) / "gone")) == "FileNotFoundError"
        )
        out["sign_no_key_raises"] = _raises(lambda: sign_release(root)) == "RuntimeError"

    out["get_backend_unknown"] = _raises(lambda: get_backend("nope")) == "KeyError"
    out["byok_is_streaming_backend"] = isinstance(
        OpenAICompatBackend(base_url="http://127.0.0.1:9/v1", api_key="k", model="m"),
        StreamingBackend,
    )
    saved_k3 = os.environ.get("MOONSHOT_API_KEY")
    os.environ["MOONSHOT_API_KEY"] = "probe-key"
    try:
        out["hosted_is_streaming_backend"] = isinstance(HostedK3Backend(), StreamingBackend)
    finally:
        if saved_k3 is None:
            os.environ.pop("MOONSHOT_API_KEY", None)
        else:
            os.environ["MOONSHOT_API_KEY"] = saved_k3

    # ---------------- chat.cited_complete ------------------------
    class _Echo:
        def __init__(self, text: str) -> None:
            self._t = text

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            return self._t

    cited = cited_complete(_Echo("fine answer"), [], receipt_hashes=["a" * 64, "b" * 64])
    out["cited_footer"] = "Evidence:" in cited and "aaaaaaaaaaaaaaaa" in cited
    out["cited_no_footer"] = "Evidence:" not in cited_complete(_Echo("x"), [])
    out["cited_honesty_blocks"] = (
        _raises(lambda: cited_complete(_Echo("Sharpe 9.9"), [])) == "Fx1HonestyError"
    )
    out["flag_footer_unverified"] = "ffffffff" in cited_complete(
        _Echo("x"),
        [],
        receipt_hashes=["f" * 64],
    )

    return out


def serve_audit_bench() -> dict[str, Any]:
    r = serve_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "serve_audit",
        "schema": "serve_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "serve layer holds: hosted K3 needs env credentials and pins "
            "temperature=0; local checkpoints fail closed without a card, "
            "on a failed ship gate, and on missing/invalid signatures when "
            "required; the release chain detects tamper and wrong keys. "
            "Local weights serve through an OpenAI-compatible engine — "
            "attach or spawn, fail-closed when neither is configured; the "
            "spawn template's checkpoint path is the verified root. "
            "Flags: unlisted post-sign artifacts pass verification "
            "(manifest-scoped, not closed-world); signature gate defaults "
            "off without FX1_SIGNING_KEY; citation footer prints "
            "unverified hashes."
            if ok
            else f"SERVE AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
