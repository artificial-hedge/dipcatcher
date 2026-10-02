"""serve_audit — adversarial probes on the ``fx1.serve`` layer.

Covers ``backends`` (hosted K3 + local checkpoint gating), ``signing``
(HMAC release chain), and ``chat`` (cited_complete honesty + footer).

Pinned contract:

- ``HostedK3Backend`` refuses to construct without ``MOONSHOT_API_KEY``;
  the request body pins ``temperature: 0.0`` (deterministic evals) and the
  Authorization header carries the key (env-only, never argv).
- ``LocalFx1Backend`` is fail-closed: no ``modelcard.json`` →
  FileNotFoundError; ``ship_eligible=False`` card → RuntimeError;
  ``require_signature=True`` with no/invalid release → RuntimeError;
  ``complete`` raises NotImplementedError until the serving stack ships.
- ``get_backend`` fails closed on unknown kinds and lists the known set.
- Signing: sign→verify roundtrip; artifact tamper, missing sig/manifest,
  wrong key, empty dir, missing dir all fail closed; manifest excludes
  its own sig/manifest files.
- ``cited_complete`` runs the honesty gate on the model output and
  appends a receipt-hash footer only when hashes are provided.

Flagged warts (documented, not fixed):

- ``flag_unlisted_artifact_passes`` — ``verify_release`` only re-hashes
  manifest-listed artifacts; a file *added* to the checkpoint dir after
  signing verifies clean (manifest-scoped, not closed-world).
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
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

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

            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *a: Any) -> None:
                return None

        def fake_urlopen(req: Any, **kw: Any) -> _Resp:
            captured["body"] = json.loads(req.data.decode())
            captured["auth"] = req.headers.get("Authorization")
            return _Resp()

        import urllib.request  # noqa: PLC0415

        with patch.object(urllib.request, "urlopen", fake_urlopen):
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
        out["local_complete_unimplemented"] = (
            _raises(lambda: ok_backend.complete([])) == "NotImplementedError"
        )
        # tamper after signing
        (root / "weights.bin").write_bytes(b"tampered")
        out["tamper_refuses"] = (
            _raises(lambda: LocalFx1Backend(root, require_signature=True)) == "RuntimeError"
        )
        # unlisted artifact added post-sign verifies clean (manifest-scoped)
        sign_release(root)
        (root / "extra.bin").write_bytes(b"rogue")
        out["flag_unlisted_artifact_passes"] = verify_release(root)
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

    # ---------------- chat.cited_complete ------------------------
    class _Echo:
        def __init__(self, text: str) -> None:
            self._t = text

        def complete(self, messages: list[dict[str, str]]) -> str:
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
