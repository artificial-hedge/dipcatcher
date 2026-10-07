"""chat_signing_audit — cited completions + detached release attestation.

Two tightly-scoped serving contracts that every other surface leans on:

``chat.py`` — the honesty-gated completion wrapper. ``sampling.stop``
truncates *before* the gate (shipped bytes are validated bytes);
``validate_fx1_output`` fails closed on forbidden headlines *inside* the
wrapper so no caller can ship an ungated string; the provenance footer
appends truncated receipt hashes only after the gate. The tool half:
``complete_with_tools`` absence fails closed (NotImplementedError →
501 semantics), the gate reads ``content`` only — ``tool_calls``
arguments are machine-bound JSON and ``logprobs`` are provider scores,
never claims — and an untouched completion returns the *same* object
identity while a gated one preserves tool_calls/finish_reason/logprobs.

``signing.py`` — detached HMAC release attestation. Signing covers the
complete regular-file inventory minus the two root metadata files;
verification authenticates before hashing, compares manifest path keys
against a fresh enumeration (never opens manifest-declared paths), and
fails closed on every deviation: tampered bytes, added/removed files,
forged manifest, forged signature, missing metadata, symlinked
artifacts/dirs, non-regular files, missing key.

Probes are literal bools; the sealed receipt names every defect found.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fx1.honesty import Fx1HonestyError
from fx1.serve.backends import SamplingParams, ToolCompletion
from fx1.serve.chat import cited_complete, cited_complete_tools
from fx1.serve.signing import (
    _MAX_MANIFEST_BYTES,
    _MAX_SIGNATURE_BYTES,
    MANIFEST_FILENAME,
    SIGNATURE_FILENAME,
    SIGNING_KEY_ENV,
    ReleaseManifest,
    build_manifest,
    sign_release,
    verify_release,
)

__all__ = ["chat_signing_audit", "chat_signing_audit_bench"]

_FORBIDDEN = "our sharpe is 2.0"
_CLEAN = "CRPS improved by 4%"


@dataclass
class _StubBackend:
    response: str = _CLEAN
    tool_result: ToolCompletion | None = None
    seen_sampling: SamplingParams | None = None
    seen_kwargs: dict[str, Any] | None = None
    _complete_with_tools_present: bool = True

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str:
        self.seen_sampling = sampling
        return self.response

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ToolCompletion:
        self.seen_kwargs = kwargs
        assert self.tool_result is not None
        return self.tool_result


class _PlainBackend:
    """Complete-only backend: no ``complete_with_tools`` channel."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str:
        return _CLEAN


def _refuses(fn: Any, *args: Any, **kwargs: Any) -> bool:
    try:
        fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 — refuse probes accept any failure
        return True
    return False


def _probe_cited() -> dict[str, bool]:
    out: dict[str, bool] = {}
    msgs = [{"role": "user", "content": "q"}]
    out["cc_passthrough_clean"] = cited_complete(_StubBackend(), msgs) == _CLEAN
    backend = _StubBackend()
    out["cc_sampling_forwarded"] = (
        cited_complete(backend, msgs, sampling=SamplingParams(stop=("##",))) == _CLEAN
        and backend.seen_sampling is not None
        and backend.seen_sampling.stop == ("##",)
    )
    # harness-side truncation applies to the shipped text
    backend = _StubBackend(response="alpha##beta")
    out["cc_stop_truncates"] = (
        cited_complete(backend, msgs, sampling=SamplingParams(stop=("##",))) == "alpha"
    )
    # gate sees post-truncation text: a forbidden tail is cut, then it passes
    backend = _StubBackend(response="fine.##" + _FORBIDDEN)
    out["cc_stop_before_gate"] = (
        cited_complete(backend, msgs, sampling=SamplingParams(stop=("##",))) == "fine."
    )
    out["cc_forbidden_fails_closed"] = _refuses(
        cited_complete, _StubBackend(response=_FORBIDDEN), msgs
    )
    try:
        cited_complete(_StubBackend(response=_FORBIDDEN), msgs)
    except Fx1HonestyError:
        out["cc_forbidden_is_honesty_error"] = True
    except Exception:  # noqa: BLE001
        out["cc_forbidden_is_honesty_error"] = False
    hashes = ["a" * 64, "b" * 64]
    cited = cited_complete(_StubBackend(), msgs, receipt_hashes=hashes)
    out["cc_footer_appended"] = cited.startswith(_CLEAN + "\n\nEvidence: ")
    out["cc_footer_truncates_hashes"] = "`aaaaaaaaaaaaaaaa…`" in cited and "a" * 17 not in cited
    out["cc_footer_verify_hint"] = "verify-research" in cited
    out["cc_no_hashes_no_footer"] = "Evidence:" not in cited_complete(_StubBackend(), msgs)
    out["cc_empty_hashes_no_footer"] = "Evidence:" not in cited_complete(
        _StubBackend(), msgs, receipt_hashes=[]
    )
    # gate precedes footer: forbidden body never reaches the footer line
    out["cc_gate_precedes_footer"] = _refuses(
        cited_complete,
        _StubBackend(response=_FORBIDDEN),
        msgs,
        receipt_hashes=hashes,
    )
    return out


def _probe_tools() -> dict[str, bool]:
    out: dict[str, bool] = {}
    msgs = [{"role": "user", "content": "q"}]
    out["ct_missing_channel_fails_closed"] = _refuses(cited_complete_tools, _PlainBackend(), msgs)
    try:
        cited_complete_tools(_PlainBackend(), msgs)
    except NotImplementedError as e:
        out["ct_failure_names_backend"] = "_PlainBackend" in str(e)
    except Exception:  # noqa: BLE001
        out["ct_failure_names_backend"] = False

    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    result = ToolCompletion(content=_CLEAN, tool_calls=(call,), finish_reason="tool_calls")
    backend = _StubBackend(tool_result=result)
    returned = cited_complete_tools(backend, msgs, tools=[{"t": 1}])
    out["ct_identity_when_untouched"] = returned is result
    out["ct_kwargs_forwarded"] = backend.seen_kwargs == {
        "sampling": None,
        "tools": [{"t": 1}],
        "tool_choice": None,
        "parallel_tool_calls": None,
        "logprobs": None,
        "top_logprobs": None,
    }
    # forbidden inside tool-call arguments is machine JSON — not gated
    evil_call = {
        "id": "c2",
        "type": "function",
        "function": {"name": "f", "arguments": f'{{"x": "{_FORBIDDEN}"}}'},
    }
    backend = _StubBackend(
        tool_result=ToolCompletion(
            content="ok", tool_calls=(evil_call,), finish_reason="tool_calls"
        )
    )
    out["ct_tool_args_not_text_gate"] = cited_complete_tools(backend, msgs).tool_calls == (
        evil_call,
    )
    # forbidden in content IS gated
    backend = _StubBackend(
        tool_result=ToolCompletion(content=_FORBIDDEN, tool_calls=(call,), finish_reason="stop")
    )
    out["ct_content_gated"] = _refuses(cited_complete_tools, backend, msgs)
    # stop truncation applies to content before the gate
    backend = _StubBackend(
        tool_result=ToolCompletion(
            content="fine.##" + _FORBIDDEN, tool_calls=None, finish_reason="stop"
        )
    )
    got = cited_complete_tools(backend, msgs, sampling=SamplingParams(stop=("##",)))
    out["ct_stop_truncates"] = got.content == "fine."
    # logprobs ride through verbatim
    backend = _StubBackend(
        tool_result=ToolCompletion(
            content=_CLEAN,
            tool_calls=None,
            finish_reason="stop",
            logprobs={"content": [{"token": "a", "logprob": -0.1}]},
        )
    )
    out["ct_logprobs_passthrough"] = cited_complete_tools(
        backend, msgs, logprobs=True, top_logprobs=3
    ).logprobs == {"content": [{"token": "a", "logprob": -0.1}]}
    backend = _StubBackend(
        tool_result=ToolCompletion(content=_CLEAN, tool_calls=None, finish_reason="stop")
    )
    out["ct_logprobs_none_stays_none"] = cited_complete_tools(backend, msgs).logprobs is None
    # content None + hashes → footer-only content in a fresh ToolCompletion
    backend = _StubBackend(
        tool_result=ToolCompletion(content=None, tool_calls=(call,), finish_reason="tool_calls")
    )
    got = cited_complete_tools(backend, msgs, receipt_hashes=["c" * 64])
    out["ct_none_content_footer"] = (
        got.content is not None
        and got.content.startswith("\n\nEvidence: ")
        and got.tool_calls == (call,)
        and got.finish_reason == "tool_calls"
    )
    # content None + no hashes → untouched identity
    got = cited_complete_tools(backend, msgs)
    out["ct_none_content_identity"] = got is backend.tool_result
    return out


def _probe_manifest(tmp: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    root = tmp / "ckpt"
    (root / "sub").mkdir(parents=True)
    (root / "weights.bin").write_bytes(b"weights" * 64)
    (root / "sub" / "config.json").write_text("{}", encoding="utf-8")
    m = build_manifest(root)
    out["mn_paths_relative"] = set(m.artifacts) == {
        "weights.bin",
        "sub/config.json",
    }
    out["mn_hashes_real"] = (
        m.artifacts["weights.bin"] == hashlib.sha256(b"weights" * 64).hexdigest()
    )
    out["mn_dir_recorded"] = m.checkpoint_dir == str(root)
    (root / SIGNATURE_FILENAME).write_text("deadbeef", encoding="utf-8")
    (root / MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    m2 = build_manifest(root)
    out["mn_metadata_excluded"] = set(m2.artifacts) == set(m.artifacts)
    empty = tmp / "empty"
    empty.mkdir()
    out["mn_empty_refused"] = _refuses(build_manifest, empty)
    out["mn_missing_refused"] = _refuses(build_manifest, tmp / "nope")
    # nested metadata name under a subdir is a real artifact, not excluded
    (root / "sub" / SIGNATURE_FILENAME).write_text("nested", encoding="utf-8")
    out["mn_nested_sig_included"] = f"sub/{SIGNATURE_FILENAME}" in build_manifest(root).artifacts
    (tmp / "outside.txt").write_bytes(b"o")
    (tmp / "bad").mkdir()
    (tmp / "bad" / "evil.bin").symlink_to(tmp / "outside.txt")
    out["mn_symlink_refused"] = _refuses(build_manifest, tmp / "bad")
    fifo_dir = tmp / "fifod"
    fifo_dir.mkdir()
    os.mkfifo(fifo_dir / "p")
    out["mn_fifo_refused"] = _refuses(build_manifest, fifo_dir)
    return out


def _probe_sign_verify(tmp: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    root = tmp / "rel"
    (root / "sub").mkdir(parents=True)
    (root / "w.bin").write_bytes(b"payload" * 128)
    (root / "sub" / "c.json").write_text('{"a": 1}', encoding="utf-8")

    key = "audit-signing-key"
    saved = os.environ.get(SIGNING_KEY_ENV)
    os.environ.pop(SIGNING_KEY_ENV, None)
    try:
        out["sv_no_key_fails_closed"] = _refuses(sign_release, root)
        out["sv_no_key_nothing_written"] = not (root / MANIFEST_FILENAME).exists()
        os.environ[SIGNING_KEY_ENV] = key
        sig_path = sign_release(root)
        out["sv_sig_written"] = sig_path == root / SIGNATURE_FILENAME
        manifest_bytes = (root / MANIFEST_FILENAME).read_bytes()
        expected = hmac.new(key.encode(), manifest_bytes, hashlib.sha256).hexdigest()
        out["sv_sig_is_hmac"] = (root / SIGNATURE_FILENAME).read_text() == expected
        out["sv_roundtrip_true"] = verify_release(root) is True

        # tampered artifact byte
        (root / "w.bin").write_bytes(b"payload" * 127 + b"x")
        out["sv_tamper_detected"] = verify_release(root) is False
        sign_release(root)
        # added file breaks the inventory
        (root / "extra.bin").write_bytes(b"e")
        out["sv_added_detected"] = verify_release(root) is False
        (root / "extra.bin").unlink()
        sign_release(root)
        # removed file breaks the inventory
        (root / "sub" / "c.json").unlink()
        out["sv_removed_detected"] = verify_release(root) is False
        (root / "sub" / "c.json").write_text('{"a": 1}', encoding="utf-8")
        sign_release(root)
        # forged manifest (valid JSON, different digest list)
        forged = ReleaseManifest(checkpoint_dir=str(root), artifacts={"w.bin": "0" * 64})
        (root / MANIFEST_FILENAME).write_bytes(forged.model_dump_json().encode())
        out["sv_forged_manifest_detected"] = verify_release(root) is False
        sign_release(root)
        # forged signature
        (root / SIGNATURE_FILENAME).write_text("f" * 64, encoding="utf-8")
        out["sv_forged_sig_detected"] = verify_release(root) is False
        sign_release(root)
        # corrupt manifest JSON → fail closed, not raise
        (root / MANIFEST_FILENAME).write_bytes(b"{not json")
        out["sv_corrupt_manifest_closed"] = verify_release(root) is False
        # manifest declaring a traversal path never opens it — key mismatch
        evil = ReleaseManifest(checkpoint_dir=str(root), artifacts={"../../etc/passwd": "x"})
        (root / MANIFEST_FILENAME).write_bytes(evil.model_dump_json().encode())
        expected = hmac.new(
            key.encode(),
            (root / MANIFEST_FILENAME).read_bytes(),
            hashlib.sha256,
        ).hexdigest()
        (root / SIGNATURE_FILENAME).write_text(expected, encoding="utf-8")
        out["sv_traversal_not_opened"] = verify_release(root) is False
        sign_release(root)
        # metadata missing entirely
        (root / SIGNATURE_FILENAME).unlink()
        out["sv_missing_sig_false"] = verify_release(root) is False
        sign_release(root)
        (root / MANIFEST_FILENAME).unlink()
        out["sv_missing_manifest_false"] = verify_release(root) is False
        sign_release(root)
        # verify with metadata present but no key configured → RuntimeError
        os.environ.pop(SIGNING_KEY_ENV, None)
        try:
            verify_release(root)
        except RuntimeError:
            out["sv_verify_no_key_raises"] = True
        except Exception:  # noqa: BLE001
            out["sv_verify_no_key_raises"] = False
        os.environ[SIGNING_KEY_ENV] = key
        # artifact swapped for a symlink after signing → closed
        (root / "w.bin").unlink()
        (root / "w.bin").symlink_to(root / "sub" / "c.json")
        out["sv_symlink_swap_closed"] = verify_release(root) is False
        (root / "w.bin").unlink()
        (root / "w.bin").write_bytes(b"payload" * 128)
        sign_release(root)
        out["sv_restored_true"] = verify_release(root) is True
    finally:
        if saved is None:
            os.environ.pop(SIGNING_KEY_ENV, None)
        else:
            os.environ[SIGNING_KEY_ENV] = saved
    out["sv_limits_sane"] = _MAX_MANIFEST_BYTES > 0 and _MAX_SIGNATURE_BYTES > 0
    out["sv_metadata_names"] = MANIFEST_FILENAME.endswith(".json") and SIGNATURE_FILENAME.endswith(
        ".sig"
    )
    return out


def chat_signing_audit() -> dict[str, bool]:
    """Every cited-completion + release-signing contract as booleans."""
    out: dict[str, bool] = {}
    out.update(_probe_cited())
    out.update(_probe_tools())
    with tempfile.TemporaryDirectory() as tmp:
        out.update(_probe_manifest(Path(tmp)))
        out.update(_probe_sign_verify(Path(tmp)))
    return out


def chat_signing_audit_bench() -> dict[str, Any]:
    """Sealed receipt for the cited-completion + signing battery."""
    r = chat_signing_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "chat_signing_audit",
        "schema": "chat_signing_audit.v1",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process stub backend + real temp checkpoint trees",
            "not_verified": [
                "Sigstore/TEE attestation tiers (documented as out of scope)",
                "concurrent-mutation discipline (documented unsynchronized)",
            ],
        },
        "interpretation": (
            "Cited completions gate post-truncation text, append "
            "provenance footers only after the gate, refuse closed on "
            "missing tool channels, and preserve provider metadata; "
            "release signing authenticates before hashing and fails "
            "closed on every inventory/signature/parse deviation."
            if ok
            else f"CHAT/SIGNING AUDIT DEFECTS: {defects}"
        ),
    }
    from quant_fund.utils.reproducibility import git_revision

    out["git_revision"] = git_revision()
    from quant_fund.research.receipt_v2 import canonical_json_bytes
    from quant_fund.utils.hashing import hash_bytes

    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(chat_signing_audit_bench(), indent=2, sort_keys=True))
