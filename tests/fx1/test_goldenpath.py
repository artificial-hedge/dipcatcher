"""Regression for scripts/fx1_goldenpath.py — the end-to-end golden-path lane.

Runs the real driver subprocess (it itself boots a real ``fx1 harness
serve``), then asserts the sealed ``goldenpath.v1`` verifies and every leg
that ran passed. Legs gated on optional-but-absent pieces (e.g. no
anthropic package) stay honest ``ran: false`` gaps; the committed
``artifacts/fx1_tiny_lm`` checkpoint makes ``weights_direct_ran`` a real
measured leg, not a gap.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # noqa: S404 — the driver is itself a subprocess harness
import sys
from pathlib import Path

import pytest

from quant_fund.research.receipt_v2 import verify_receipt_file

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "fx1_goldenpath.py"
_CKPT = _ROOT / "artifacts" / "fx1_tiny_lm"


def test_goldenpath_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "goldenpath.v1.json"
    proc = subprocess.run(  # noqa: S603 — argv is a fixed local list
        [
            sys.executable,
            str(_SCRIPT),
            "--out",
            str(out),
            "--work-dir",
            str(tmp_path / "work"),
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(_ROOT / "src")),
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    verdict = verify_receipt_file(out)
    assert verdict["valid"], verdict["errors"]
    receipt = json.loads(out.read_text())
    claim = receipt["claim"]
    results = claim["results"]

    # every leg that really ran succeeded — honest gaps are excluded by design
    assert claim["runnable_ok"], claim["gaps"]
    for name, entry in results.items():
        if entry["ran"]:
            assert entry["ok"], f"{name}: {entry['detail']}"
        else:
            assert entry["detail"], f"{name}: honest gaps must carry a detail"

    # weights-direct is a measured leg now: a real committed weight artifact
    # loaded in-process AND served over the wire — not a recorded gap.
    assert results["weights_direct_ran"]["ran"] is True
    assert results["weights_direct_ran"]["ok"] is True
    assert "weights_direct_ran" not in claim["gaps"]
    assert results["local_fx1_link_ran"]["ok"] is True

    # the non-negotiable legs all ran for real
    for must in (
        "serve_subprocess_boots",
        "harness_client_surface",
        "inprocess_sdk_parity",
        "local_fx1_link_ran",
        "eval_suite_terminal",
        "eval_diff_ran",
        "webhook_signed_delivery",
        "finetune_submitted",
        "finetune_terminal",
        "sigkill_recovery",
        "wire_receipts_verify",
        "goldenpath_receipt_verifies",
    ):
        assert results[must]["ran"], f"{must} did not run: {results[must]}"
        assert results[must]["ok"], f"{must}: {results[must]['detail']}"


def test_weights_direct_checkpoint_loads_and_serves(tmp_path: Path) -> None:
    """The committed checkpoint really loads: card + manifest + safetensors.

    In-process ``complete_messages`` must be deterministic, non-empty, and
    honesty-gate clean; served over a real loopback socket it returns the
    same bytes; a byte-flipped weights file or a cardless dir refuses to
    load — the weights-direct leg can never pass on a stub.
    """
    pytest.importorskip("safetensors.numpy", reason="nn extra not installed")
    from fx1.honesty import validate_fx1_output
    from fx1.serve.local_engine import LocalWeightsEngine

    engine = LocalWeightsEngine(_CKPT)
    assert engine.served_model == "fx-1.v0.1"
    msgs = [{"role": "user", "content": "weights-direct-ping"}]
    a = engine.complete_messages(msgs)
    b = engine.complete_messages(msgs)
    assert a.text
    assert a.text == b.text
    assert a.prompt_tokens > 0
    assert a.completion_tokens > 0
    validate_fx1_output(a.text)  # what the wire ships must clear the gate

    manifest = json.loads((_CKPT / "weights.manifest.json").read_text())
    assert manifest["weights_sha256"] == engine.weights_sha256
    assert manifest["arch"]["n_params"] == engine.model.n_params

    # served over a real loopback socket: /v1/models advertises the loaded
    # sha, /v1/chat/completions returns the same bytes as direct generation
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from fx1.serve.local_engine import _EngineHTTPServer, _Handler

    srv: ThreadingHTTPServer = _EngineHTTPServer(("127.0.0.1", 0), _Handler)
    srv.engine = engine
    port = int(srv.server_address[1])
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:  # noqa: S310 — loopback test server  # nosec B310
            models = json.loads(r.read().decode())
        assert models["data"][0]["fx1"]["weights_sha256"] == engine.weights_sha256
        body = json.dumps({"model": "fx-1.v0.1", "messages": msgs, "temperature": 0.0}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310 — loopback test server  # nosec B310
            done = json.loads(r.read().decode())
        assert done["choices"][0]["message"]["content"] == a.text
        assert done["usage"]["total_tokens"] == a.prompt_tokens + a.completion_tokens
        assert done["usage"]["prompt_tokens"] == a.prompt_tokens
    finally:
        srv.shutdown()
        srv.server_close()

    # byte-flipped weights refuse to serve — the sha pin is fail-closed
    bad = tmp_path / "tampered"
    shutil.copytree(_CKPT, bad)
    blob = bytearray((bad / "weights.safetensors").read_bytes())
    blob[-1] ^= 0xFF
    (bad / "weights.safetensors").write_bytes(bytes(blob))
    with pytest.raises(RuntimeError, match="integrity pin"):
        LocalWeightsEngine(bad)

    # a card-only checkpoint dir (the pre-artifact state) refuses too
    cardless = tmp_path / "cardless"
    cardless.mkdir()
    with pytest.raises(FileNotFoundError):
        LocalWeightsEngine(cardless)
