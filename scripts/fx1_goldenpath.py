"""fx-1 golden path — the end-to-end proof lane.

Boots a real ``fx1 harness serve`` subprocess on a real port with the
closest-to-real backend that exists in this checkout, then drives the whole
product story through it and seals the run as ``goldenpath.v1``:

* serve subprocess — real uvicorn, real middleware, real ``--state-dir``
  journals (no in-process shortcuts for the wire legs);
* four client surfaces — the stock ``openai`` SDK, the stock ``anthropic``
  SDK, ``HarnessClient``, and the in-process ``Fx1Harness`` twin — all
  pointed at the same booted server (the SDK twin answers without a socket);
* the three consumption modes, honestly: ``byok`` end-to-end against a real
  OpenAI-compatible stub engine, ``local_fx1`` end-to-end over the wire
  against a card-only checkpoint dir, and ``weights_direct_ran`` pinned
  ``false`` — no fx-1 weight artifact exists anywhere in this repository;
* an eval suite submitted over ``/harness/evals``, polled to terminal, its
  sealed receipt fetched and verified, and two runs diffed through the
  promotion-gate diff route;
* the async surface — a real OpenAI ``/v1/batches`` run, a fine-tune job
  carried to its honest terminal state, and a signed terminal webhook
  delivered to a loopback sink;
* durability — SIGKILL mid-flight, a fresh ``fx1 harness serve`` on the same
  ``--state-dir``, and every pre-crash record still resolving;
* evidence — every wire receipt written to disk and re-verified by
  ``verify_receipt_file``, then this run's own ``goldenpath.v1`` sealed and
  verified the same way.

Honest gaps are the deliverable, not failures: any leg that cannot run for
real reports ``{"ran": false, "ok": false, "detail": ...}`` and lands in
``claim.gaps`` instead of pretending.

Usage::

    uv run --no-sync python scripts/fx1_goldenpath.py [--out PATH] [--work-dir DIR] [--keep]

Exit code: 0 when every leg that ran succeeded (honest gaps excluded), 2
otherwise. The sealed receipt is written to ``--out`` (default
``receipts/fx1_goldenpath.json``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_PROBE_KEY = (
    "goldenpath-changeme-probe"  # obvious placeholder token — a probe string, not a credential
)
_WEBHOOK_SECRET = "goldenpath-webhook-secret"  # a probe string, not a credential
_ENV_KEYS = (
    "FX1_API_KEY",
    "FX1_BYOK_BASE_URL",
    "FX1_BYOK_API_KEY",
    "FX1_BYOK_MODEL",
    "FX1_BYOK_ALLOW_PRIVATE_NETWORKS",
    "FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS",
    "FX1_CHECKPOINT_DIR",
    "FX1_LOCAL_SERVE_URL",
    "FX1_LOCAL_SERVE_CMD",
    "FX1_LOCAL_MODEL",
    "FX1_SIGNING_KEY",
    "MOONSHOT_API_KEY",
)
# Model-weight artifact suffixes only — `.npz` is deliberately excluded
# (in-repo .npz files are metrics dumps like `*.losses.npz`, never weights).
_WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".ckpt", ".gguf", ".bin", ".h5"}


@dataclass
class _Legs:
    """Per-leg results: ``ran`` (the leg really executed), ``ok`` (it did what
    it claimed), ``detail`` (what happened, or why it could not run)."""

    order: list[str] = field(default_factory=list)
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record(self, name: str, ran: bool, ok: bool, detail: str = "") -> bool:
        self.order.append(name)
        self.entries[name] = {"ran": ran, "ok": ok, "detail": detail}
        return ran and ok


class _SinkHandler:
    """Collects webhook deliveries: (path, headers, raw body)."""

    hits: list[tuple[str, dict[str, str], bytes]]

    def __init__(self) -> None:
        self.hits = []


def _sink_handler(sink: _SinkHandler) -> type:
    from http.server import BaseHTTPRequestHandler

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 — stdlib hook name
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n)
            sink.hits.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args: Any) -> None:
            return None

    return _Handler


def _free_port() -> int:
    import socket

    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _wait_health(base: str, timeout_s: float) -> str:
    """Poll GET /health until 200; returns "" on success else the last error."""
    deadline = time.monotonic() + timeout_s
    err = "no attempt"
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{base}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:  # noqa: S310 — loopback URL built by this module  # nosec B310
                if resp.status == 200:
                    return ""
                err = f"health http {resp.status}"
        except Exception as exc:  # noqa: BLE001 — readiness poll reports, never escapes
            err = f"{type(exc).__name__}: {exc}"
        time.sleep(0.1)
    return err


def _spawn_serve(state_dir: Path, port: int, env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(  # noqa: S603 — argv is a fixed list built here  # nosec B603
        [
            sys.executable,
            "-m",
            "fx1.cli",
            "harness",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--state-dir",
            str(state_dir),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _all_weight_hits() -> list[str]:
    """Weight-artifact candidates anywhere in the checkout (tracked or not)."""
    return sorted(
        str(p.relative_to(_REPO_ROOT))
        for p in _REPO_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in _WEIGHT_SUFFIXES
    )


def _make_checkpoint(ckpt: Path) -> None:
    """Write a real, ship-gate-passing model card — the checkpoint dir is real,
    the weight tensors it would point at do not exist (recorded honestly)."""
    from fx1.modelcard import EvalDelta, ModelCard

    ckpt.mkdir(parents=True, exist_ok=True)
    ModelCard(
        version="fx-1.v0.1",
        corpus_sha256="a" * 64,
        corpus_receipt_range="b5942241..f0e1d2c3",
        training_manifest_sha256="b" * 64,
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.5,
            domain_pass_rate_candidate=0.7,
            general_pass_rate_base=0.9,
            general_pass_rate_candidate=0.9,
            honesty_gate_candidate=True,
        ),
    ).save(ckpt / "modelcard.json")


def _openai_legs(legs: _Legs, base: str, api_key: str) -> None:
    try:
        import openai
    except ImportError:
        for name in ("openai_sdk_models", "openai_sdk_chat", "openai_sdk_stream"):
            legs.record(name, False, False, "openai package not installed")
        return
    client = openai.OpenAI(base_url=f"{base}/v1", api_key=api_key, timeout=30.0)
    try:
        models = client.models.list()
        legs.record(
            "openai_sdk_models",
            True,
            any(m.id in {"fx1", "byok", "local_fx1"} for m in models.data),
            f"models={[m.id for m in models.data][:8]}",
        )
        done = client.chat.completions.create(
            model="byok", messages=[{"role": "user", "content": "openai-ping"}]
        )
        content = done.choices[0].message.content or ""
        legs.record(
            "openai_sdk_chat",
            True,
            "stub:openai-ping" in content,
            f"model={done.model} content={content[:48]!r}",
        )
        chunks = client.chat.completions.create(
            model="byok",
            messages=[{"role": "user", "content": "openai-stream"}],
            stream=True,
        )
        text = "".join(c.choices[0].delta.content for c in chunks if c.choices[0].delta.content)
        legs.record(
            "openai_sdk_stream",
            True,
            "stub:openai-stream" in text,
            f"reassembled={text[:48]!r}",
        )
    except Exception as exc:  # noqa: BLE001
        legs.record("openai_sdk_chat", True, False, f"{type(exc).__name__}: {exc}")


def _anthropic_legs(legs: _Legs, base: str, api_key: str) -> None:
    try:
        import anthropic
    except ImportError:
        for name in ("anthropic_sdk_messages", "anthropic_sdk_stream"):
            legs.record(name, False, False, "anthropic package not installed")
        return
    client = anthropic.Anthropic(base_url=base, api_key=api_key, timeout=30.0)
    try:
        msg = client.messages.create(
            model="byok",
            max_tokens=64,
            messages=[{"role": "user", "content": "anthropic-ping"}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        legs.record(
            "anthropic_sdk_messages",
            True,
            "anthropic-ping" in text,
            f"model={msg.model} stop={msg.stop_reason} text={text[:48]!r}",
        )
        with client.messages.stream(
            model="byok",
            max_tokens=64,
            messages=[{"role": "user", "content": "anthropic-stream"}],
        ) as stream:
            streamed = "".join(stream.text_stream)
        legs.record(
            "anthropic_sdk_stream",
            True,
            "anthropic-stream" in streamed,
            f"reassembled={streamed[:48]!r}",
        )
    except Exception as exc:  # noqa: BLE001
        legs.record("anthropic_sdk_messages", True, False, f"{type(exc).__name__}: {exc}")


def _harness_client_legs(legs: _Legs, base: str, api_key: str, timeout_s: float) -> str:
    """The HarnessClient surface; returns the submitted doctor job id."""
    from fx1.serve.client import HarnessClient

    client = HarnessClient(base, api_key=api_key, timeout_s=timeout_s)
    health = client.health()
    cmds = client.commands()
    done = client.complete([{"role": "user", "content": "hc-ping"}], backend="byok")
    chunks = "".join(
        client.stream_complete([{"role": "user", "content": "hc-stream"}], backend="byok")
    )
    idem_key = "goldenpath-idem"
    a = client.submit_run("doctor", idempotency_key=idem_key)
    b = client.submit_run("doctor", idempotency_key=idem_key)
    legs.record(
        "harness_client_surface",
        True,
        health.backends.get("byok") is True
        and len(cmds) > 0
        and done.content.startswith("stub:hc-ping")
        and "stub:hc-stream" in chunks
        and a == b,
        f"backends={health.backends} commands={len(cmds)} idem_replay={a == b}",
    )
    return a


def _wait_terminal(fetch: Any, ident: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = dict(fetch(ident))
        if last.get("status") not in (
            "queued",
            "running",
            "validating",
            "validating_files",
            "in_progress",
            "finalizing",
        ):
            return last
        time.sleep(0.25)
    return last


def _leg_detail(obj: Any, limit: int = 180) -> str:
    try:
        return json.dumps(obj, default=str)[:limit]
    except Exception:  # noqa: BLE001
        return str(obj)[:limit]


def run_goldenpath(
    *,
    work_dir: Path | None = None,
    out: Path | None = None,
    keep: bool = False,
    boot_timeout_s: float = 60.0,
    leg_timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Run the whole golden path; returns the sealed ``goldenpath.v1`` dict.

    ``work_dir`` holds the serve state dir, the checkpoint dir, the webhook
    sink log and the exported receipts; ``out`` is where the sealed receipt
    lands (default ``receipts/fx1_goldenpath.json``).
    """
    from quant_fund.research.receipt_v2 import verify_receipt_file, verify_receipt_payload
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    legs = _Legs()
    owned_dir = work_dir is None
    root = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="fx1gp_"))
    root.mkdir(parents=True, exist_ok=True)
    state_dir = root / "state"
    ckpt_dir = root / "checkpoint"
    receipts_dir = root / "wire_receipts"
    receipts_dir.mkdir(exist_ok=True)
    out_path = Path(out) if out is not None else _REPO_ROOT / "receipts" / "fx1_goldenpath.json"

    # --- real checkpoint dir (card only — no weight tensors exist in-repo) ---
    weight_hits = _all_weight_hits()
    _make_checkpoint(ckpt_dir)
    legs.record(
        "weights_direct_ran",
        False,
        False,
        "no loadable fx-1 checkpoint exists in this checkout "
        f"(weight-artifact suffix search over {sorted(_WEIGHT_SUFFIXES)} "
        f"found {len(weight_hits)} candidate(s): {weight_hits[:3] or 'none'}) — "
        "the local_fx1 wire path still ran end-to-end against a real "
        "card-only checkpoint dir",
    )

    # --- real stub engine (the closest-to-real in-repo OpenAI-compatible) ----
    from fx1.serve.e2e_audit import _StubChat  # noqa: PLC0415 — audit stub reuse

    stub = ThreadingHTTPServer(("127.0.0.1", 0), _StubChat)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    stub_url = f"http://127.0.0.1:{int(stub.server_address[1])}/v1"

    # --- webhook sink ---------------------------------------------------------
    sink = _SinkHandler()
    sink_srv = ThreadingHTTPServer(("127.0.0.1", 0), _sink_handler(sink))
    threading.Thread(target=sink_srv.serve_forever, daemon=True).start()
    sink_url = f"http://127.0.0.1:{int(sink_srv.server_address[1])}/hook"

    server_env = dict(os.environ)
    for k in ("MOONSHOT_API_KEY", "FX1_SIGNING_KEY", "FX1_LOCAL_SERVE_CMD"):
        server_env.pop(k, None)
    server_env.update(
        {
            "PYTHONPATH": str(_SRC),
            "FX1_API_KEY": _PROBE_KEY,
            "FX1_BYOK_BASE_URL": stub_url,
            "FX1_BYOK_API_KEY": "stub-engine-key",
            "FX1_BYOK_MODEL": "stub-v0",
            "FX1_BYOK_ALLOW_PRIVATE_NETWORKS": "1",
            "FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS": "1",
            "FX1_CHECKPOINT_DIR": str(ckpt_dir),
            "FX1_LOCAL_SERVE_URL": stub_url,
        }
    )

    saved_local = {k: os.environ.get(k) for k in _ENV_KEYS}
    proc: subprocess.Popen[bytes] | None = None
    proc2: subprocess.Popen[bytes] | None = None
    try:
        port = _free_port()
        proc = _spawn_serve(state_dir, port, server_env)
        boot_err = _wait_health(f"http://127.0.0.1:{port}", boot_timeout_s)
        legs.record(
            "serve_subprocess_boots",
            True,
            boot_err == "",
            boot_err or f"fx1 harness serve on 127.0.0.1:{port}, pid={proc.pid}",
        )
        if boot_err:
            raise RuntimeError(f"serve subprocess never came up: {boot_err}")
        base = f"http://127.0.0.1:{port}"

        from fx1.harness import Harness
        from fx1.sdk import Fx1Harness
        from fx1.serve.client import HarnessClient
        from fx1.serve.webhooks import (
            WEBHOOK_SIGNATURE_HEADER,
            WEBHOOK_TIMESTAMP_HEADER,
            verify_webhook,
        )

        client = HarnessClient(base, api_key=_PROBE_KEY, timeout_s=leg_timeout_s)

        # four client surfaces ------------------------------------------------
        doctor_job = _harness_client_legs(legs, base, _PROBE_KEY, leg_timeout_s)
        _openai_legs(legs, base, _PROBE_KEY)
        _anthropic_legs(legs, base, _PROBE_KEY)

        # weights-direct wire path (card-only checkpoint; honest gap above) ---
        try:
            import openai as _openai_mod
        except ImportError:
            _openai_mod = None
        if _openai_mod is None:
            legs.record("local_fx1_link_ran", False, False, "openai package not installed")
        else:
            try:
                oai = _openai_mod.OpenAI(base_url=f"{base}/v1", api_key=_PROBE_KEY, timeout=30.0)
                loc = oai.chat.completions.create(
                    model="local_fx1", messages=[{"role": "user", "content": "local-ping"}]
                )
                loc_done = client.complete(
                    [{"role": "user", "content": "local-hc"}], backend="local_fx1"
                )
                legs.record(
                    "local_fx1_link_ran",
                    True,
                    "local-ping" in (loc.choices[0].message.content or "")
                    and loc_done.backend == "local_fx1",
                    f"oai_model={loc.model} hc_model={loc_done.model} "
                    f"content={(loc.choices[0].message.content or '')[:48]!r}",
                )
            except Exception as exc:  # noqa: BLE001
                legs.record("local_fx1_link_ran", True, False, f"{type(exc).__name__}: {exc}")

        # in-process SDK twin (no socket — parity surface) --------------------
        try:
            os.environ["FX1_BYOK_BASE_URL"] = stub_url
            os.environ["FX1_BYOK_API_KEY"] = "stub-engine-key"
            os.environ["FX1_BYOK_MODEL"] = "stub-v0"
            os.environ["FX1_BYOK_ALLOW_PRIVATE_NETWORKS"] = "1"
            fx = Fx1Harness(harness=Harness())
            ip = fx.complete([{"role": "user", "content": "inproc"}], backend="byok")
            legs.record(
                "inprocess_sdk_parity",
                True,
                ip.content.startswith("stub:inproc"),
                f"backend={ip.backend} model={ip.model}",
            )
        except Exception as exc:  # noqa: BLE001
            legs.record("inprocess_sdk_parity", True, False, f"{type(exc).__name__}: {exc}")

        # eval suite over the wire + terminal webhook -------------------------
        eval_a_id = ""
        eval_b_id = ""
        try:
            ev_a = client.submit_eval(
                "calibration",
                backend="byok",
                seed=0,
                callback_url=sink_url,
                callback_secret=_WEBHOOK_SECRET,
            )
            eval_a_id = str(ev_a["eval_id"])
            fin_a = _wait_terminal(client.eval_status, eval_a_id, leg_timeout_s)
            receipt_a = client.eval_receipt(eval_a_id)
            ok_a = fin_a.get("status") == "succeeded" and verify_receipt_payload(receipt_a)["valid"]
            legs.record(
                "eval_suite_terminal",
                True,
                bool(ok_a),
                f"status={fin_a.get('status')} "
                f"report={{n_questions={(fin_a.get('report') or {}).get('n_questions')},"
                f"n_unparseable={(fin_a.get('report') or {}).get('n_unparseable')}}}",
            )
            (receipts_dir / f"eval_{eval_a_id}.json").write_text(
                json.dumps(receipt_a, indent=2) + "\n"
            )
            ev_b = client.submit_eval("calibration", backend="byok", seed=0)
            eval_b_id = str(ev_b["eval_id"])
            fin_b = _wait_terminal(client.eval_status, eval_b_id, leg_timeout_s)
            receipt_b = client.eval_receipt(eval_b_id)
            (receipts_dir / f"eval_{eval_b_id}.json").write_text(
                json.dumps(receipt_b, indent=2) + "\n"
            )
            diff = client.diff_evals(eval_a_id, eval_b_id)
            legs.record(
                "eval_diff_ran",
                True,
                fin_b.get("status") == "succeeded"
                and verify_receipt_payload(receipt_b)["valid"]
                and diff.get("comparable") is True,
                f"comparable={diff.get('comparable')} "
                f"gate_transition={diff.get('gate_transition')} "
                f"fixed={len(diff.get('tasks_fixed') or [])} "
                f"regressed={len(diff.get('tasks_regressed') or [])}",
            )
        except Exception as exc:  # noqa: BLE001
            legs.record("eval_suite_terminal", True, False, f"{type(exc).__name__}: {exc}")

        # webhook signature over the eval terminal delivery --------------------
        try:
            deadline = time.monotonic() + 30.0
            while not sink.hits and time.monotonic() < deadline:
                time.sleep(0.1)
            hit = sink.hits[-1]
            hdrs = hit[1]
            verified = verify_webhook(
                _WEBHOOK_SECRET,
                hdrs.get(WEBHOOK_TIMESTAMP_HEADER.lower()),
                hdrs.get(WEBHOOK_SIGNATURE_HEADER.lower()),
                hit[2],
            )
            legs.record(
                "webhook_signed_delivery",
                True,
                verified,
                f"hits={len(sink.hits)} sig_ok={verified}",
            )
        except Exception as exc:  # noqa: BLE001
            legs.record("webhook_signed_delivery", True, False, f"{type(exc).__name__}: {exc}")

        # real OpenAI batch over the wire --------------------------------------
        try:
            import openai as _openai_mod
        except ImportError:
            _openai_mod = None
        if _openai_mod is None:
            legs.record("openai_batch_completed", False, False, "openai package not installed")
        else:
            try:
                oai = _openai_mod.OpenAI(base_url=f"{base}/v1", api_key=_PROBE_KEY, timeout=30.0)
                jsonl = b"".join(
                    json.dumps(
                        {
                            "custom_id": cid,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": "byok",
                                "messages": [{"role": "user", "content": f"batch-{i}"}],
                            },
                        }
                    ).encode()
                    + b"\n"
                    for i, cid in enumerate(("g0", "g1"))
                )
                up = oai.files.create(file=("goldenpath-batch.jsonl", jsonl), purpose="batch")
                batch = oai.batches.create(
                    input_file_id=up.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
                final = _wait_terminal(
                    lambda bid: oai.batches.retrieve(bid).model_dump(),
                    batch.id,
                    leg_timeout_s,
                )
                output = ""
                out_id = final.get("output_file_id")
                if out_id:
                    output = oai.files.content(out_id).text
                legs.record(
                    "openai_batch_completed",
                    True,
                    final.get("status") == "completed"
                    and '"custom_id":"g0"' in output
                    and "batch-1" in output,
                    f"status={final.get('status')} output_lines={len(output.splitlines())}",
                )
            except Exception as exc:  # noqa: BLE001
                legs.record("openai_batch_completed", True, False, f"{type(exc).__name__}: {exc}")

        # fine-tune: submit for real, carry to its honest terminal -------------
        ft_job_id = ""
        try:
            ft_jsonl = b"".join(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"q{i}"},
                            {"role": "assistant", "content": f"a{i}"},
                        ]
                    }
                ).encode()
                + b"\n"
                for i in range(3)
            )
            fup = client.upload_file(ft_jsonl, filename="goldenpath-ft.jsonl", purpose="fine-tune")
            ft = client.create_finetune_job(
                model="fx1", training_file=str(fup["id"]), suffix="goldenpath"
            )
            ft_job_id = str(ft["id"])
            legs.record(
                "finetune_submitted",
                True,
                bool(ft_job_id),
                f"job={ft_job_id} status={ft.get('status')}",
            )
        except Exception as exc:  # noqa: BLE001
            legs.record("finetune_submitted", True, False, f"{type(exc).__name__}: {exc}")

        # SIGKILL mid-flight + restart on the same --state-dir -----------------
        try:
            if ft_job_id:
                deadline = time.monotonic() + 20.0
                st: dict[str, Any] = {}
                while time.monotonic() < deadline:
                    st = dict(client.finetune_job(ft_job_id))
                    if st.get("status") in ("running", "failed", "succeeded", "cancelled"):
                        break
                    time.sleep(0.1)
            proc.kill()
            proc.wait(timeout=15)
            port2 = _free_port()
            proc2 = _spawn_serve(state_dir, port2, server_env)
            boot2 = _wait_health(f"http://127.0.0.1:{port2}", boot_timeout_s)
            if boot2:
                raise RuntimeError(f"restart never came up: {boot2}")
            client2 = HarnessClient(
                f"http://127.0.0.1:{port2}", api_key=_PROBE_KEY, timeout_s=leg_timeout_s
            )
            rec_eval = client2.eval_status(eval_a_id) if eval_a_id else {}
            rec_ft = client2.finetune_job(ft_job_id) if ft_job_id else {}
            rec_job = client2.job_status(doctor_job) if doctor_job else {}
            files = client2.files()
            ok = (
                boot2 == ""
                and rec_eval.get("status") == "succeeded"
                and rec_ft.get("status") in ("succeeded", "failed", "cancelled")
                and rec_job.get("status") in ("succeeded", "failed", "cancelled")
                and len(files) >= 2
            )
            legs.record(
                "sigkill_recovery",
                True,
                bool(ok),
                f"eval={rec_eval.get('status')} ft={rec_ft.get('status')} "
                f"job={rec_job.get('status')} files={len(files)}",
            )
            client = client2
            base = f"http://127.0.0.1:{port2}"
        except Exception as exc:  # noqa: BLE001
            legs.record("sigkill_recovery", True, False, f"{type(exc).__name__}: {exc}")

        # the ft job's honest terminal state (post-restart view) ---------------
        if ft_job_id:
            try:
                rec_ft = _wait_terminal(client.finetune_job, ft_job_id, 60.0)
                events = client.finetune_job_events(ft_job_id).get("data", [])
                msgs = [str(e.get("message")) for e in events[:4]] if events else []
                legs.record(
                    "finetune_terminal",
                    True,
                    rec_ft.get("status") in ("succeeded", "failed", "cancelled"),
                    f"status={rec_ft.get('status')} "
                    f"error={_leg_detail(rec_ft.get('error'))} stages={msgs}",
                )
            except Exception as exc:  # noqa: BLE001
                legs.record("finetune_terminal", True, False, f"{type(exc).__name__}: {exc}")

        # every wire receipt re-verifies from disk ------------------------------
        try:
            job_rc = client.job_receipt(doctor_job) if doctor_job else None
            if job_rc is not None:
                (receipts_dir / f"job_{doctor_job}.json").write_text(
                    json.dumps(job_rc, indent=2) + "\n"
                )
            verified = []
            for rc_file in sorted(receipts_dir.glob("*.json")):
                verified.append(verify_receipt_file(rc_file)["valid"])
            legs.record(
                "wire_receipts_verify",
                True,
                bool(verified) and all(verified),
                f"verified={len(verified)} all_valid={all(verified)}",
            )
        except Exception as exc:  # noqa: BLE001
            legs.record("wire_receipts_verify", True, False, f"{type(exc).__name__}: {exc}")
    finally:
        for p in (proc, proc2):
            if p is not None and p.poll() is None:
                p.kill()
        stub.shutdown()
        sink_srv.shutdown()
        for k, v in saved_local.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _sealed() -> dict[str, Any]:
        ran = [k for k in legs.order if legs.entries[k]["ran"]]
        gaps = [k for k in legs.order if not legs.entries[k]["ran"]]
        runnable_ok = all(legs.entries[k]["ok"] for k in ran)
        results = {k: legs.entries[k] for k in legs.order}
        body: dict[str, Any] = {
            "kind": "goldenpath",
            "schema": "goldenpath.v1",
            "git_revision": git_revision(),
            "data_label": "SYNTHETIC",
            "research_only": True,
            "live_pnl_claim": False,
            "claim": {
                "results": results,
                "ok": runnable_ok and not gaps,
                "runnable_ok": runnable_ok,
                "gaps": gaps,
                "legs_ran": len(ran),
                "legs_total": len(legs.order),
            },
            "interpretation": (
                "end-to-end golden path over a real `fx1 harness serve` subprocess: "
                "the stock openai + anthropic SDKs, HarnessClient, and the in-process "
                "Fx1Harness twin all drove the same booted server; a seeded eval ran "
                "to terminal and diffed against a second run; a real /v1/batches job "
                "completed; a fine-tune job reached its honest terminal state; a "
                "signed webhook delivered to loopback; SIGKILL + same-state-dir "
                "restart recovered every journaled record. weights_direct_ran is "
                "false — no fx-1 weight artifact exists in this checkout, so the "
                "card-only local_fx1 link carried the wire path instead."
                if runnable_ok
                else f"GOLDENPATH DEFECT: {results}"
            ),
        }
        body["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
        return body

    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = _sealed()
    out_path.write_text(json.dumps(body, indent=2) + "\n")
    legs.record(
        "goldenpath_receipt_verifies",
        True,
        bool(verify_receipt_file(out_path)["valid"]),
        f"out={out_path}",
    )
    # re-seal so the receipt pins its own verify leg too, then re-verify the
    # final bytes on disk (the printed outcome is outside the seal)
    body = _sealed()
    out_path.write_text(json.dumps(body, indent=2) + "\n")
    body["_final_file_valid"] = bool(verify_receipt_file(out_path)["valid"])
    if owned_dir and not keep:
        import shutil

        with contextlib.suppress(Exception):
            shutil.rmtree(root)
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="sealed receipt path")
    parser.add_argument(
        "--work-dir", type=Path, default=None, help="durable state dir (default: tempdir)"
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the work dir (default tempdir is removed)"
    )
    args = parser.parse_args()
    receipt = run_goldenpath(work_dir=args.work_dir, out=args.out, keep=args.keep)
    claim = receipt["claim"]
    for name, entry in claim["results"].items():
        mark = "ok" if entry["ok"] else ("GAP" if not entry["ran"] else "FAIL")
        print(f"{mark:>4}  {name}  {entry['detail']}", flush=True)
    print(
        f"goldenpath: runnable_ok={claim['runnable_ok']} "
        f"gaps={claim['gaps']} receipt={args.out or 'receipts/fx1_goldenpath.json'}",
        flush=True,
    )
    return 0 if claim["runnable_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
