# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # fx1 quickstart — HTTP server + managed keys + HarnessClient
#
# Data label: SYNTHETIC (stub backend — no weights, no off-box network)
#
# Not investment advice. No live-trading claim.
#
# The production FastAPI app booted on loopback the same way
# `fx1 harness serve` does it, then driven end-to-end with
# `HarnessClient`: bootstrap credential → mint a scoped managed key →
# gated completion + seeded eval → sealed receipts verified → key usage
# → revoke → drain. `examples/fx1_quickstart_sdk.py` is the in-process
# twin of this flow.

# %%
from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import uvicorn

from fx1.serve.api import create_app
from fx1.serve.client import HarnessAuthError, HarnessClient

# Probe credential standing in for the real bootstrap secret. A real
# deployment injects FX1_API_KEY from its secret store and never writes
# it into source — this value exists only so auth resolution has
# something to compare against on loopback.
BOOTSTRAP_API_KEY = "changeme-quickstart-bootstrap-credential"


class _StubBackend:
    """Echo backend standing in for a configured model lane.

    Same contract `api_audit.py`'s `_MeterBackend` implements: the app
    calls `complete(messages, sampling=...)`, reads `_model`/`last_usage`
    for metering, and calls `close()` when it is done with a resolve.
    """

    def __init__(self) -> None:
        self._model = "stub-v0"
        self.calls = 0
        self.last_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.closed = 0

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del sampling  # the stub ignores decode params; a real backend honors them
        self.calls += 1
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 8,
            "total_tokens": prompt_tokens + 8,
        }
        return f"stub:{messages[-1]['content']}"

    def close(self) -> None:
        self.closed += 1


def _serve(app: Any) -> tuple[uvicorn.Server, threading.Thread, int]:
    """Run a FastAPI app on a free loopback port — the same uvicorn
    invocation `fx1 harness serve` makes, minus the CLI flags."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    return server, thread, port


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fx1-quickstart-http-") as tmp:
        receipts_dir = Path(tmp) / "receipts"
        receipts_dir.mkdir()
        stub = _StubBackend()
        # The bootstrap credential turns auth on; without it the app
        # still serves but refuses to bind non-loopback in production.
        os.environ["FX1_API_KEY"] = BOOTSTRAP_API_KEY
        # backend_resolver answers for whatever lane name is requested —
        # here 'byok', since the stub emulates an OpenAI-compatible
        # user-supplied endpoint.
        app = create_app(
            backend_resolver=lambda *a, **k: stub,
            receipts_dir=receipts_dir,
        )
        server, thread, port = _serve(app)
        base_url = f"http://127.0.0.1:{port}"
        try:
            root = HarnessClient(base_url, api_key=BOOTSTRAP_API_KEY)
            print(f"server_status={root.health().status}")
            print(f"api_version={root.server_version()['api_version']}")

            # --- bootstrap: mint a scoped managed key ----------------------
            mint = root.key_create(
                "quickstart-workload",
                scopes=["read", "write"],
                max_requests=100,
            )
            print(f"key_id={mint['id']}")
            print(f"key_scopes={','.join(mint['scopes'])}")
            client = HarnessClient(base_url, api_key=mint["key"])

            # --- one gated chat completion ---------------------------------
            messages = [{"role": "user", "content": "summarize the harness in one word"}]
            completion = client.complete(messages, backend="byok")
            print(f"completion_id={completion.completion_id}")
            print(f"completion_content={completion.content!r}")

            # --- one seeded eval ---------------------------------------------
            submitted = client.submit_eval("tooluse", backend="byok")
            record = client.wait_eval(submitted["eval_id"], timeout_s=120.0)
            report = record.get("report") or {}
            print(f"eval_id={submitted['eval_id']}")
            print(f"eval_status={record['status']}")
            print(f"eval_tasks={report.get('n_tasks')}")
            print(f"eval_pass_rate={report.get('pass_rate')}")

            # --- seal + verify both ops receipts -----------------------------
            completion_doc = client.completion_receipt(completion.completion_id)
            eval_doc = client.eval_receipt(submitted["eval_id"])
            for name, doc in (("completion", completion_doc), ("eval", eval_doc)):
                path = receipts_dir / f"fx1_{name}_record.json"
                path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                verdict = client.verify_receipt(doc)
                if not verdict.valid:
                    raise SystemExit(f"{name} receipt failed verify: {verdict.errors}")
                print(f"{name}_receipt_schema={doc['schema']}")
                print(f"{name}_receipt_path={path}")
                print(f"{name}_receipt_valid={str(verdict.valid).lower()}")

            # --- usage accounting on the managed key -------------------------
            # /harness/self is read-scope — the key watches its own card;
            # /harness/keys/{id}/usage is admin, so the bootstrap client
            # asks for the operator view.
            self_card = client.self_usage()
            print(f"self_credential={self_card['credential']}")
            usage = root.key_usage(mint["id"])
            print(f"key_uses={usage['uses']}")
            served = usage.get("served", {})
            print(f"key_tokens={served.get('total_tokens')}")

            # --- revoke: the tombstone fails closed immediately --------------
            root.key_revoke(mint["id"])
            try:
                client.complete(messages, backend="byok")
            except HarnessAuthError:
                print("revoked_key_rejected=true")
            else:
                raise SystemExit("revoked key still authenticated")

            # --- drain before shutdown ----------------------------------------
            drain = root.drain(wait_s=0.0)
            print(f"draining={str(drain['draining']).lower()}")
        finally:
            server.should_exit = True
            thread.join(timeout=10)

        # --- honesty contract ---------------------------------------------
        for doc in (completion_doc, eval_doc):
            if doc.get("data_label") != "OPS" or doc.get("research_only") is not True:
                raise SystemExit("ops receipt lost its research_only/OPS label")
            if doc.get("live_pnl_claim") is not False:
                raise SystemExit("ops receipt claims live P&L — impossible")
        print("data_label=OPS")
        print(f"stub_backend_calls={stub.calls}")
        print("not_investment_advice=true")
        print("no_live_trading_claim=true")
        print("claim=research_only")


if __name__ == "__main__":
    main()
