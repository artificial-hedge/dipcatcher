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
# # fx1 quickstart — in-process SDK
#
# Data label: SYNTHETIC (stub backend — no weights, no network)
#
# Not investment advice. No live-trading claim.
#
# The smallest real `Fx1Harness` loop a deployment runs — one gated chat
# completion and one seeded eval — against a stub backend that stands in
# for the BYOK lane (the same resolver-injection pattern
# `src/fx1/serve/api_audit.py` uses to drive the app without weights).
# Both calls seal as ops receipts on disk and re-verify.

# %%
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from fx1.sdk import Fx1Harness


class _StubBackend:
    """Echo backend standing in for a configured model lane.

    Same contract `api_audit.py`'s `_MeterBackend` implements: the harness
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


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fx1-quickstart-sdk-") as tmp:
        receipts_dir = Path(tmp) / "receipts"
        receipts_dir.mkdir()
        stub = _StubBackend()
        # backend_resolver answers for whatever lane name is requested —
        # here 'byok', since the stub emulates an OpenAI-compatible
        # user-supplied endpoint. A real deployment leaves the resolver
        # unset so get_backend() reads FX1_BYOK_* / checkpoint config.
        fx = Fx1Harness(
            backend_resolver=lambda *a, **k: stub,
            receipts_dir=receipts_dir,
        )

        # --- one gated chat completion ------------------------------------
        messages = [{"role": "user", "content": "summarize the harness in one word"}]
        completion = fx.complete(messages, backend="byok")
        print(f"completion_id={completion.completion_id}")
        print(f"completion_content={completion.content!r}")

        # --- one seeded eval ----------------------------------------------
        # 'tooluse' is the 12-task seeded bank the audit suite uses —
        # deterministic, no judge, no crash on bad model output.
        record = fx.run_eval("tooluse", backend="byok", seed=0)
        report = record.report or {}
        print(f"eval_id={record.eval_id}")
        print(f"eval_status={record.status}")
        print(f"eval_tasks={report.get('n_tasks')}")
        print(f"eval_pass_rate={report.get('pass_rate')}")

        # --- seal + verify both ops receipts ------------------------------
        completion_doc = fx.completion_receipt(completion.completion_id)
        eval_doc = fx.eval_receipt(record.eval_id)
        for name, doc in (("completion", completion_doc), ("eval", eval_doc)):
            path = receipts_dir / f"fx1_{name}_record.json"
            path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            verdict = fx.verify_receipt(doc)
            if not verdict.valid:
                raise SystemExit(f"{name} receipt failed verify: {verdict.errors}")
            print(f"{name}_receipt_schema={doc['schema']}")
            print(f"{name}_receipt_path={path}")
            print(f"{name}_receipt_valid={str(verdict.valid).lower()}")

        # The receipts dir now indexes both sealed docs by content hash —
        # the same store GET /receipts* serves on the wire.
        for ref in fx.receipts():
            print(f"stored_receipt_sha256={ref.sha256[:16]}... name={ref.name}")

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
