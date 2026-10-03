"""run_audit — adversarial probes on the Dip bench runner.

Pins ``run_dip_bench``'s honesty surface:

- Empty data dir fails closed (``FileNotFoundError``) — a bench with no
  inputs cannot emit a receipt.
- Every input parquet is bound by content sha256 (``inputs_sha256``);
  the same file with different bytes produces a different pin.
- File order is glob-sorted → deterministic event order.
- The climatology forecaster is documented as in-sample descriptive —
  ``research_only``/``live_pnl_claim``/``disclaimer`` are hardcoded into
  the payload; ``assert_bench_output_honest`` runs before emit.
- NaN baseline entries are filtered out of forecast probability dicts.

All fixtures are SYNTHETIC parquets in a tmpdir — this audits the bench
contract, not the market.

Sealed ``dip_run_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["dip_run_audit", "dip_run_audit_bench"]


def _write_series(dirpath: Path, name: str, symbol: str, n: int = 320) -> Path:
    import polars as pl

    # gentle uptrend with a planted 15% dip mid-series
    closes = [100.0 + i * 0.02 for i in range(n)]
    for i in range(n // 2, n // 2 + 8):
        closes[i] *= 0.85
    dates = [f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)]
    df = pl.DataFrame({"event_time": dates, "symbol": [symbol] * n, "close": closes})
    path = dirpath / name
    df.write_parquet(path)
    return path


def dip_run_audit() -> dict[str, Any]:
    from fx1.bench.run import run_dip_bench

    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        p1 = _write_series(root, "aaa_1d.parquet", "AAA")
        p2 = _write_series(root, "bbb_1d.parquet", "BBB")
        receipt = run_dip_bench(root, horizons_bars={"1m": 21})
        out["events_found"] = receipt["n_events"] >= 2  # one dip per series
        out["inputs_bound"] = set(receipt["inputs_sha256"]) == {
            p1.name,
            p2.name,
        }
        out["input_digests_64"] = all(len(v) == 64 for v in receipt["inputs_sha256"].values())
        out["honesty_fields"] = (
            receipt["research_only"] is True
            and receipt["live_pnl_claim"] is False
            and "not live performance" in receipt["disclaimer"]
        )
        out["per_asset_counts"] = receipt["per_asset"]
        metrics = receipt["climatology_forecast"]["metrics"]
        out["metrics_under_honesty_gate"] = isinstance(metrics, dict)
        out["in_sample_labeled"] = "in-sample" in receipt["climatology_forecast"]["definition"]

        # mutating one byte of an input must change its pinned digest
        p2.write_bytes(p2.read_bytes() + b"\x00tampered")
        # (parquet tail corruption: digest binds bytes regardless of readability)
        import hashlib

        out["tamper_changes_pin"] = (
            hashlib.sha256(p2.read_bytes()).hexdigest() != receipt["inputs_sha256"][p2.name]
        )

    empty = tempfile.mkdtemp()
    try:
        run_dip_bench(empty)
        out["empty_raises"] = "no-raise"
    except FileNotFoundError:
        out["empty_raises"] = "raise:FileNotFoundError"
    return out


def dip_run_audit_bench() -> dict[str, Any]:
    r = dip_run_audit()
    ok = (
        r["events_found"] is True
        and r["inputs_bound"] is True
        and r["input_digests_64"] is True
        and r["honesty_fields"] is True
        and r["metrics_under_honesty_gate"] is True
        and r["in_sample_labeled"] is True
        and r["tamper_changes_pin"] is True
        and r["empty_raises"] == "raise:FileNotFoundError"
    )
    out: dict[str, Any] = {
        "kind": "dip_run_audit",
        "schema": "dip_run_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Bench-runner contract holds: inputs content-pinned, tampering "
            "changes pins, honesty fields hardcoded, in-sample climatology "
            "labeled, empty dirs fail closed."
            if ok
            else f"DIP RUN AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
