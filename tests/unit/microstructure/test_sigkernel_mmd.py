"""sigkernel_mmd lane tests — signature-kernel MMD on order flow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.sigkernel_mmd import (
    SCHEMA,
    _sign_path_windows,
    real_sign_stream,
    sigkernel_mmd_bench,
    sigkernel_mmd_contract_errors,
    write_sigkernel_mmd_receipt,
)
from quant_fund.models.sigkernel import (
    mmd2,
    mmd2_permutation,
    sigkernel_gram,
    sigkernel_pde,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload


def _bench(**kw: object) -> dict:
    # production-scale params: regime-vs-iid separation needs enough windows
    # for the permutation null to be sharp (mmd ~8x null std at this size)
    return sigkernel_mmd_bench(horizon=8000, n_windows=40, n_perm=99, **kw)  # type: ignore[arg-type]


def test_kernel_symmetry_and_psd() -> None:
    rng = np.random.default_rng(0)
    a = np.column_stack([np.linspace(0, 1, 16), np.cumsum(rng.normal(size=16) * 0.3)])
    b = np.column_stack([np.linspace(0, 1, 16), np.cumsum(rng.normal(size=16) * 0.3)])
    assert sigkernel_pde(a, b) == pytest.approx(sigkernel_pde(b, a), abs=1e-12)
    assert sigkernel_pde(a, a) >= 1.0
    gram = sigkernel_gram(np.stack([a, b]), np.stack([a, b]))
    assert gram.shape == (2, 2)


def test_pde_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        sigkernel_pde(np.zeros((3, 0)), np.zeros((3, 1)))  # 0 channels
    with pytest.raises(ValueError):
        sigkernel_pde(np.zeros(4), np.zeros((3, 1)))  # 1-D is not a path
    with pytest.raises(ValueError):
        sigkernel_pde(np.zeros((1, 2)), np.zeros((3, 2)))  # <2 points
    with pytest.raises(ValueError):
        sigkernel_pde(np.zeros((4, 2)), np.zeros((4, 1)))  # channel mismatch
    bad = np.full((4, 2), np.nan)
    with pytest.raises(ValueError):
        sigkernel_pde(bad, np.zeros((4, 2)))


def test_mmd_self_zero_and_permutation_shape() -> None:
    rng = np.random.default_rng(1)
    x = np.stack(
        [np.column_stack([np.linspace(0, 1, 12), np.cumsum(rng.normal(size=12))]) for _ in range(6)]
    )
    gxx = sigkernel_gram(x, x)
    assert abs(mmd2(gxx, gxx, gxx)) < 1e-9
    with pytest.raises(ValueError):
        mmd2(gxx, np.zeros((2, 2)), gxx)
    res = mmd2_permutation(x, x.copy(), n_perm=9, seed=0)
    assert set(res) == {"mmd2", "p_value", "null_mean", "null_std", "n_perm"}
    assert res["n_perm"] == 9
    assert 0.0 < res["p_value"] <= 1.0


def test_window_chunker_fail_closed() -> None:
    with pytest.raises(ValueError):
        _sign_path_windows(np.ones(10), window=8, n_windows=4)
    paths = _sign_path_windows(np.ones(200), window=16, n_windows=8)
    assert paths.shape == (8, 16, 2)
    # all +1 window: cumulative channel is the diagonal ramp
    assert np.allclose(paths[0, :, 1], np.arange(1, 17) / 4.0)
    assert np.allclose(paths[0, :, 0], np.linspace(0, 1, 16))


def test_real_stream_requires_tape(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        real_sign_stream(tmp_path)


def test_bench_all_probes_pass_and_shape() -> None:
    payload = _bench()
    assert payload["kind"] == "sigkernel_mmd"
    assert payload["schema"] == SCHEMA
    assert payload["data_label"] == "SYNTHETIC"
    claim = payload["claim"]
    assert claim["ok"] is True
    assert claim["n_passed"] == claim["n_probes"] == len(claim["results"])
    assert claim["tape"] == "absent"
    assert verify_receipt_payload(payload)["valid"]


def test_contract_errors_fail_closed() -> None:
    payload = _bench()
    assert sigkernel_mmd_contract_errors(payload) == []
    assert "schema_mismatch" in sigkernel_mmd_contract_errors({**payload, "schema": "x"})
    bad = json.loads(json.dumps(payload))
    bad["claim"]["n_passed"] = 0
    assert "n_passed_mismatch" in sigkernel_mmd_contract_errors(bad)
    bad2 = json.loads(json.dumps(payload))
    bad2["claim"]["ok"] = False
    assert "ok_mismatch" in sigkernel_mmd_contract_errors(bad2)
    assert "claim_not_mapping" in sigkernel_mmd_contract_errors({**payload, "claim": 1})
    assert "live_pnl_claim_not_false" in sigkernel_mmd_contract_errors(
        {**payload, "live_pnl_claim": True}
    )


def test_write_receipt_seals_and_verifies(tmp_path: Path) -> None:
    payload = _bench()
    out = write_sigkernel_mmd_receipt(payload, receipts_dir=tmp_path)
    sealed = json.loads(out.read_text())
    assert sealed["receipt_sha256"]
    assert verify_receipt_payload(sealed)["valid"]
    with pytest.raises(ValueError):
        write_sigkernel_mmd_receipt({**payload, "kind": "other"}, receipts_dir=tmp_path)


def test_committed_receipt_dispatches_contract() -> None:
    path = Path("receipts/sigkernel_mmd.json")
    if not path.exists():
        pytest.skip("receipt not generated yet")
    payload = json.loads(path.read_text())
    assert verify_receipt_payload(payload)["valid"]
    assert sigkernel_mmd_contract_errors(payload) == []


def test_lane_dispatch_routes_sigkernel() -> None:
    from quant_fund.research.lane_contracts import lane_contract_errors

    payload = _bench()
    assert lane_contract_errors(payload) == []
    bad = {**payload, "claim": {"results": {"a": True}, "n_probes": 9, "n_passed": 1, "ok": False}}
    errs = lane_contract_errors(bad)
    assert errs and "n_probes_mismatch" in errs
