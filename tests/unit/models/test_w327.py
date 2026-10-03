"""Wave-327 zero-knowledge module unit tests."""

from __future__ import annotations

from quant_fund.models.bulletproof_ip import verify as ip_verify
from quant_fund.models.kzg_commit import commit, open_at, srs, verify
from quant_fund.models.plonkish_gate import check_circuit, copy_consistent
from quant_fund.models.qap_encode import h_exist, qap_check
from quant_fund.models.r1cs_check import mul_gadget, satisfies
from quant_fund.models.snark_circuit import prove


def test_r1cs() -> None:
    a, b, c, w = mul_gadget(3, 4, 12)
    assert satisfies(a, b, c, w)
    w[3] = 13
    assert not satisfies(a, b, c, w)


def test_qap() -> None:
    a = [[0, 1, 0, 0]]
    b = [[0, 0, 1, 0]]
    c = [[0, 0, 0, 1]]
    assert qap_check(a, b, c, [1, 3, 4, 12])
    assert h_exist(a, b, c, [1, 3, 4, 12])


def test_kzg() -> None:
    powers = srs(42, 4)
    p = [3, 5, 7]
    c = commit(p, powers)
    v, pi = open_at(p, 4, powers)
    assert verify(c, 4, v, pi, 42)


def test_plonkish() -> None:
    assert check_circuit([(1, 1, 0, -1, 0)], [(3, 4, 7)])
    assert copy_consistent([7, 7, 3], [1, 0, 2])


def test_snark() -> None:
    pf = prove(("add", ("mul", "x", "y"), "x"), {"x": 3, "y": 4})
    assert pf["ok"] and pf["w"][-1] == 15


def test_bulletproof_api() -> None:
    assert ip_verify.__name__ == "verify"
