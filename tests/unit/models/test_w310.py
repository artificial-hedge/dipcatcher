"""Wave-310 quantum-error-correction module unit tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.gottesman_knill import (
    cnot,
    h,
    init_state,
    measure,
    stabilizer_vec,
)
from quant_fund.models.repetition_qec import analytic_failure, decode, encode
from quant_fund.models.shor_code import apply_pauli, decode_state, encode_zero
from quant_fund.models.steane_code import H as STEANE_H
from quant_fund.models.steane_code import simplex_codewords
from quant_fund.models.surface_code import (
    X_STABS,
    Z_STABS,
    _intvec,
    _stab_row_space,
    _vecint,
    build_decode_table,
    logical_failure,
)
from quant_fund.models.syndrome_circuit import bell_state


def test_gk_bell_stabilizers() -> None:
    tab = init_state(2)
    h(tab, 0, 2)
    cnot(tab, 0, 1, 2)
    assert {stabilizer_vec(tab, k, 2) for k in range(2)} == {"XX", "ZZ"}
    rng = np.random.default_rng(0)
    a = measure(tab, 0, 2, rng)
    b = measure(tab, 1, 2, rng)
    assert a == b


def test_steane_simplex_in_hamming_kernel() -> None:
    for c in simplex_codewords():
        bits = np.array([(c >> (6 - q)) & 1 for q in range(7)], dtype=np.uint8)
        assert not (STEANE_H @ bits % 2).any()
    # simplex dual C subset = Hamming kernel: all rows are even weight
    assert all(bin(c).count("1") % 2 == 0 for c in simplex_codewords())


def test_surface_code_properties() -> None:
    assert not (X_STABS @ Z_STABS.T % 2).any()
    xrow = _stab_row_space(X_STABS)
    ker_z = [v for v in range(512) if not ((Z_STABS @ _intvec(v)) % 2).any()]
    dz = min(bin(v).count("1") for v in ker_z if v not in xrow)
    assert dz == 3
    table = build_decode_table(Z_STABS)
    for q in range(9):
        syn = _vecint((Z_STABS @ _intvec(1 << q)) % 2)
        assert not logical_failure(1 << q, table[syn], Z_STABS, X_STABS)


def test_shor_corrects_x_error() -> None:
    psi0 = encode_zero()
    err = apply_pauli(psi0, 4, "X")
    rec = decode_state(err)
    rec = rec / np.linalg.norm(rec)
    assert abs(np.vdot(psi0, rec)) ** 2 > 0.999


def test_syndrome_bell_clean_support() -> None:
    psi = bell_state()
    assert abs(psi[0]) ** 2 > 0.49 and abs(psi[0b1100]) ** 2 > 0.49


def test_repetition_analytic() -> None:
    assert decode(np.array([1, 0, 0], dtype=np.uint8)) == 0
    assert np.array_equal(encode(1, 3), np.ones(3, dtype=np.uint8))
    assert abs(analytic_failure(3, 0.1) - (3 * 0.1**2 * 0.9 + 0.1**3)) < 1e-9
