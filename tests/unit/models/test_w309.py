"""Wave-309 post-quantum-3 module unit tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.bike_lite import bitflip_decode, poly_inv, poly_mul
from quant_fund.models.hqc_lite import concat_decode, concat_encode
from quant_fund.models.mceliece_lite import _build_gh, _mm
from quant_fund.models.sidh_lite import (
    _chain_with_push,
    _find_basis,
    j_invariant,
    order,
)
from quant_fund.models.uov_sig import (
    _compose_public,
    _gf2_inv,
    _inverse,
    _rand_central,
    _terms,
    public_eval,
)


def test_mceliece_g_h_orthogonal() -> None:
    g, h = _build_gh()
    assert np.array_equal(_mm(g, h.T) % 2, np.zeros((11, 4), dtype=np.uint8))
    cols = {tuple(int(b) for b in h[:, j]) for j in range(h.shape[1])}
    expect = {
        tuple(int(b) for b in np.array([(v >> k) & 1 for k in range(4)], dtype=np.uint8))
        for v in range(1, 16)
    }
    assert cols == expect


def test_bike_poly_inv_roundtrip() -> None:
    r = 61
    rng = np.random.default_rng(0)
    a = np.zeros(r, dtype=np.uint8)
    a[rng.choice(r, 7, replace=False)] = 1
    ai = poly_inv(a, r)
    assert ai is not None
    prod = poly_mul(a, ai, r)
    assert prod[0] == 1 and prod[1:].sum() == 0


def test_bike_bitflip_recovers() -> None:
    rng = np.random.default_rng(1)
    r = 61
    h0 = np.zeros(r, dtype=np.uint8)
    h1 = np.zeros(r, dtype=np.uint8)
    h0[rng.choice(r, 7, replace=False)] = 1
    h1[rng.choice(r, 7, replace=False)] = 1
    e = np.zeros(2 * r, dtype=np.uint8)
    e[rng.choice(2 * r, 3, replace=False)] = 1
    syn = poly_mul(e[:r], h0, r) ^ poly_mul(e[r:], h1, r)
    dec = bitflip_decode(syn, h0, h1, r)
    assert dec is not None
    assert np.array_equal(np.concatenate(dec), e)


def test_hqc_concat_roundtrip() -> None:
    rng = np.random.default_rng(2)
    m = rng.integers(0, 2, 4).astype(np.uint8)
    code = concat_encode(m, 5)
    assert len(code) == 40
    err = code.copy()
    err[rng.choice(40, 4, replace=False)] ^= 1
    assert np.array_equal(concat_decode(err, 5), m)


def test_uov_public_matches_private() -> None:
    rng = np.random.default_rng(3)
    v, o = 4, 3
    n = v + o
    F = _rand_central(rng, v, o)
    T = _gf2_inv(rng, n)
    assert np.array_equal(T @ _inverse(T) % 2, np.eye(n, dtype=np.uint8))
    P = _compose_public(F, T, n)
    x = rng.integers(0, 2, n).astype(np.uint8)
    t = _terms(n)
    # F(Ti x) == P(x)
    z = T @ x % 2
    lhs = (
        np.array(
            [
                sum(int(F[k, idx]) * (int(z[i]) & int(z[j])) for idx, (i, j) in enumerate(t))
                + int(F[k, -1])
                for k in range(o)
            ],
            dtype=np.uint8,
        )
        % 2
    )
    assert np.array_equal(public_eval(P, x, n), lhs)


def test_sidh_shared_j() -> None:
    pa = _find_basis(8, 54)
    pb = _find_basis(27, 16)
    assert order(pa) == 8 and order(pb) == 27
    ea_a, eb_a, phi_a_pb = _chain_with_push(pa, 2, 3, pb)
    ea_b, eb_b, phi_b_pa = _chain_with_push(pb, 3, 3, pa)
    ea_ab, eb_ab, _ = _chain_with_push(phi_b_pa, 2, 3, phi_b_pa, ea_b, eb_b)
    ea_ba, eb_ba, _ = _chain_with_push(phi_a_pb, 3, 3, phi_a_pb, ea_a, eb_a)
    assert j_invariant(ea_ab, eb_ab) == j_invariant(ea_ba, eb_ba)
