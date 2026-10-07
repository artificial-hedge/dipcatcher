"""Adversarial probes for _tn_synth (SYNTHETIC)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._tn_synth import mps_contract, smooth_tensor, tfim_h, to_mps


def test_smooth_tensor_deterministic_and_smooth():
    t1 = smooth_tensor(0)
    t2 = smooth_tensor(0)
    np.testing.assert_array_equal(t1, t2)
    assert t1.shape == (8, 8, 8, 8)
    assert (t1 > 0).all()
    assert np.isfinite(t1).all()


@pytest.mark.parametrize("d,n", [(0, 8), (4, 1), (4, 0)])
def test_smooth_tensor_hostile(d, n):
    with pytest.raises(ValueError):
        smooth_tensor(0, d=d, n=n)


def test_tfim_h_hermitian_and_shape():
    h2 = tfim_h(2)
    assert h2.shape == (4, 4)
    np.testing.assert_allclose(h2, h2.T)
    # n=2, h=1: ham = -ZZ - X⊗I - I⊗X → ground energy = -sqrt( (2)^2 + ... )
    evals = np.linalg.eigvalsh(h2)
    assert evals[0] < 0  # nontrivial spectrum


def test_tfim_h_field_freezes_at_h0():
    h_nofield = tfim_h(2, h=0.0)
    h_field = tfim_h(2, h=2.0)
    assert not np.allclose(h_nofield, h_field)


@pytest.mark.parametrize("n,h", [(0, 1.0), (2, np.nan), (2, np.inf)])
def test_tfim_h_hostile(n, h):
    with pytest.raises(ValueError):
        tfim_h(n, h=h)


def test_to_mps_roundtrip():
    T = smooth_tensor(0, d=3, n=4)  # 4^3 = 64 = 2^6? no — use d_phys=4
    cores = to_mps(T, d_phys=4, chi=8)
    rec = mps_contract(cores)
    np.testing.assert_allclose(rec, T.reshape(-1), atol=1e-10)


def test_to_mps_bad_size_raises():
    T = np.zeros((3, 3, 3))  # 27 not a power of 2
    with pytest.raises(ValueError):
        to_mps(T, d_phys=2)


@pytest.mark.parametrize("d_phys,chi", [(1, 4), (2, 0)])
def test_to_mps_hostile_params(d_phys, chi):
    T = smooth_tensor(0, d=2, n=2)  # 16 elements
    with pytest.raises(ValueError):
        to_mps(T, d_phys=d_phys, chi=chi)


def test_mps_contract_empty():
    with pytest.raises(ValueError):
        mps_contract([])
