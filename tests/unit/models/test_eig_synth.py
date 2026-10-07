"""Unit tests for quant_fund.models._eig_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._eig_synth import nonsym_planted, sym_planted, top_eig_err


def test_sym_planted_eigenvalues_match_spectrum() -> None:
    spec = [0.5, 1.0, 2.0, 4.0]
    A, lam = sym_planted(0, n=4, spectrum=spec)
    np.testing.assert_allclose(A, A.T, atol=1e-12)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(A)), np.sort(spec), atol=1e-10)
    np.testing.assert_array_equal(lam, spec)


def test_sym_planted_deterministic() -> None:
    A1, _ = sym_planted(7)
    A2, _ = sym_planted(7)
    np.testing.assert_array_equal(A1, A2)


def test_sym_planted_rejects_mismatched_spectrum() -> None:
    with pytest.raises(ValueError, match="spectrum"):
        sym_planted(0, n=4, spectrum=[1.0, 2.0])
    with pytest.raises(ValueError, match="n"):
        sym_planted(0, n=0)


def test_nonsym_planted_eigenvalues() -> None:
    A, lam = nonsym_planted(3)
    np.testing.assert_allclose(np.sort(np.linalg.eigvals(A).real), np.sort(lam), atol=1e-8)


def test_top_eig_err_perfect_recovery() -> None:
    lam_true = np.array([5.0, 1.0, 0.1])
    assert top_eig_err(lam_true.copy(), lam_true) < 1e-12


def test_top_eig_err_sorts_unsorted_truth() -> None:
    # truth arriving unsorted must still compare correctly: the top-|lam_true|
    # of lam_hat vs lam_true sorted descending.
    lam_true = np.array([0.1, 5.0, 1.0])  # unsorted
    lam_hat = np.array([5.0, 1.0, 0.1])
    assert top_eig_err(lam_hat, lam_true) < 1e-12


def test_top_eig_err_reports_real_error() -> None:
    lam_true = np.array([5.0, 1.0])
    lam_hat = np.array([4.0, 1.0])
    assert abs(top_eig_err(lam_hat, lam_true) - 0.5) < 1e-12
