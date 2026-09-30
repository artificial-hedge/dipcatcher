"""Tests for quant_fund.models.path_signatures — signatures / rough-path features.

References: Chen (1958, Trans. AMS 89:395–407); Lyons (1998, Rev. Mat.
Iberoam. 14:215–310); Chevyrev & Kormilitzin (2016, arXiv:1603.03788);
Kidger & Foster (2020, arXiv:2005.08328); Hall (1950); Reutenauer (1993);
Witt (1937).
"""

import math

import numpy as np
import pytest

from quant_fund.models.path_signatures import (
    _chen_product,
    _lie_series_from_coordinates,
    _lyndon_coordinates,
    _signature_tensors,
    _tensor_exp,
    _tensor_log,
    _witt_dimension,
    lead_lag_transform,
    logsignature,
    signature,
    signature_kernel,
)

Array = np.ndarray


def _random_path(rng: np.random.Generator, n_inc: int, dims: int) -> Array:
    increments = rng.normal(size=(n_inc, dims))
    return np.concatenate([np.zeros((1, dims)), np.cumsum(increments, axis=0)])


def _level_slices(dims: int, order: int) -> list[slice]:
    """Word-order flat-layout slices per level (offsets d + d^2 + ...)."""
    out = []
    offset = 0
    for k in range(1, order + 1):
        width = dims**k
        out.append(slice(offset, offset + width))
        offset += width
    return out


def _levy_area_term(lead_lag_sig2: Array) -> float:
    """(1/2) (Sig^2_{lead_1, lag_2} - Sig^2_{lead_2, lag_1}) for a 2-D lead-lag path."""
    dims = lead_lag_sig2.shape[0] // 2
    return 0.5 * float(lead_lag_sig2[0, dims + 1] - lead_lag_sig2[1, dims + 0])


def test_straight_line_closed_form() -> None:
    """Level-k term of a straight line is exactly delta^{⊗ k} / k! (Chen 1958)."""
    rng = np.random.default_rng(101)
    for dims in (1, 2, 3):
        delta = rng.normal(size=dims)
        path = np.stack([np.zeros(dims), delta])
        order = 4
        sig = signature(path, order)
        for k, sl in enumerate(_level_slices(dims, order), start=1):
            expected = delta
            for _ in range(k - 1):
                expected = np.kron(expected, delta)
            expected = expected / math.factorial(k)
            np.testing.assert_allclose(sig[sl], expected, rtol=1e-14, atol=1e-15)


def test_chen_identity_concatenation() -> None:
    """Sig(path1 * path2) = Sig(path1) ⊗ Sig(path2) truncated (Chen 1958, Thm 5.1)."""
    rng = np.random.default_rng(202)
    order = 4
    for dims in (1, 2, 3):
        path1 = _random_path(rng, 7, dims)
        path2 = _random_path(rng, 5, dims)
        joint = np.concatenate([path1, path1[-1:] + (path2 - path2[0])])
        left = _signature_tensors(joint, order)
        right = _chen_product(
            _signature_tensors(path1, order), _signature_tensors(path2, order), order
        )
        for k in range(order):
            np.testing.assert_allclose(left[k], right[k], rtol=1e-12, atol=1e-13)


def test_translation_invariance() -> None:
    """Adding a constant to every point leaves the signature unchanged."""
    rng = np.random.default_rng(303)
    for dims in (1, 2, 3):
        path = _random_path(rng, 11, dims)
        shifted = path + rng.normal(size=dims)
        np.testing.assert_allclose(
            signature(path, 4), signature(shifted, 4), rtol=1e-12, atol=1e-13
        )


def test_reparametrization_invariance() -> None:
    """Time-warped sampling of the same smooth curve gives the same signature.

    The signature of a piecewise-linear interpolation depends only on the
    sequence of points (Lyons 1998; Chevyrev & Kormilitzin 2016, §2.2), so a
    smooth monotone time change collapses to interpolation error, which is
    O(1/N^2) here.
    """
    rng = np.random.default_rng(404)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=3)

    def curve(u: Array) -> Array:
        return np.stack(
            [
                np.cos(2.0 * np.pi * u) + 0.25 * np.sin(np.pi * u + phase[0]),
                np.sin(2.0 * np.pi * u) * (0.5 + 0.25 * u) + phase[1] * 0.1 * u,
                0.3 * u**2 + phase[2] * 0.05,
            ],
            axis=1,
        )

    n_grid = 20000
    u = np.linspace(0.0, 1.0, n_grid + 1)
    phi = u + 0.2 * np.sin(np.pi * u)  # smooth, monotone on [0, 1]
    phi = (phi - phi[0]) / (phi[-1] - phi[0])
    sig_uniform = signature(curve(u), 4)
    sig_warped = signature(curve(phi), 4)
    np.testing.assert_allclose(sig_warped, sig_uniform, rtol=1e-6, atol=1e-6)


def test_lead_lag_layout() -> None:
    """Lead-lag path shape and interleaving (Bonnier et al. 2019, §2)."""
    path = np.array([[0.0, 5.0], [1.0, 7.0], [3.0, 8.0]])
    ll = lead_lag_transform(path)
    assert ll.shape == (5, 4)
    np.testing.assert_allclose(
        ll,
        np.array(
            [
                [0.0, 5.0, 0.0, 5.0],
                [1.0, 7.0, 0.0, 5.0],
                [1.0, 7.0, 1.0, 7.0],
                [3.0, 8.0, 1.0, 7.0],
                [3.0, 8.0, 3.0, 8.0],
            ]
        ),
    )


def test_lead_lag_levy_area_circle() -> None:
    """Closed circle: antisymmetric level-2 term = inscribed-polygon signed area."""
    n_points = 96
    angle = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    radius = 1.7
    circle = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    circle = np.concatenate([circle, circle[:1]])
    sig2 = signature(lead_lag_transform(circle), 2)[4:].reshape(4, 4)
    # shoelace closed form for the inscribed polygon: (N/2) r^2 sin(2π/N)
    expected = 0.5 * n_points * radius**2 * math.sin(2.0 * math.pi / n_points)
    assert _levy_area_term(sig2) == pytest.approx(expected, rel=1e-12)


def test_lead_lag_levy_area_triangle() -> None:
    """Closed triangle: antisymmetric level-2 term matches shoelace area."""
    height = 2.5
    triangle = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [1.5, height],
            [0.0, 0.0],
        ]
    )
    sig2 = signature(lead_lag_transform(triangle), 2)[4:].reshape(4, 4)
    expected = 0.5 * 3.0 * height  # shoelace
    assert _levy_area_term(sig2) == pytest.approx(expected, rel=1e-12)
    # reversed orientation flips the sign
    sig2_rev = signature(lead_lag_transform(triangle[::-1]), 2)[4:].reshape(4, 4)
    assert _levy_area_term(sig2_rev) == pytest.approx(-expected, rel=1e-12)


def test_logsignature_dimension_and_basis() -> None:
    """Length equals the Witt dimension; L-shaped path has a known closed form."""
    dims, order = 2, 4
    expected_dim = _witt_dimension(dims, order)
    assert expected_dim == 8  # l_1..l_4 = 2 + 1 + 2 + 3
    rng = np.random.default_rng(505)
    coords = logsignature(_random_path(rng, 9, dims), order)
    assert coords.shape == (expected_dim,)
    # hand-computed: increments (1,0) then (0,1); log = e_1 + e_2 + (1/2)[e_1, e_2]
    ell = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(logsignature(ell, 2), np.array([1.0, 1.0, 0.5]), atol=1e-15)


def test_logsignature_group_like_roundtrip() -> None:
    """exp(logsignature) reconstructs the truncated signature (Magnus 1954)."""
    rng = np.random.default_rng(606)
    for dims in (2, 3):
        for order in (2, 3, 4):
            path = _random_path(rng, 13, dims)
            tensors = _signature_tensors(path, order)
            coords = _lyndon_coordinates(_tensor_log(tensors, order), order)
            recovered = _tensor_exp(_lie_series_from_coordinates(coords, dims, order), order)
            for k in range(order):
                np.testing.assert_allclose(recovered[k], tensors[k], rtol=1e-10, atol=1e-12)


def test_logsignature_of_line_is_level_one() -> None:
    """The logsignature of a straight line is exactly its displacement delta."""
    rng = np.random.default_rng(707)
    for dims in (1, 2, 3):
        delta = rng.normal(size=dims)
        line = np.stack([np.zeros(dims), delta])
        coords = logsignature(line, 4)
        np.testing.assert_allclose(coords[:dims], delta, atol=1e-15)
        assert np.all(np.abs(coords[dims:]) < 1e-12)


def test_signature_kernel_symmetry_and_positivity() -> None:
    """k(a,b) = k(b,a), k(x,x) > 0, and Gram matrices are PSD."""
    rng = np.random.default_rng(808)
    order = 3
    paths = [_random_path(rng, 8 + i, 2) for i in range(6)]
    for i in range(len(paths)):
        for j in range(len(paths)):
            assert signature_kernel(paths[i], paths[j], order) == pytest.approx(
                signature_kernel(paths[j], paths[i], order), rel=1e-14
            )
    for path in paths:
        assert signature_kernel(path, path, order) > 0.0
    gram = np.array([[signature_kernel(p, q, order) for q in paths] for p in paths])
    eigvals = np.linalg.eigvalsh(gram)
    assert float(eigvals.min()) >= -1e-10


def test_signature_kernel_matches_inner_product() -> None:
    """k_m(a, b; σ) = 1 + Σ_k σ^{2k} ⟨Sig^k(a), Sig^k(b)⟩ by hand."""
    rng = np.random.default_rng(909)
    order, sigma = 3, 0.7
    path_a = _random_path(rng, 9, 2)
    path_b = _random_path(rng, 6, 2)
    expected = 1.0
    for k, (ta, tb) in enumerate(
        zip(_signature_tensors(path_a, order), _signature_tensors(path_b, order), strict=True),
        start=1,
    ):
        expected += sigma ** (2 * k) * float(np.sum(ta * tb))
    assert signature_kernel(path_a, path_b, order, sigma=sigma) == pytest.approx(
        expected, rel=1e-14
    )
    # sigma rescales increments: k(a, b; σ) == k(σa, σb; 1)
    assert signature_kernel(path_a, path_b, order, sigma=sigma) == pytest.approx(
        signature_kernel(sigma * path_a, sigma * path_b, order), rel=1e-14
    )


def test_fail_closed_edges() -> None:
    """Invalid paths / orders / sigma fail closed with ValueError."""
    rng = np.random.default_rng(111)
    good = _random_path(rng, 4, 2)
    with pytest.raises(ValueError):
        signature(good.ravel(), 2)  # not 2-D
    with pytest.raises(ValueError):
        signature(good[:1], 2)  # fewer than 2 points
    bad = good.copy()
    bad[1, 0] = np.nan
    with pytest.raises(ValueError):
        signature(bad, 2)
    bad = good.copy()
    bad[2, 1] = np.inf
    with pytest.raises(ValueError):
        logsignature(bad, 2)
    with pytest.raises(ValueError):
        signature(good, 0)
    with pytest.raises(ValueError):
        signature(good, 7)
    with pytest.raises(ValueError):
        logsignature(good, 5)  # logsignature capped at order 4
    with pytest.raises(ValueError):
        signature_kernel(good, good, 0)
    with pytest.raises(ValueError):
        signature_kernel(good, good, 7)
    other = _random_path(rng, 4, 3)
    with pytest.raises(ValueError):
        signature_kernel(good, other, 2)  # channel mismatch
    with pytest.raises(ValueError):
        signature_kernel(good, good, 2, sigma=0.0)
    with pytest.raises(ValueError):
        signature_kernel(good, good, 2, sigma=-1.0)
    with pytest.raises(ValueError):
        signature_kernel(good, good, 2, sigma=np.inf)
    with pytest.raises(ValueError):
        lead_lag_transform(good[:1])
