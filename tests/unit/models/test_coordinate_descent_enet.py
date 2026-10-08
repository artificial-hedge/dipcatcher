import numpy as np

from quant_fund.models.coordinate_descent_enet import (
    bench_coordinate_descent_enet,
    enet_fit,
    enet_path,
)


def test_enet_sparsifies():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (100, 10))
    beta = np.r_[3.0, np.zeros(9)]
    y = x @ beta + rng.normal(0, 0.1, 100)
    b = enet_fit(x, y, lam=0.1, alpha=1.0, it=300)
    assert abs(b[0]) > 1.0
    assert (np.abs(b[1:]) < 1e-6).sum() >= 7


def test_enet_path_shapes():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (80, 8))
    y = x[:, 0] * 2 + rng.normal(0, 0.2, 80)
    path = enet_path(x, y, n_lam=10, alpha=0.8)
    betas = np.asarray(path["betas"])
    assert betas.shape == (10, 8)
    lams = np.asarray(path["lams"])
    assert (np.diff(lams) < 0).all()


def test_enet_ridge_limit_matches_closed_form() -> None:
    # alpha=0 -> pure ridge: argmin (1/2n)||y-Xb||^2 + lam/2 ||b||^2 has the
    # closed form (X'X/n + lam I)^{-1} X'y/n. The rho update used to inject
    # spurious lam(1-alpha)(beta-1) terms, hidden by the bench's alpha=1.0.
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (100, 8))
    y = x @ rng.normal(0, 1, 8) + rng.normal(0, 0.3, 100)
    n, p = x.shape
    lam = 2.0
    ref = np.linalg.solve(x.T @ x / n + lam * np.eye(p), x.T @ y / n)
    got = enet_fit(x, y, lam, alpha=0.0, it=2000)
    assert np.abs(got - ref).max() < 1e-6


def test_enet_kkt_stationarity_alpha_half() -> None:
    # KKT of (1/2n)||y-Xb||^2 + lam*a*|b| + lam(1-a)/2 b^2:
    # x_j'r/n - lam(1-a) b_j = lam*a*sign(b_j) for b_j != 0, |.| <= lam*a else.
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (120, 6))
    y = x @ rng.normal(0, 1, 6) + rng.normal(0, 0.3, 120)
    lam, alpha = 0.4, 0.5
    b = enet_fit(x, y, lam, alpha=alpha, it=3000)
    r = y - x @ b
    for j in range(x.shape[1]):
        g = float(x[:, j] @ r / x.shape[0] - lam * (1 - alpha) * b[j])
        if abs(b[j]) > 1e-6:
            assert abs(g - lam * alpha * np.sign(b[j])) < 1e-4
        else:
            assert abs(g) <= lam * alpha + 1e-4


def test_bench_coordinate_descent_enet():
    out = bench_coordinate_descent_enet(seed=554)
    assert out["synthetic_enet_true_support"] >= 4
    assert out["synthetic_enet_test_mse"] < out["synthetic_ols_test_mse"]
