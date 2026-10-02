"""Normalizing-flow tail-risk model (Exec-Summary generative item). A
quantile-map marginal flow + Gaussian-copula core learns the joint density
of factor returns: each marginal is pushed through its empirical CDF ->
normal quantiles (an exact, monotone normalizing flow), and the latent
correlation is estimated in the transformed space. Sampling inverts the
map for scenario generation.

Synthetic bench: heavy-tail factor sim (t-marginals, shared shock -> tail
dependence); flow-implied ES vs Gaussian ES vs empirical truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray


def synth_factors(n: int, rng: np.random.Generator) -> FloatArray:
    z = rng.standard_t(3.5, n)
    e = rng.standard_t(5, (n, 2))
    return 0.01 * np.column_stack([0.8 * z + e[:, 0], 0.8 * z + e[:, 1]])


def _erfinv(y: float) -> float:
    a = 0.147
    ly = np.log(max(1e-12, 1 - float(y) ** 2))
    t = 2 / (np.pi * a) + ly / 2
    return float(np.sign(float(y)) * np.sqrt(np.sqrt(t * t - ly / a) - t))


def norm_ppf(u: FloatArray) -> FloatArray:
    u = np.asarray(u, float)
    flat = [np.sqrt(2) * _erfinv(2 * float(uu) - 1) for uu in u.ravel()]
    return np.asarray(flat).reshape(u.shape)


def norm_cdf(z: FloatArray) -> FloatArray:
    from math import erf

    return np.asarray([0.5 * (1 + erf(float(zz) / np.sqrt(2))) for zz in np.asarray(z)])


@dataclass
class QuantileFlow:
    """Marginal empirical-CDF flow: z_i = Phi^-1(Fhat_i(x_i)), latent Gaussian
    with estimated correlation. Exact invertible density model."""

    def fit(self, x: FloatArray) -> None:
        self.sorted_ = [np.sort(x[:, i]) for i in range(x.shape[1])]
        u = np.column_stack([self._u_of(x[:, i], i) for i in range(x.shape[1])])
        z = norm_ppf(u)
        self.rho_ = float(np.corrcoef(z.T)[0, 1])

    def _u_of(self, v: FloatArray, i: int) -> FloatArray:
        s = self.sorted_[i]
        u = np.searchsorted(s, v, side="right") / (len(s) + 1)
        return np.clip(u, 1e-4, 1 - 1e-4)

    def sample(self, n: int, rng: np.random.Generator) -> FloatArray:
        z1 = rng.standard_normal(n)
        z2 = self.rho_ * z1 + np.sqrt(1 - self.rho_**2) * rng.standard_normal(n)
        out = np.zeros((n, 2))
        for i, zz in enumerate([z1, z2]):
            u = norm_cdf(zz)
            s = self.sorted_[i]
            out[:, i] = np.interp(u, np.linspace(1e-4, 1 - 1e-4, len(s)), s)
        return out


def gaussian_es(port_mu: float, port_sd: float, alpha: float = 0.98) -> float:
    from math import exp, pi, sqrt

    z = -sqrt(2) * _erfinv(2 * alpha - 1)
    pdf = exp(-(z**2) / 2) / sqrt(2 * pi)
    return float(-(port_mu - port_sd * pdf / (1 - alpha)))


def emp_es(port: FloatArray, alpha: float = 0.98) -> float:
    thr = np.quantile(port, 1 - alpha)
    return float(-port[port <= thr].mean())


def bench_risk_flow(seed: int = 41) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x = synth_factors(3000, rng)
    flow = QuantileFlow()
    flow.fit(x)
    port = 0.5 * (x[:, 0] + x[:, 1])
    es_true = emp_es(port)
    es_g = gaussian_es(float(port.mean()), float(port.std()))
    draws = flow.sample(20000, rng)
    port_s = 0.5 * (draws[:, 0] + draws[:, 1])
    es_flow = emp_es(port_s)
    # also joint tail dependence check: P(both in worst 2%)
    q = np.quantile(x, 0.02, axis=0)
    qs = np.quantile(draws, 0.02, axis=0)
    td_true = float(np.mean((x[:, 0] <= q[0]) & (x[:, 1] <= q[1])))
    td_flow = float(np.mean((draws[:, 0] <= qs[0]) & (draws[:, 1] <= qs[1])))
    return {
        "synthetic_flow_es": es_flow,
        "synthetic_flow_es_true": es_true,
        "synthetic_flow_es_err": abs(es_flow - es_true),
        "synthetic_flow_gauss_es": es_g,
        "synthetic_flow_gauss_es_err": abs(es_g - es_true),
        "synthetic_flow_tail_dep_err": abs(td_flow - td_true),
        "synthetic_flow_latent_rho": flow.rho_,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_risk_flow(), indent=1))
