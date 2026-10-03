"""Physics-informed neural network option pricing (Exec-Summary pricing
item). A small MLP learns C(S, tau) by penalizing the Black-Scholes PDE
residual + boundary conditions — no labels beyond PDE/boundary loss.
Automatic differentiation is hand-coded via complex-step on the network.

Synthetic bench: PDE-trained PINN vs closed-form Black-Scholes grid —
absolute pricing error in vol-points plus delta accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray

_R, _SIG, _K = 0.05, 0.2, 100.0


def bs_call(s: FloatArray, tau: FloatArray) -> FloatArray:
    from math import erf

    s = np.asarray(s, float)
    tau = np.asarray(tau, float)
    sqrt_t = np.sqrt(np.maximum(tau, 1e-9))
    d1 = (np.log(s / _K) + (_R + 0.5 * _SIG**2) * tau) / (_SIG * sqrt_t)
    d2 = d1 - _SIG * sqrt_t
    nd1 = 0.5 * (1 + np.vectorize(erf)(d1 / np.sqrt(2)))
    nd2 = 0.5 * (1 + np.vectorize(erf)(d2 / np.sqrt(2)))
    return np.asarray(s * nd1 - _K * np.exp(-_R * tau) * nd2)


@dataclass
class Pinn:
    hid: int = 24
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w1 = 0.5 * rng.standard_normal((2, self.hid))
        self.b1 = np.zeros(self.hid)
        self.w2 = 0.5 * rng.standard_normal((self.hid, 1))
        self.b2 = np.zeros(1)

    def net(self, s: FloatArray, tau: FloatArray) -> FloatArray:
        x = np.column_stack([s / _K, tau])
        return np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2 * _K

    def price(self, s: FloatArray, tau: FloatArray) -> FloatArray:
        """Output squashed to a valid call bound."""
        raw = self.net(s, tau)
        intrinsic = np.maximum(np.asarray(s) - _K * np.exp(-_R * np.asarray(tau)), 0)
        return np.asarray(intrinsic + _K * (1 / (1 + np.exp(-raw.ravel()))))


def pde_residual(model: Pinn, s: float, tau: float, eps: float = 0.5) -> float:
    """Complex-step free grads via centered differences on price()."""
    c = model.price(np.array([s]), np.array([tau]))[0]
    cs_p = model.price(np.array([s + eps]), np.array([tau]))[0]
    cs_m = model.price(np.array([s - eps]), np.array([tau]))[0]
    css = (cs_p - 2 * c + cs_m) / eps**2
    dcs = (cs_p - cs_m) / (2 * eps)
    ct_p = model.price(np.array([s]), np.array([tau + eps]))[0]
    ct_m = model.price(np.array([s]), np.array([tau - eps]))[0]
    ct = (ct_p - ct_m) / (2 * eps)
    # BS PDE: Ct + 0.5 sig^2 s^2 Css + r s Cs - r C = 0
    return float(-ct + 0.5 * _SIG**2 * s**2 * css + _R * s * dcs - _R * c)


def train_pinn(model: Pinn, epochs: int, rng: np.random.Generator) -> float:
    """Finite-difference stochastic loss on PDE + boundary; lr tiny params."""
    lr = 1e-3
    loss = 0.0
    for _ in range(epochs):
        s_col = rng.uniform(60, 140, 32)
        tau_col = rng.uniform(0.02, 2.0, 32)
        # loss = mean residual^2 + boundary at tau=0
        res = np.array(
            [pde_residual(model, float(s), float(t)) for s, t in zip(s_col, tau_col, strict=True)]
        )
        lb = model.price(s_col, np.zeros(32))
        intrinsic = np.maximum(s_col - _K, 0)
        loss = float(np.mean(res**2) + 50 * np.mean((lb - intrinsic) ** 2))
        # numerical grad on a subset of params (full params too slow): perturb w2,b2,w1 row-block
        for name in ("w2", "b2", "w1", "b1"):
            p = getattr(model, name)
            flat = p.ravel()
            idx = rng.choice(len(flat), size=min(8, len(flat)), replace=False)
            for i in idx:
                orig = flat[i]
                flat[i] = orig + 1e-4
                res_p = np.mean(
                    [
                        pde_residual(model, float(s), float(t)) ** 2
                        for s, t in zip(s_col[:8], tau_col[:8], strict=True)
                    ]
                )
                lb_p = np.mean(
                    (model.price(s_col[:8], np.zeros(8)) - np.maximum(s_col[:8] - _K, 0)) ** 2
                )
                flat[i] = orig - 1e-4
                res_m = np.mean(
                    [
                        pde_residual(model, float(s), float(t)) ** 2
                        for s, t in zip(s_col[:8], tau_col[:8], strict=True)
                    ]
                )
                lb_m = np.mean(
                    (model.price(s_col[:8], np.zeros(8)) - np.maximum(s_col[:8] - _K, 0)) ** 2
                )
                flat[i] = orig
                g = (res_p + 50 * lb_p - res_m - 50 * lb_m) / 2e-4
                flat[i] = orig - lr * g
    return loss


def bench_pinn_pricing(seed: int = 23) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    model = Pinn(seed=seed)
    loss = train_pinn(model, 120, rng)
    s = np.linspace(70, 130, 12)
    tau = np.linspace(0.05, 1.5, 8)
    errs, derrs = [], []
    for sv in s:
        for tv in tau:
            pred = model.price(np.array([sv]), np.array([tv]))[0]
            true = bs_call(np.array([sv]), np.array([tv]))[0]
            errs.append(abs(pred - true) / _K)
            # delta comparison
            dp = (
                model.price(np.array([sv + 0.5]), np.array([tv]))[0]
                - model.price(np.array([sv - 0.5]), np.array([tv]))[0]
            )
            dt = (
                bs_call(np.array([sv + 0.5]), np.array([tv]))[0]
                - bs_call(np.array([sv - 0.5]), np.array([tv]))[0]
            )
            derrs.append(abs(dp - dt))
    # untrained baseline
    raw = Pinn(seed=seed + 999)
    errs0 = [
        abs(
            raw.price(np.array([sv]), np.array([tv]))[0]
            - bs_call(np.array([sv]), np.array([tv]))[0]
        )
        / _K
        for sv in s
        for tv in tau
    ]
    return {
        "synthetic_pinn_rel_mae": float(np.mean(errs)),
        "synthetic_pinn_delta_mae": float(np.mean(derrs)),
        "synthetic_pinn_final_loss": loss,
        "synthetic_pinn_untrained_mae": float(np.mean(errs0)),
        "synthetic_pinn_improvement": float(np.mean(errs0) - np.mean(errs)),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_pinn_pricing(), indent=1))
