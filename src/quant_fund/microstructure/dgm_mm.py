"""Deep Galerkin solver for the market-making inventory PDE.

Sirignano & Spiliopoulos (2018) DGM applied to the linear inventory-value
kernel underlying Avellaneda–Stoikov: for a maker holding ``q`` units at
time ``t`` under exponential utility, the risk-bearing kernel
``u(t, q)`` solves

    u_t - (gamma * sigma^2 / 2) * q^2 * u = 0,   u(T, q) = 1,

with exact solution ``u(t, q) = exp(-0.5 * gamma * sigma^2 * q^2 * (T - t))``
— the discount factor a risk-averse maker applies to a held inventory over
the remaining horizon. The reservation-price shift (price adjustment per
unit of inventory that leaves expected utility unchanged) is

    rho(t, q) = (1/gamma) * ln(u(t, q) / u(t, q + 1))
              = -0.5 * gamma * sigma^2 * (T - t) * (2q + 1),

i.e. a linear-in-inventory slope proportional to remaining time — the
exact AS skew. DGM recovers both without ever being shown the formula:
``u_theta(t, q)`` is trained on the PDE residual over a sampled (t, q)
cylinder plus the terminal condition, and ``rho`` is re-derived from the
learned ``u`` — so a network that memorized the kernel but violated the
PDE would still fail the residual gate.

``dgm_mm_bench`` is the sealed drill: solver error vs the closed form
(interior residual RMSE, boundary error, reservation-slope error) with a
``SYNTHETIC`` label — this is a numerical-methods lane, not market
evidence.

Torch is the optional ``nn`` extra, imported lazily via :func:`_torch`
(mirrors ``rl_market_maker``/``deep_hedging``): the module imports
torch-free and every solver entry point raises ``ImportError`` with
install guidance. Training is CPU single-thread, deterministic given
``seed``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

DGM_MM_SCHEMA = "dgm_mm.v1"


def _torch() -> Any:
    """Lazily import torch (optional ``nn`` extra); fail closed with guidance."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "the DGM solver needs the optional 'nn' extra (torch): uv sync --extra nn"
        ) from exc
    return torch


def _positive_float(name: str, x: float, floor: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {x!r}") from exc
    if not math.isfinite(v) or v <= floor:
        raise ValueError(f"{name} must be a finite number > {floor}, got {x!r}")
    return v


def _positive_int(name: str, x: int, floor: int = 1) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < floor:
        raise ValueError(f"{name} must be an int >= {floor}, got {x!r}")
    return x


def _seed_int(x: int) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise ValueError(f"seed must be a non-negative int, got {x!r}")
    return int(x)


@dataclass(frozen=True)
class InventoryPDESpec:
    """The inventory kernel problem on ``[0, T] x [-q_max, q_max]``.

    PDE: ``u_t - 0.5 * gamma * sigma^2 * q^2 * u = 0`` on the interior with
    terminal condition ``u(T, q) = 1`` and spatial boundary pinned to the
    exact kernel at ``|q| = q_max`` (the Dirichlet condition that makes
    the cylinder problem well-posed for a parabolic-in-t equation).
    """

    T: float = 1.0
    gamma: float = 0.5
    sigma: float = 0.2
    q_max: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "T", _positive_float("T", self.T))
        object.__setattr__(self, "gamma", _positive_float("gamma", self.gamma))
        object.__setattr__(self, "sigma", _positive_float("sigma", self.sigma))
        object.__setattr__(self, "q_max", _positive_float("q_max", self.q_max))

    @property
    def k(self) -> float:
        """``0.5 * gamma * sigma^2`` — the curvature of the kernel."""
        return 0.5 * self.gamma * self.sigma * self.sigma

    def exact(self, t: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
        """``exp(-k * q^2 * (T - t))`` pointwise; exact on the cylinder."""
        tt = np.asarray(t, dtype=float)
        qq = np.asarray(q, dtype=float)
        return np.exp(-self.k * np.square(qq) * (self.T - tt))

    def exact_reservation(
        self, t: NDArray[np.float64], q: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """``rho(t,q) = k (T - t) (2q + 1) / gamma`` — the forward
        indifference price: compensation demanded to hold one more unit
        (the discrete skew matching AS's ``q gamma sigma^2 (T - t)``
        inventory shading under the opposite sign convention)."""
        tt = np.asarray(t, dtype=float)
        qq = np.asarray(q, dtype=float)
        return self.k * (self.T - tt) * (2.0 * qq + 1.0) / self.gamma


class DGMInventorySolver:
    """Sirignano–Spiliopoulos DGM on the (t, q) cylinder.

    Network: tanh MLP (``2 -> hidden -> ... -> 1``) for the potential
    ``phi_theta(t, q) = -ln u`` — linear PDE ``phi_t + k q^2 = 0``,
    polynomial target, far better conditioned than fitting ``u`` and
    log-rationg it afterward.

    Loss per step — the three DGM terms, resampled fresh each iteration
    (the Galerkin step never reuses collocation points):

    - interior: ``|phi_t + k q^2|^2`` at uniform samples in the cylinder
    - terminal: ``|phi(T, q)|^2`` at uniform ``q``
    - boundary: ``|phi(t, +/-q_max) - k q_max^2 (T - t)|^2`` at uniform ``t``

    Deterministic given ``seed``: seeded torch + a numpy collocation RNG.
    """

    def __init__(
        self,
        spec: InventoryPDESpec,
        *,
        hidden: int = 48,
        depth: int = 3,
        lr: float = 1e-3,
        n_interior: int = 512,
        n_terminal: int = 128,
        n_boundary: int = 128,
        seed: int = 0,
    ) -> None:
        torch = _torch()
        self.spec = spec
        self.hidden = _positive_int("hidden", hidden)
        self.depth = _positive_int("depth", depth, floor=1)
        self.n_interior = _positive_int("n_interior", n_interior)
        self.n_terminal = _positive_int("n_terminal", n_terminal)
        self.n_boundary = _positive_int("n_boundary", n_boundary)
        self.seed = _seed_int(seed)
        if not (math.isfinite(lr) and lr > 0.0):
            raise ValueError(f"lr must be positive and finite, got {lr}")
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        layers: list[Any] = []
        in_dim = 2
        for _ in range(self.depth):
            lin = torch.nn.Linear(in_dim, self.hidden)
            torch.nn.init.xavier_normal_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
            layers += [lin, torch.nn.Tanh()]
            in_dim = self.hidden
        out = torch.nn.Linear(in_dim, 1)
        torch.nn.init.xavier_normal_(out.weight)
        torch.nn.init.zeros_(out.bias)
        layers.append(out)
        self._net = torch.nn.Sequential(*layers)
        self._opt = torch.optim.AdamW(self._net.parameters(), lr=lr)
        self._rng = np.random.default_rng(self.seed)
        self.loss_history: list[dict[str, float]] = []

    def _u(self, t: Any, q: Any) -> Any:
        """Potential network on *normalized* inputs ``(t/T, q/q_max)`` —
        tanh MLPs need unit-scale inputs or the steep quadratic corner
        never converges. Callers pass raw ``(t, q)``."""
        torch = _torch()
        z = torch.stack([t / self.spec.T, q / self.spec.q_max], dim=-1)
        return self._net(z).squeeze(-1)

    def _sample(self) -> tuple[tuple[Any, Any], tuple[Any, Any], tuple[Any, Any]]:
        torch = _torch()
        s = self.spec
        t_i = torch.tensor(self._rng.uniform(0.0, s.T, self.n_interior), dtype=torch.float32)
        q_i = torch.tensor(
            self._rng.uniform(-s.q_max, s.q_max, self.n_interior),
            dtype=torch.float32,
        )
        t_t = torch.full((self.n_terminal,), s.T, dtype=torch.float32)
        q_t = torch.tensor(
            self._rng.uniform(-s.q_max, s.q_max, self.n_terminal),
            dtype=torch.float32,
        )
        t_b = torch.tensor(self._rng.uniform(0.0, s.T, self.n_boundary), dtype=torch.float32)
        q_b = torch.tensor(
            self._rng.choice([-s.q_max, s.q_max], self.n_boundary),
            dtype=torch.float32,
        )
        return (t_i, q_i), (t_t, q_t), (t_b, q_b)

    def step(self) -> dict[str, float]:
        """One Galerkin step on a fresh collocation cloud; returns the
        three loss terms (post-step)."""
        torch = _torch()
        s = self.spec
        (t_i, q_i), (t_t, q_t), (t_b, q_b) = self._sample()
        self._opt.zero_grad(set_to_none=True)
        t_i = t_i.requires_grad_(True)
        phi_i = self._u(t_i, q_i)
        phi_t = torch.autograd.grad(phi_i.sum(), t_i, create_graph=True)[0]
        resid = phi_t + s.k * torch.square(q_i)
        l_int = torch.mean(torch.square(resid))
        phi_term = self._u(t_t, q_t)
        l_term = torch.mean(torch.square(phi_term))
        phi_b = self._u(t_b, q_b)
        phi_b_exact = torch.tensor(
            s.k
            * np.square(np.asarray(q_b.detach().numpy()))
            * (s.T - np.asarray(t_b.detach().numpy())),
            dtype=torch.float32,
        )
        l_bnd = torch.mean(torch.square(phi_b - phi_b_exact))
        loss = l_int + l_term + l_bnd
        loss.backward()
        self._opt.step()
        out = {
            "interior": float(l_int.item()),
            "terminal": float(l_term.item()),
            "boundary": float(l_bnd.item()),
            "total": float(loss.item()),
        }
        self.loss_history.append(out)
        return out

    def train(self, n_steps: int) -> None:
        _positive_int("n_steps", n_steps)
        for _ in range(n_steps):
            self.step()

    def phi(self, t: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
        """``phi_theta(t, q) = -ln u_theta`` on numpy inputs (no grad)."""
        torch = _torch()
        tt = torch.tensor(np.asarray(t, dtype=float), dtype=torch.float32)
        qq = torch.tensor(np.asarray(q, dtype=float), dtype=torch.float32)
        with torch.no_grad():
            out: NDArray[np.float64] = self._u(tt, qq).numpy().astype(float)
        return out

    def predict(self, t: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
        """``u_theta(t, q) = exp(-phi_theta)`` on numpy inputs."""
        return np.exp(-np.asarray(self.phi(t, q), dtype=float))

    def reservation(self, t: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
        """``rho(t,q) = (phi(t,q+1) - phi(t,q)) / gamma`` from the learned
        potential — the indifference price per unit, never fit directly."""
        tt = np.asarray(t, dtype=float)
        qq = np.asarray(q, dtype=float)
        return (self.phi(tt, qq + 1.0) - self.phi(tt, qq)) / self.spec.gamma

    def residual_rmse(self, n: int = 2048, seed: int = -1) -> float:
        """Interior PDE residual RMSE on a fresh cloud — the metric that
        can't be gamed by memorizing the kernel."""
        torch = _torch()
        s = self.spec
        rng = np.random.default_rng(seed if seed >= 0 else self.seed + 1)
        t = torch.tensor(rng.uniform(0.0, s.T, n), dtype=torch.float32)
        q = torch.tensor(rng.uniform(-s.q_max, s.q_max, n), dtype=torch.float32)
        t.requires_grad_(True)
        phi = self._u(t, q)
        phi_t = torch.autograd.grad(phi.sum(), t, create_graph=True)[0]
        resid = phi_t + s.k * torch.square(q)
        return float(torch.mean(torch.square(resid)).item() ** 0.5)


def dgm_mm_bench(
    *,
    spec: InventoryPDESpec | None = None,
    n_steps: int = 800,
    n_grid: int = 33,
    hidden: int = 48,
    depth: int = 3,
    lr: float = 1e-3,
    n_interior: int = 512,
    n_terminal: int = 128,
    n_boundary: int = 128,
    seed: int = 0,
) -> dict[str, Any]:
    """Train the DGM solver and score it against the closed form.

    Metrics: uniform-grid MSE of ``u_theta`` vs exact, RMSE of the
    recovered reservation-price surface vs the exact AS skew, and the
    interior residual RMSE (the honesty metric — a network that matched
    the answer without satisfying the PDE still shows up here).
    SYNTHETIC drill; deterministic given ``seed``.
    """
    _torch()  # fail closed before doing anything
    _positive_int("n_steps", n_steps)
    _positive_int("n_grid", n_grid, floor=3)
    s = spec or InventoryPDESpec()
    solver = DGMInventorySolver(
        s,
        hidden=hidden,
        depth=depth,
        lr=lr,
        n_interior=n_interior,
        n_terminal=n_terminal,
        n_boundary=n_boundary,
        seed=seed,
    )
    solver.train(n_steps)
    tg = np.linspace(0.0, s.T, n_grid)
    qg = np.linspace(-s.q_max, s.q_max, n_grid)
    tt, qq = np.meshgrid(tg, qg, indexing="ij")
    u_hat = solver.predict(tt.ravel(), qq.ravel()).reshape(tt.shape)
    u_ex = s.exact(tt, qq)
    u_mse = float(np.mean(np.square(u_hat - u_ex)))
    # rho(q) needs u(q + 1): evaluate only where q + 1 stays in-domain,
    # else the reservation surface is graded on extrapolation noise.
    interior = qq <= s.q_max - 1.0
    rho_hat = solver.reservation(tt[interior], qq[interior])
    rho_ex = s.exact_reservation(tt[interior], qq[interior])
    rho_rmse = float(np.mean(np.square(rho_hat - rho_ex)) ** 0.5)
    resid_rmse = solver.residual_rmse()
    final = solver.loss_history[-1] if solver.loss_history else {}
    frame = pl.DataFrame(
        {
            "t": tt.ravel(),
            "q": qq.ravel(),
            "u_dgm": u_hat.ravel(),
            "u_exact": u_ex.ravel(),
        }
    ).join(
        pl.DataFrame(
            {
                "t": tt[interior],
                "q": qq[interior],
                "rho_dgm": rho_hat,
                "rho_exact": rho_ex,
            }
        ),
        on=["t", "q"],
        how="left",
    )
    receipt: dict[str, Any] = {
        "schema": DGM_MM_SCHEMA,
        "kind": "dgm_mm",
        "data_label": "SYNTHETIC",
        "level": "research",
        "inputs_sha256": hash_bytes(frame.write_csv().encode("utf-8")),
        "code_revision": git_revision(),
        "params": {
            "T": s.T,
            "gamma": s.gamma,
            "sigma": s.sigma,
            "q_max": s.q_max,
            "n_steps": n_steps,
            "hidden": hidden,
            "depth": depth,
            "lr": lr,
            "n_interior": n_interior,
            "n_terminal": n_terminal,
            "n_boundary": n_boundary,
            "seed": seed,
        },
        "metrics": {
            "u_mse": u_mse,
            "rho_rmse": rho_rmse,
            "interior_residual_rmse": resid_rmse,
            "final_interior_loss": float(final.get("interior", float("nan"))),
            "final_terminal_loss": float(final.get("terminal", float("nan"))),
            "final_boundary_loss": float(final.get("boundary", float("nan"))),
        },
        "evidence": [
            "deep_galerkin_method",
            "pde_residual_training",
            "closed_form_validation",
        ],
        "claims": [
            {
                "text": (
                    "DGM recovers the AS inventory kernel and its linear "
                    "reservation-price skew to within the reported errors; "
                    "interior_residual_rmse is the PDE-consistency audit "
                    "that cannot be satisfied by memorizing the kernel"
                ),
                "kind": "numerical",
            }
        ],
    }
    return {"frame": frame, "receipt": receipt, "solver": solver}


__all__ = [
    "DGM_MM_SCHEMA",
    "DGMInventorySolver",
    "InventoryPDESpec",
    "dgm_mm_bench",
]
