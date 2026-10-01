"""Tests for mean_field_games.py — Cardaliaguet–Lehalle trade-crowding MFG.

All tests are **SYNTHETIC**: deterministic seeds, grid parameters fixed.
No market data, no live-trading claims.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.mean_field_games import (
    MFGConfig,
    MFGSolution,
    ac_benchmark_trajectory,
    mfg_to_ac_alpha,
    solve_mfg_trade_crowding,
)

# ---------------------------------------------------------------------------
# Shared configs
# ---------------------------------------------------------------------------

_SMALL_CFG = MFGConfig(
    Q0=1.0,
    T=1.0,
    kappa=1.0,
    gamma=0.0,
    psi=0.5,
    terminal_penalty=10.0,
    n_t=60,
    n_q=60,
)

_CROWDED_CFG = MFGConfig(
    Q0=1.0,
    T=1.0,
    kappa=1.0,
    gamma=0.4,
    psi=0.5,
    terminal_penalty=10.0,
    n_t=60,
    n_q=60,
)


# ---------------------------------------------------------------------------
# Fail-closed edges
# ---------------------------------------------------------------------------


class TestFailClosed:
    """Input validation guards (tested via the solver entry point)."""

    def _bad_cfg(self, **overrides) -> MFGConfig:
        base = dict(Q0=1.0, T=1.0, kappa=1.0, gamma=0.0, psi=1.0, terminal_penalty=1.0)
        base.update(overrides)
        return MFGConfig(**base)  # type: ignore[arg-type]

    def test_kappa_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="kappa"):
            solve_mfg_trade_crowding(self._bad_cfg(kappa=0.0))

    def test_gamma_negative(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            solve_mfg_trade_crowding(self._bad_cfg(gamma=-0.1))

    def test_psi_negative(self) -> None:
        with pytest.raises(ValueError, match="psi"):
            solve_mfg_trade_crowding(self._bad_cfg(psi=-1.0))

    def test_terminal_penalty_negative(self) -> None:
        with pytest.raises(ValueError, match="terminal_penalty"):
            solve_mfg_trade_crowding(self._bad_cfg(terminal_penalty=-0.5))

    def test_T_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="T"):
            solve_mfg_trade_crowding(self._bad_cfg(T=0.0))

    def test_n_t_too_small(self) -> None:
        with pytest.raises(ValueError, match="n_t"):
            solve_mfg_trade_crowding(self._bad_cfg(n_t=2))

    def test_n_q_too_small(self) -> None:
        with pytest.raises(ValueError, match="n_q"):
            solve_mfg_trade_crowding(self._bad_cfg(n_q=4))

    def test_q_min_geq_q_max(self) -> None:
        with pytest.raises(ValueError, match="q_min"):
            solve_mfg_trade_crowding(self._bad_cfg(q_min=5.0, q_max=2.0))

    def test_nan_Q0_raises(self) -> None:
        """Non-finite Q0 should raise."""
        with pytest.raises(ValueError, match="Q0"):
            solve_mfg_trade_crowding(self._bad_cfg(Q0=float("nan")))


# ---------------------------------------------------------------------------
# Zero-coupling: MFG → single-agent limit
# ---------------------------------------------------------------------------


class TestZeroCoupling:
    """ε → 0 recovery of the solitary optimal execution schedule."""

    def test_small_gamma_yields_solitary_solution(self) -> None:
        """At γ ≈ 0, the MFG solution should match γ = 0 bitwise."""
        cfg_zero = MFGConfig(
            Q0=1.0,
            T=1.0,
            kappa=1.0,
            gamma=0.0,
            psi=0.5,
            terminal_penalty=10.0,
            n_t=80,
            n_q=80,
        )
        sol_zero = solve_mfg_trade_crowding(cfg_zero)

        cfg_tiny = MFGConfig(
            Q0=1.0,
            T=1.0,
            kappa=1.0,
            gamma=1e-8,
            psi=0.5,
            terminal_penalty=10.0,
            n_t=80,
            n_q=80,
        )
        sol_tiny = solve_mfg_trade_crowding(cfg_tiny)

        mse = float(np.mean((sol_zero.optimal_trajectory - sol_tiny.optimal_trajectory) ** 2))
        assert mse < 1e-8, f"γ→0 trajectory MSE = {mse:.2e} > 1e-8"

    def test_solitary_mfg_vs_ac_benchmark(self) -> None:
        """The solitary (γ=0) MFG trajectory should match the AC closed form.

        With terminal_penalty large (hard liquidation), the trading rate
        α(t) from the MFG should approach the AC schedule:
            α_AC(t) = −Q₀ ω cosh(ω(T−t)) / sinh(ω T).
        """
        T_val = 1.0
        Q0 = 1.0
        kappa_val = 0.5
        psi_val = 0.6
        A = 1e6  # near-infinite terminal penalty → hard liquidation

        cfg = MFGConfig(
            Q0=Q0,
            T=T_val,
            kappa=kappa_val,
            gamma=0.0,
            psi=psi_val,
            terminal_penalty=A,
            n_t=400,
            n_q=200,
        )
        sol = solve_mfg_trade_crowding(cfg)

        n_slices = 100
        alpha_mfg = mfg_to_ac_alpha(sol, n_slices=n_slices)
        alpha_ac = ac_benchmark_trajectory(Q0, T_val, n_slices, kappa=kappa_val, psi=psi_val)

        rmse = float(np.sqrt(np.mean((alpha_mfg - alpha_ac) ** 2)))
        norm = float(np.max(np.abs(alpha_ac)))
        rel_rmse = rmse / max(norm, 1e-10)
        assert rel_rmse < 0.08, f"MFG-AC α relative RMSE = {rel_rmse:.4f} >= 0.08"

    def test_solitary_trajectory_decays_to_zero(self) -> None:
        """With high terminal penalty, q*(T) should be near zero."""
        sol = solve_mfg_trade_crowding(_SMALL_CFG)
        q_final = float(sol.optimal_trajectory[-1])
        assert abs(q_final) < 0.2, f"terminal inventory {q_final:.4f} not near zero"

    def test_solitary_alpha_is_negative_for_liquidation(self) -> None:
        """For Q0 > 0 (long), α* should be negative (selling) at q = q*(t)."""
        sol = solve_mfg_trade_crowding(_SMALL_CFG)
        # α*(t, q*(t)) = q*'(t) should be negative for liquidation
        q_traj = sol.optimal_trajectory
        mask = q_traj > 0.01
        if np.any(mask):
            alpha_at_qstar = np.array(
                [float(np.interp(q_traj[k], sol.q_grid, sol.alpha[k])) for k in range(len(q_traj))]
            )
            assert np.all(alpha_at_qstar[mask] < 0), (
                "α*(t,q*) should be negative when inventory remains"
            )


# ---------------------------------------------------------------------------
# Crowding externality
# ---------------------------------------------------------------------------


class TestCrowdingExternality:
    """The MFG equilibrium cost exceeds the solitary cost."""

    def test_crowding_cost_ratio_gt_one(self) -> None:
        """At positive γ, the crowding cost ratio must be > 1."""
        sol = solve_mfg_trade_crowding(_CROWDED_CFG)
        assert sol.crowding_cost_ratio > 1.0, (
            f"crowding_cost_ratio = {sol.crowding_cost_ratio:.4f} should be > 1"
        )

    def test_crowding_ratio_increases_with_gamma(self) -> None:
        """Higher coupling γ should increase the crowding cost ratio."""
        gammas = [0.01, 0.05, 0.2]
        ratios = []
        for g in gammas:
            cfg = MFGConfig(
                Q0=1.0,
                T=1.0,
                kappa=1.0,
                gamma=g,
                psi=0.5,
                terminal_penalty=10.0,
                n_t=60,
                n_q=60,
            )
            sol = solve_mfg_trade_crowding(cfg)
            ratios.append(sol.crowding_cost_ratio)
        assert ratios == sorted(ratios), (
            f"crowding cost ratios {ratios} should be monotonic increasing"
        )

    def test_crowded_trajectory_deviates_from_solitary(self) -> None:
        """With γ > 0, the equilibrium trajectory should differ."""
        sol_solitary = solve_mfg_trade_crowding(_SMALL_CFG)
        sol_crowded = solve_mfg_trade_crowding(_CROWDED_CFG)

        mse = float(
            np.mean((sol_solitary.optimal_trajectory - sol_crowded.optimal_trajectory) ** 2)
        )
        assert mse > 1e-8, f"crowded trajectory should differ from solitary (MSE={mse:.2e})"

    def test_crowding_slows_liquidation(self) -> None:
        """Crowding (γ > 0) should make agents trade slower.

        Intuition: when the aggregate trading rate pushes the price
        against each agent (permanent impact), the equilibrium
        response is to trade more slowly to reduce per-unit adverse
        drift.  This is the "crowding externality" — the opposite
        of "race to the bottom" front-loading.
        """
        sol_solitary = solve_mfg_trade_crowding(_SMALL_CFG)
        sol_crowded = solve_mfg_trade_crowding(_CROWDED_CFG)

        def _half_life(q_traj: np.ndarray, t_grid: np.ndarray) -> float:
            half = q_traj[0] / 2.0
            for k in range(len(q_traj)):
                if q_traj[k] <= half:
                    return float(t_grid[k])
            return float(t_grid[-1])

        hl_sol = _half_life(sol_solitary.optimal_trajectory, sol_solitary.t_grid)
        hl_crowd = _half_life(sol_crowded.optimal_trajectory, sol_crowded.t_grid)
        # Crowded equilibrium should have slower liquidation
        assert hl_crowd >= hl_sol, (
            f"crowded half-life {hl_crowd:.4f} should be ≥ solitary {hl_sol:.4f}"
        )


# ---------------------------------------------------------------------------
# Grid convergence
# ---------------------------------------------------------------------------


class TestGridConvergence:
    """Refining the time grid should reduce the discretisation error."""

    def test_cost_converges_with_grid_refinement(self) -> None:
        """Cost difference between N and 2N time steps should shrink."""
        base_n = 50

        cfg_c = MFGConfig(
            Q0=1.0,
            T=1.0,
            kappa=1.0,
            gamma=0.1,
            psi=0.5,
            terminal_penalty=10.0,
            n_t=base_n,
            n_q=80,
        )
        coarse = solve_mfg_trade_crowding(cfg_c)

        cfg_f = MFGConfig(
            Q0=1.0,
            T=1.0,
            kappa=1.0,
            gamma=0.1,
            psi=0.5,
            terminal_penalty=10.0,
            n_t=2 * base_n,
            n_q=80,
        )
        fine = solve_mfg_trade_crowding(cfg_f)

        cost_diff = abs(coarse.cost - fine.cost)
        rel_diff = cost_diff / max(abs(coarse.cost), 1e-10)
        assert rel_diff < 0.1, f"cost convergence poor: |ΔC|/|C| = {rel_diff:.4f} >= 0.10"

    def test_trajectory_l2_converges_with_grid(self) -> None:
        """L2 norm of trajectory difference between coarse and fine grids."""
        base_n = 50

        cfg_c = MFGConfig(
            Q0=1.0,
            T=1.0,
            kappa=1.0,
            gamma=0.1,
            psi=0.5,
            terminal_penalty=10.0,
            n_t=base_n,
            n_q=80,
        )
        coarse = solve_mfg_trade_crowding(cfg_c)

        cfg_f = MFGConfig(
            Q0=1.0,
            T=1.0,
            kappa=1.0,
            gamma=0.1,
            psi=0.5,
            terminal_penalty=10.0,
            n_t=2 * base_n,
            n_q=80,
        )
        fine = solve_mfg_trade_crowding(cfg_f)

        # Interpolate coarse onto fine time grid
        t_coarse = np.linspace(0, 1.0, base_n + 1)
        t_fine = np.linspace(0, 1.0, 2 * base_n + 1)
        q_interp = np.interp(t_fine, t_coarse, coarse.optimal_trajectory)

        l2 = float(np.sqrt(np.mean((q_interp - fine.optimal_trajectory) ** 2)))
        assert l2 < 0.05, f"L2 trajectory discrepancy {l2:.4f} >= 0.05"


# ---------------------------------------------------------------------------
# Determinism and invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    """Structural invariants the solver must preserve."""

    def test_deterministic_output(self) -> None:
        """Same config → same output (bitwise)."""
        sol1 = solve_mfg_trade_crowding(_CROWDED_CFG)
        sol2 = solve_mfg_trade_crowding(_CROWDED_CFG)
        np.testing.assert_array_equal(sol1.t_grid, sol2.t_grid)
        np.testing.assert_array_equal(sol1.optimal_trajectory, sol2.optimal_trajectory)
        assert sol1.cost == sol2.cost
        assert sol1.n_fp_iters == sol2.n_fp_iters

    def test_density_normalised(self) -> None:
        """The Fokker–Planck density must integrate to 1 at every time step."""
        sol = solve_mfg_trade_crowding(_CROWDED_CFG)
        dq = sol.q_grid[1] - sol.q_grid[0]
        for k in range(len(sol.t_grid)):
            mass = float(np.sum(sol.density[k]) * dq)
            assert abs(mass - 1.0) < 1e-10, f"mass at t={k}: {mass:.2e} ≠ 1"

    def test_value_terminal_condition(self) -> None:
        """v(T, q) = A q² must hold at the terminal step."""
        sol = solve_mfg_trade_crowding(_SMALL_CFG)
        A = _SMALL_CFG.terminal_penalty
        v_T_expected = A * sol.q_grid**2
        np.testing.assert_allclose(sol.value[-1], v_T_expected, atol=1e-12)

    def test_mu_equals_agent_rate_when_gamma_zero(self) -> None:
        """At γ = 0, μ̄_t equals the solitary agent's own trading rate.

        There is no mean-field coupling, so the aggregate rate is just
        the individual optimal trading rate — nonzero for liquidation.
        """
        sol = solve_mfg_trade_crowding(_SMALL_CFG)
        # α* evaluated at q*(t): the agent's own optimal trading speed
        alpha_at_qstar = np.array(
            [
                float(np.interp(sol.optimal_trajectory[k], sol.q_grid, sol.alpha[k]))
                for k in range(len(sol.t_grid))
            ]
        )
        # μ̄ should equal α* at q* (within interpolation tolerance)
        np.testing.assert_allclose(sol.mu, alpha_at_qstar, atol=1e-6)
        # And for liquidation (Q0 > 0), μ̄ should be negative
        assert np.all(sol.mu[0] < 0), "μ̄ should be negative for liquidation"

    def test_density_nonnegative(self) -> None:
        """Density must be non-negative everywhere."""
        sol = solve_mfg_trade_crowding(_CROWDED_CFG)
        assert np.all(sol.density >= -1e-15), "density has negative entries"

    def test_alpha_consistency(self) -> None:
        """α*(t,q) = −∂_q v(t,q) / (2κ) must hold on interior q-grid points.

        Boundary points have larger finite-difference errors, so we
        only check the interior (trimming one cell from each edge).
        """
        sol = solve_mfg_trade_crowding(_CROWDED_CFG)
        dq = sol.q_grid[1] - sol.q_grid[0]
        for k in [0, len(sol.t_grid) // 2, len(sol.t_grid) - 1]:
            dv = (sol.value[k, 2:] - sol.value[k, :-2]) / (2.0 * dq)
            expected = -dv / (2.0 * _CROWDED_CFG.kappa)
            # Interior only: skip boundary cells
            np.testing.assert_allclose(sol.alpha[k, 1:-1], expected, atol=1e-12)

    def test_h1_terminal_zero(self) -> None:
        """h₁(T) must be zero (terminal condition)."""
        sol = solve_mfg_trade_crowding(_CROWDED_CFG)
        assert abs(sol.h1[-1]) < 1e-12, f"h₁(T) = {sol.h1[-1]:.2e} ≠ 0"

    def test_h2_terminal_equals_A(self) -> None:
        """h₂(T) must equal the terminal penalty A."""
        sol = solve_mfg_trade_crowding(_CROWDED_CFG)
        assert abs(sol.h2[-1] - _CROWDED_CFG.terminal_penalty) < 1e-12

    def test_h0_terminal_zero(self) -> None:
        """h₀(T) must be zero (terminal condition)."""
        sol = solve_mfg_trade_crowding(_CROWDED_CFG)
        assert abs(sol.h0[-1]) < 1e-12, f"h₀(T) = {sol.h0[-1]:.2e} ≠ 0"

    def test_lq_solver_no_iteration(self) -> None:
        """The LQ solver should always converge in 1 iteration."""
        sol = solve_mfg_trade_crowding(_CROWDED_CFG)
        assert sol.n_fp_iters == 1
        assert sol.fp_residual == 0.0


# ---------------------------------------------------------------------------
# AC benchmark helpers
# ---------------------------------------------------------------------------


class TestACBenchmark:
    """The analytic Almgren–Chriss benchmark helper."""

    def test_ac_shape(self) -> None:
        """AC α(t) should be negative and monotone-decreasing in magnitude."""
        alpha = ac_benchmark_trajectory(Q0=1.0, T=1.0, n_slices=50, kappa=1.0, psi=0.5)
        assert alpha.shape == (50,)
        assert np.all(alpha < 0), "α should be negative for liquidation"
        # |α| should decrease over time (later slices gentler)
        assert np.all(np.diff(np.abs(alpha)) <= 0), "|α| should decrease monotonically"

    def test_ac_total_volume(self) -> None:
        """The AC schedule should liquidate the full quantity."""
        Q0 = 1.0
        T_val = 1.0
        n = 100
        alpha = ac_benchmark_trajectory(Q0=Q0, T=T_val, n_slices=n, kappa=1.0, psi=0.5)
        total = float(np.sum(-alpha) * (T_val / n))
        assert abs(total - Q0) < 0.02, f"total traded {total:.4f} ≠ Q0={Q0}"

    def test_ac_fail_closed_psi(self) -> None:
        with pytest.raises(ValueError, match="psi"):
            ac_benchmark_trajectory(Q0=1.0, T=1.0, n_slices=10, kappa=1.0, psi=0.0)

    def test_ac_large_omega_runs(self) -> None:
        """Very large omega should produce a well-behaved schedule (no overflow)."""
        alpha = ac_benchmark_trajectory(Q0=1.0, T=10.0, n_slices=50, kappa=1e-6, psi=100.0)
        assert np.all(np.isfinite(alpha))

    def test_mfg_to_ac_alpha_shape(self) -> None:
        sol = solve_mfg_trade_crowding(_SMALL_CFG)
        alpha = mfg_to_ac_alpha(sol, n_slices=30)
        assert alpha.shape == (30,)
        assert np.all(np.isfinite(alpha))


# ---------------------------------------------------------------------------
# Bench keys
# ---------------------------------------------------------------------------


class TestBenchKeys:
    """Tests producing numbers for suggested bench keys.

    These are source-of-truth tests asserting expected behaviour.
    """

    def test_mfg_ac_recovery_mse(self) -> None:
        """mfg_ac_recovery_mse: MSE between MFG γ=0 α and AC α schedule."""
        T_val = 1.0
        Q0 = 1.0
        kappa_val = 0.5
        psi_val = 0.6
        A_val = 1e6

        cfg = MFGConfig(
            Q0=Q0,
            T=T_val,
            kappa=kappa_val,
            gamma=0.0,
            psi=psi_val,
            terminal_penalty=A_val,
            n_t=400,
            n_q=200,
        )
        sol = solve_mfg_trade_crowding(cfg)
        alpha_mfg = mfg_to_ac_alpha(sol, n_slices=100)
        alpha_ac = ac_benchmark_trajectory(Q0, T_val, 100, kappa=kappa_val, psi=psi_val)
        mse = float(np.mean((alpha_mfg - alpha_ac) ** 2))
        assert mse < 0.01, f"mfg_ac_recovery_mse = {mse:.6f} >= 0.01"

    def test_mfg_crowding_cost_ratio(self) -> None:
        """mfg_crowding_cost_ratio: equilibrium vs solitary cost > 1."""
        sol = solve_mfg_trade_crowding(_CROWDED_CFG)
        ratio = sol.crowding_cost_ratio
        assert 1.0 < ratio < 15.0, f"mfg_crowding_cost_ratio = {ratio:.4f} out of expected range"

    def test_mfg_grid_convergence_rate(self) -> None:
        """mfg_grid_convergence_rate: cost convergence with grid refinement."""
        base_ns = [30, 50, 80]
        costs = []
        for ns in base_ns:
            cfg = MFGConfig(
                Q0=1.0,
                T=1.0,
                kappa=1.0,
                gamma=0.1,
                psi=0.5,
                terminal_penalty=10.0,
                n_t=ns,
                n_q=80,
            )
            sol = solve_mfg_trade_crowding(cfg)
            costs.append(sol.cost)

        diffs = [abs(costs[i] - costs[i + 1]) for i in range(len(costs) - 1)]
        assert all(diffs[i] <= diffs[0] * 2 for i in range(1, len(diffs))), (
            f"cost convergence not monotonic: costs={costs}, diffs={diffs}"
        )

    def test_fixed_point_iters_is_one(self) -> None:
        """LQ solver always converges in 1 step."""
        sol = solve_mfg_trade_crowding(_CROWDED_CFG)
        assert sol.n_fp_iters == 1, f"n_fp_iters = {sol.n_fp_iters} ≠ 1"


# ---------------------------------------------------------------------------
# State space
# ---------------------------------------------------------------------------


class TestStateSpace:
    def test_initial_mean_near_Q0(self) -> None:
        """The initial density mean should be ≈ Q0."""
        cfg = MFGConfig(
            Q0=1.5,
            T=1.0,
            kappa=1.0,
            gamma=0.0,
            psi=0.5,
            terminal_penalty=10.0,
            n_t=60,
            n_q=80,
        )
        sol = solve_mfg_trade_crowding(cfg)
        dq = sol.q_grid[1] - sol.q_grid[0]
        mean_q0 = float(np.sum(sol.q_grid * sol.density[0]) * dq)
        assert abs(mean_q0 - cfg.Q0) < dq, f"initial mean q={mean_q0:.4f} far from Q0={cfg.Q0}"

    def test_custom_q_bounds(self) -> None:
        cfg = MFGConfig(
            Q0=1.0,
            T=1.0,
            kappa=1.0,
            gamma=0.0,
            psi=0.5,
            terminal_penalty=10.0,
            n_t=60,
            n_q=50,
            q_min=-0.5,
            q_max=2.5,
        )
        sol = solve_mfg_trade_crowding(cfg)
        assert sol.q_grid[0] == pytest.approx(-0.5)
        assert sol.q_grid[-1] == pytest.approx(2.5)

    def test_solution_has_all_fields(self) -> None:
        sol = solve_mfg_trade_crowding(_CROWDED_CFG)
        assert isinstance(sol, MFGSolution)
        for attr in [
            "t_grid",
            "q_grid",
            "value",
            "density",
            "alpha",
            "mu",
            "optimal_trajectory",
            "h2",
            "h1",
            "h0",
        ]:
            val = getattr(sol, attr)
            assert isinstance(val, np.ndarray), f"{attr} not ndarray, got {type(val)}"
        for attr in [
            "n_fp_iters",
            "fp_residual",
            "cost",
            "solitary_cost",
            "crowding_cost_ratio",
        ]:
            val = getattr(sol, attr)
            assert isinstance(val, (int, float)), f"{attr} not numeric, got {type(val)}"

    def test_large_terminal_penalty_runs(self) -> None:
        """The solver should handle A = 1e12 without overflow."""
        cfg = MFGConfig(
            Q0=1.0,
            T=1.0,
            kappa=1.0,
            gamma=0.0,
            psi=0.5,
            terminal_penalty=1e12,
            n_t=200,
            n_q=100,
        )
        sol = solve_mfg_trade_crowding(cfg)
        assert np.all(np.isfinite(sol.value))
        assert np.all(np.isfinite(sol.optimal_trajectory))


# ---------------------------------------------------------------------------
# Analytical h₂ verification
# ---------------------------------------------------------------------------


class TestH2Analytics:
    """Verify h₂(t) against known analytic limits."""

    def test_h2_psi_zero(self) -> None:
        """At ψ = 0, h₂(t) = κ / (κ/A + (T−t))."""
        T_val = 1.0
        kappa_val = 2.0
        A_val = 3.0
        cfg = MFGConfig(
            Q0=1.0,
            T=T_val,
            kappa=kappa_val,
            gamma=0.0,
            psi=0.0,
            terminal_penalty=A_val,
            n_t=100,
        )
        sol = solve_mfg_trade_crowding(cfg)
        t = sol.t_grid
        expected = kappa_val / (kappa_val / A_val + (T_val - t))
        np.testing.assert_allclose(sol.h2, expected, atol=1e-12)

    def test_h2_equilibrium(self) -> None:
        """When A = √(κψ) (the stable equilibrium), h₂(t) is constant."""
        kappa_val = 4.0
        psi_val = 9.0
        A_val = math.sqrt(kappa_val * psi_val)  # = 6
        cfg = MFGConfig(
            Q0=1.0,
            T=1.0,
            kappa=kappa_val,
            gamma=0.0,
            psi=psi_val,
            terminal_penalty=A_val,
            n_t=100,
        )
        sol = solve_mfg_trade_crowding(cfg)
        np.testing.assert_allclose(sol.h2, A_val, atol=1e-12)
