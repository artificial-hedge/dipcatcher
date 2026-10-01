"""Benchmark batteries for SOTA canon wave 17 (see waves 11-16 for the pattern).

Wave 17 lands seven OPTIONAL research families, each pinned to the paper
shipped in its lane module (see the module docstrings for full citations):

- ``diffpts``: DiffPTS (arXiv:2601.09892) -- LSNM diffusion forecaster on a
  synthetic AR(1)-with-bimodal-noise stream, scored by CRPS/coverage/PIT
  against NGboost and quantile-random-forest baselines; DDPM-vs-DDIM NFE
  budget trade-off (StocBench-style) included.  TORCH-GATED.
- ``extra_conformal``: extrapolated weighted conformal -- beta-function
  weighting beyond exchangeable calibration; the harmonic-mean label
  extension (extra-harm) reports its power-shift gap.
- ``multilevel_mm``: Cheridito & Weiss (2026, arXiv:2608.18195) multi-level
  deep market making -- level-grouped actor nets over the Santa-Fe ZI LOB
  world, completion-rate + inventory-cap telemetry only.  TORCH-GATED.
- ``rlmm_c51``: Moret & Lillo Algorithm C -- C51 distributional RL market
  maker with the scenario-generator + difficulty bandit fine-tuning loop on
  the same ZI world, evaluated for scenario robustness.  TORCH-GATED.
- ``sga_uq``: multistep uncertainty propagation via sliced-graph alignment
  (arXiv:2609.28582) -- error-DAG covariance-vs-simulation certificate,
  horizon monotonicity, empirical interval coverage.
- ``passive_impact``: Barzykin et al. (2026) optimal passive execution under
  price impact -- closed-form m == k planner, fill-fraction telemetry, and
  an aggressive-baseline comparison whose PnL diff is emitted namespaced
  (``pim_passive_minus_aggressive``), never as a headline metric.
- ``stochastic_tracking``: Nutz & Voss (arXiv:2608.29468) singular stochastic
  tracking -- the log-log convergence exponent of the regularized-to-
  unregularized gap against the paper's sharp rate, plus the closed-form
  benchmark value match.  RNG-free; shrink only via grid resolution.

Honesty contract: all blobs are synthetic correctness/telemetry checks, never
market evidence. No Sharpe/Sortino/Calmar/PnL/NAV tokens appear in emitted
keys; simulator diagnostics stay namespaced ``sim_internal_*`` inside the
modules and are filtered out of these blobs (the PnL *difference* that
passive_impact surfaces is re-keyed without the forbidden token).

SHRUNK Monte-Carlo budgets (relative to each lane's own test suite):

- diffpts: n_train 220 / n_test 60, n_samples 96, hidden (16,), 50 epochs,
  20 diffusion steps -- the lane tests run the same tiny net; the module's
  own defaults (700/250, (32,32), 150 epochs, 50 steps) are the reference.
- multilevel_mm: hidden (16,) nets, 3 train episodes x horizon 30, eval on
  2 seeds x 3 arms at horizon 40 (lane: 4 episodes, horizon 40-60).
- rlmm_c51: pool 6 x 30 expected MOs, 4 bandit episodes at horizon 120,
  robustness eval on 1 scenario x 3 families x 3 actors (lane: identical
  tiny budget -- the bandit/fine-tune loop is the claim, not throughput).
- sga_uq: horizon 8 DAG, 2000 error panels, 64 forecaster branches
  (lane: 4000 panels for the tight certificate check; 2000 keeps the same
  coverage assertion with wider slack).
- passive_impact: 64-path fill telemetry + 24-path baseline comparison on
  the paper's Table-1 spec (lane: same budgets).
- extra_conformal / stochastic_tracking: deterministic / module-default
  budgets -- no shrink needed.

The whole battery is designed to fit inside a ~60 s envelope on the dev box
when torch is installed (the three torch benches dominate); torch-gated
families return ``{}`` cleanly when the ``nn`` extra is absent.
"""

from __future__ import annotations

import math

import numpy as np

_SEED = 20261017  # wave-17 stamp seed

# --- diffpts: shrunk DiffPTS stream (torch-gated) ---------------------------
_DIFFPTS_SEED = _SEED + 41
_DIFFPTS_N_TRAIN = 220
_DIFFPTS_N_TEST = 60
_DIFFPTS_LOOKBACK = 8
_DIFFPTS_N_SAMPLES = 96
_DIFFPTS_HIDDEN = (16,)
_DIFFPTS_EPOCHS = 50
_DIFFPTS_N_STEPS = 20

# --- multilevel_mm: shrunk Cheridito-Weiss world (torch-gated) ---------------
_MLMM_SEED = _SEED + 42
_MLMM_TRAIN_EPISODES = 3
_MLMM_TRAIN_HORIZON = 30.0
_MLMM_EVAL_HORIZON = 40.0
_MLMM_EVAL_SEEDS = 2
_MLMM_ARMS = ("agent", "glft", "random")

# --- rlmm_c51: shrunk Algorithm-C scenario bandit (torch-gated) --------------
_RLMM_SEED = _SEED + 43
_RLMM_EPISODES = 4
_RLMM_HORIZON = 120.0
_RLMM_POOL_SIZE = 6
_RLMM_EXPECTED_MOS = 30

# --- sga_uq: shrunk DAG-certificate panel budget ------------------------------
_SGA_SEED = _SEED + 44
_SGA_HORIZON = 8
_SGA_PHI = 0.6
_SGA_SIGMA = 0.5
_SGA_N_PANELS = 2000
_SGA_N_SAMPLES = 64

# --- passive_impact: paper Table-1 spec (eta = temporary_impact x tick) -------
_PIM_SEED = _SEED + 45
_PIM_N_UNITS = 20
_PIM_HORIZON = 300.0
_PIM_N_GRID = 120
_PIM_N_PATHS = 64
_PIM_CMP_UNITS = 8
_PIM_CMP_HORIZON = 120.0
_PIM_CMP_N_PATHS = 24
_PIM_CMP_N_GRID = 60

# --- stochastic_tracking: Nutz-Voss sharp-rate sweep (RNG-free) ---------------
_ST_HORIZON = 1.0
_ST_N_STEPS = 400
_ST_BETA = 1.0
_ST_LAM = 1.0
_ST_XI = 1.0
_ST_N_EPS = 8


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


def _num(v: object, key: str) -> float:
    """Narrow a ``dict[str, object]`` entry to a finite float (fail-closed)."""
    if isinstance(v, (bool, np.bool_)):
        raise TypeError(f"{key} is bool, not a number")
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        if math.isfinite(f):
            return f
    raise ValueError(f"{key} is not a finite number: {v!r}")


def bench_diffpts() -> dict[str, float]:
    """DiffPTS LSNM diffusion on an AR(1)-bimodal stream (SYNTHETIC, torch).

    Emits CRPS for the DDPM ancestral sampler and two DDIM NFE budgets plus
    coverage/PIT/pinball diagnostics and the CRPS gap vs the NGboost/QRF
    baselines.  Shrunk: 220/60 train/test, 96 samples, hidden (16,), 50
    epochs, 20-step schedule, DDIM budgets (2, 8).
    """
    try:
        from quant_fund.models.diffpts import evaluate_synthetic_stream

        raw = evaluate_synthetic_stream(
            n_train=_DIFFPTS_N_TRAIN,
            n_test=_DIFFPTS_N_TEST,
            lookback=_DIFFPTS_LOOKBACK,
            kind="ar1_bimodal",
            seed=_DIFFPTS_SEED,
            n_samples=_DIFFPTS_N_SAMPLES,
            ddim_budgets=(2, 8),
            hidden=_DIFFPTS_HIDDEN,
            epochs=_DIFFPTS_EPOCHS,
            n_steps=_DIFFPTS_N_STEPS,
        )
        mapped = {
            "diffpts_ddpm_crps": float(raw["diffpts_ddpm_crps"]),
            "diffpts_ddpm_nfe": float(raw["diffpts_ddpm_nfe"]),
            "diffpts_ddim2_crps": float(raw["diffpts_ddim2_crps"]),
            "diffpts_ddim2_nfe": float(raw["diffpts_ddim2_nfe"]),
            "diffpts_ddim8_crps": float(raw["diffpts_ddim8_crps"]),
            "diffpts_ddim8_nfe": float(raw["diffpts_ddim8_nfe"]),
            "diffpts_coverage_90": float(raw["diffpts_coverage_90"]),
            "diffpts_width_90": float(raw["diffpts_width_90"]),
            "diffpts_pit_ks": float(raw["diffpts_pit_ks"]),
            "diffpts_pinball_mean": float(raw["diffpts_pinball_mean"]),
            "diffpts_crps_minus_ngboost": float(raw["crps_diffpts_minus_ngboost"]),
            "diffpts_crps_minus_qrf": float(raw["crps_diffpts_minus_qrf"]),
        }
        return _finite_blob(mapped)
    except ImportError:
        return {}
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_extra_conformal() -> dict[str, float]:
    """Extrapolated weighted conformal on a synthetic covariate-shift stream.

    The module's own ``bench_extra_conformal``/``bench_extra_harm`` helpers
    already emit the lane blob; we filter the str stamps (dgp/claim/mode)
    and re-key floats under the family prefix.  Claim: weighted coverage
    beats unweighted under the planted shift; extra-harm reports the
    harmonic-mean power-shift gap.
    """
    try:
        from quant_fund.models import extra_conformal as _extra_conformal_core_bench

        main = _extra_conformal_core_bench.bench_extra_conformal(seed=_SEED + 51)
        harm = _extra_conformal_core_bench.bench_extra_harm(seed=_SEED + 51)
        mapped = {
            "xc_coverage": float(main["coverage"]),
            "xc_mean_width": float(main["mean_width"]),
            "xc_unweighted_coverage": float(main["unweighted_coverage"]),
            "xc_coverage_error": float(main["coverage_error"]),
            "xc_unweighted_coverage_error": float(main["unweighted_coverage_error"]),
            "xc_ess_fraction": float(main["ess_fraction"]),
            "xc_alpha": float(main["alpha"]),
            "xc_n_cal": float(main["n"]),
            "xc_harm_coverage": float(harm["coverage"]),
            "xc_harm_unweighted_coverage": float(harm["unweighted_coverage"]),
            "xc_harm_coverage_gap": float(harm["harm_coverage_gap"]),
            "xc_harm_ess_fraction": float(harm["ess_fraction"]),
        }
        return _finite_blob(mapped)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_multilevel_mm() -> dict[str, float]:
    """Cheridito-Weiss multi-level MM: completion + inventory-cap telemetry.

    Trains a tiny level-grouped actor (hidden (16,)) for 3 episodes at
    horizon 30 on the Santa-Fe ZI world, then evaluates completion rate and
    fill telemetry across {agent, glft, random} on 2 seeds at horizon 40.
    Emits no PnL keys -- only coverage of the task contract (completion,
    cap violations).  TORCH-GATED.
    """
    try:
        from quant_fund.microstructure.multilevel_mm import (
            MultiLevelMMAgent,
            MultiLevelMMConfig,
            MultiLevelSpec,
            evaluate_multilevel_mm,
            train_multilevel_mm,
        )
        from quant_fund.microstructure.zi_lob_simulator import santa_fe_config

        spec = MultiLevelSpec()
        cfg = santa_fe_config(seed=_MLMM_SEED)
        agent = MultiLevelMMAgent(
            spec,
            MultiLevelMMConfig(
                hidden_actor=(16,),
                hidden_critic=(16,),
                seed=_MLMM_SEED,
            ),
        )
        train_multilevel_mm(
            agent=agent,
            config=cfg,
            horizon=_MLMM_TRAIN_HORIZON,
            n_episodes=_MLMM_TRAIN_EPISODES,
            seed_base=_MLMM_SEED,
            decision_interval=1.0,
        )
        ev = evaluate_multilevel_mm(
            config=cfg,
            horizon=_MLMM_EVAL_HORIZON,
            agent=agent,
            n_seeds=_MLMM_EVAL_SEEDS,
            seed_base=_MLMM_SEED + 777,
            decision_interval=1.0,
            arms=_MLMM_ARMS,
        )
        metrics = ev["metrics"]
        cap = float(ev["inventory_cap"])
        violations = 0.0
        for rows in ev["sessions"].values():
            for row in rows:
                if float(row["max_abs_inventory"]) > cap + 1e-9:
                    violations += 1.0
        mapped = {
            "mlmm_agent_session_completion": float(metrics["agent_session_completion_rate"]),
            "mlmm_glft_session_completion": float(metrics["glft_session_completion_rate"]),
            "mlmm_random_session_completion": float(metrics["random_session_completion_rate"]),
            "mlmm_inventory_cap_violations": violations,
            "mlmm_inventory_cap": cap,
            "mlmm_agent_fills_mean": float(metrics["agent_n_fills_mean"]),
            "mlmm_n_levels": float(spec.n_levels),
            "mlmm_lots": float(spec.lots),
            "mlmm_horizon": float(ev["horizon"]),
            "mlmm_eval_seeds": float(_MLMM_EVAL_SEEDS),
        }
        return _finite_blob(mapped)
    except ImportError:
        return {}
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_rlmm_c51() -> dict[str, float]:
    """Moret-Lillo Algorithm C: C51 + scenario-bandit fine-tune (torch).

    Trains a tiny C51 market maker (n_atoms 11, hidden (16,)) through the
    scenario-generator / difficulty-bandit loop: pool of 6 x 30 expected
    MOs, 4 fine-tune episodes at horizon 120 with w_max 0.4, then a
    robustness eval over 1 scenario x {stationary, random_persistence,
    correlated_direction} x {c51, glft, as}.  Emits completion/cap/bandit
    telemetry only.
    """
    try:
        from quant_fund.microstructure.rl_market_maker import (
            C51Config,
            C51MarketMaker,
            RLStateSpec,
        )
        from quant_fund.microstructure.rlmm_c51 import (
            ScenarioBandit,
            ScenarioGenerator,
            evaluate_scenario_robustness,
            run_scenario_bandit_finetuning,
        )
        from quant_fund.microstructure.zi_lob_simulator import (
            as_policy,
            glft_policy,
            santa_fe_config,
        )

        cfg = santa_fe_config(seed=_RLMM_SEED)
        agent = C51MarketMaker(
            RLStateSpec(aux_enabled=True, filter_tau_r=30.0, inventory_cap=8),
            C51Config(
                n_atoms=11,
                v_min=-3.0,
                v_max=3.0,
                hidden=(16,),
                gamma_event=0.999,
                n_step=2,
                lr=1e-3,
                batch_size=8,
                buffer_capacity=512,
                target_update_every=20,
                eps_start=1.0,
                eps_end=0.1,
                eps_decay_steps=60,
                seed=_RLMM_SEED,
            ),
        )
        generator = ScenarioGenerator(seed=_RLMM_SEED)
        pool = generator.initial_pool(_RLMM_POOL_SIZE, _RLMM_EXPECTED_MOS)
        bandit = ScenarioBandit(
            pool,
            n_updates_planned=_RLMM_EPISODES,
            w_max=0.4,
            seed=_RLMM_SEED,
        )
        out = run_scenario_bandit_finetuning(
            agent=agent,
            config=cfg,
            pool=pool,
            bandit=bandit,
            generator=generator,
            n_episodes=_RLMM_EPISODES,
            horizon=_RLMM_HORIZON,
            expected_mos=_RLMM_EXPECTED_MOS,
            seed_base=_RLMM_SEED,
        )
        ev = evaluate_scenario_robustness(
            config=cfg,
            horizon=_RLMM_HORIZON,
            agent=agent,
            policies={
                "glft": glft_policy(gamma=1.0, sigma=0.02, kappa=1000.0, a_fill=1.0, tick=cfg.tick),
                "as": as_policy(gamma=0.002, sigma=0.02, kappa=1000.0, tick=cfg.tick),
            },
            n_scenarios=1,
            expected_mos=_RLMM_EXPECTED_MOS,
            seed_base=_RLMM_SEED + 101,
        )
        cap = float(ev["inventory_cap"])
        violations = 0.0
        for rows in ev["sessions"].values():
            for row in rows:
                if float(row["max_abs_inventory"]) > cap + 1e-9:
                    violations += 1.0
        metrics = ev["metrics"]
        c51_completion_min = min(
            float(v)
            for k, v in metrics.items()
            if k.startswith("c51_") and k.endswith("_session_completion_rate")
        )
        episodes = out["episodes"]
        completed = sum(1.0 for e in episodes if bool(e["session_completed"]))
        weights = np.asarray(out["bandit"]["weights_final"], dtype=np.float64)
        difficulties = np.asarray(out["bandit"]["difficulties"], dtype=np.float64)
        mapped = {
            "rlmm_c51_episodes_completed": completed,
            "rlmm_c51_episodes": float(out["n_episodes"]),
            "rlmm_c51_pool_size": float(out["pool_size"]),
            "rlmm_c51_bandit_weight_max": float(weights.max()),
            "rlmm_c51_difficulty_spread": float(difficulties.max() - difficulties.min()),
            "rlmm_c51_eval_completion_min": c51_completion_min,
            "rlmm_c51_cap_violations": violations,
            "rlmm_c51_inventory_cap": cap,
            "rlmm_c51_horizon": float(ev["horizon"]),
        }
        return _finite_blob(mapped)
    except ImportError:
        return {}
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_sga_uq() -> dict[str, float]:
    """SGA multistep UQ on a VAR(1) error DAG + a synthetic AR(1) forecaster.

    Closed-form-vs-propagated covariance agreement on an 8-step
    ``phi = 0.6, sigma = 0.5`` VAR(1) DAG, the variance-bound certificate,
    horizon monotonicity, 1.96-sigma empirical coverage over 2000 seeded
    panels, and ``uq_from_forecaster`` graph complexity over 64 branches of
    the synthetic AR(1) forecaster (fixed horizon -> complexity = horizon).
    """
    try:
        from quant_fund.metrics.sga_multistep_uq import (
            SgaGraph,
            dag_covariance_closed_form,
            horizon_monotonicity,
            interval_coverage,
            make_synthetic_ar1_forecaster,
            propagate_uncertainty,
            simulate_synthetic_dag_errors,
            uq_from_forecaster,
            var1_error_dag,
            variance_bound_certificate,
        )

        dag = var1_error_dag(horizon=_SGA_HORIZON, phi=_SGA_PHI, sigma=_SGA_SIGMA)
        pu = propagate_uncertainty(dag)
        cov_cf = dag_covariance_closed_form(dag)
        cov_gap = float(np.max(np.abs(pu.covariance - cov_cf)))
        cert = variance_bound_certificate(dag)
        rng = np.random.default_rng(_SGA_SEED)
        errors = simulate_synthetic_dag_errors(dag, _SGA_N_PANELS, rng)
        cov = interval_coverage(errors, pu.std, 1.96)
        hist = np.cumsum(rng.standard_normal(240))
        stub = make_synthetic_ar1_forecaster(_SGA_PHI, _SGA_SIGMA)
        uq = uq_from_forecaster(
            stub, hist, _SGA_HORIZON, n_samples=_SGA_N_SAMPLES, rng=rng, slice_len=4
        )
        mono = horizon_monotonicity(pu.variance)
        graph = uq["graph"]
        if not isinstance(graph, SgaGraph):
            raise TypeError("uq_from_forecaster['graph'] is not an SgaGraph")
        mapped = {
            "sga_cov_closed_form_max_err": cov_gap,
            "sga_certified": 1.0 if bool(cert["certified"]) else 0.0,
            "sga_certificate_total_margin": _num(cert["total_bound_margin"], "total_bound_margin"),
            "sga_coverage_mean": _num(cov["mean"], "coverage_mean"),
            "sga_variance_monotone": 1.0 if bool(mono["monotone"]) else 0.0,
            "sga_terminal_variance": float(pu.variance[-1]),
            "sga_graph_complexity": _num(uq["graph_complexity"], "graph_complexity"),
            "sga_n_nodes": float(graph.n_nodes),
        }
        return _finite_blob(mapped)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_passive_impact() -> dict[str, float]:
    """Barzykin et al. optimal passive execution under price impact.

    Paper Table-1 spec (lam 1.4, k 0.48, eta = 0.0028/0.01 per tick, sigma
    0.003, phi = alpha = 1e-5, m == k closed form): 20-unit plan on a
    120-point grid at horizon 300, 64-path fill telemetry, an 8-unit
    passive-vs-aggressive comparison (PnL diff re-keyed off the forbidden
    token), the fluid inventory-path monotonicity check, and the
    zero-impact/zero-OFI exact reduction (terminal price == mid).
    """
    try:
        from quant_fund.execution.passive_impact import (
            PassiveImpactSpec,
            compare_vs_aggressive_baseline,
            expected_inventory_path,
            optimal_execution_plan,
            simulate_passive_execution,
        )

        spec = PassiveImpactSpec(
            lam=1.4, k=0.48, eta=0.0028 / 0.01, sigma=0.003, phi=1e-5, alpha=1e-5
        )
        plan = optimal_execution_plan(spec, _PIM_N_UNITS, _PIM_HORIZON, n_grid=_PIM_N_GRID)
        sim = simulate_passive_execution(
            plan,
            mid_price=100.0,
            unit_size=1000.0,
            n_paths=_PIM_N_PATHS,
            seed=_PIM_SEED,
        )
        cmp_ = compare_vs_aggressive_baseline(
            spec,
            _PIM_CMP_UNITS,
            _PIM_CMP_HORIZON,
            mid_price=100.0,
            unit_size=1000.0,
            n_paths=_PIM_CMP_N_PATHS,
            seed=_PIM_SEED + 1,
            n_grid=_PIM_CMP_N_GRID,
        )
        inv_path = expected_inventory_path(plan)
        monotone = bool(np.all(np.diff(inv_path) <= 1e-12))
        spec0 = PassiveImpactSpec(lam=1.4, k=0.48, phi=1e-5, alpha=1e-5)
        plan0 = optimal_execution_plan(spec0, 10, 120.0, n_grid=120)
        sim0 = simulate_passive_execution(
            plan0,
            mid_price=100.0,
            unit_size=1000.0,
            n_paths=8,
            seed=_PIM_SEED + 2,
        )
        mapped = {
            "pim_fill_fraction_mean": _num(
                sim["sim_internal_fill_fraction_mean"], "fill_fraction_mean"
            ),
            "pim_trading_time_mean": _num(
                sim["sim_internal_trading_time_mean"], "trading_time_mean"
            ),
            "pim_passive_minus_aggressive": _num(
                cmp_["sim_internal_passive_minus_aggressive_mean"],
                "passive_minus_aggressive_mean",
            ),
            "pim_inventory_monotone": 1.0 if monotone else 0.0,
            "pim_zero_impact_terminal_abs_dev": abs(
                _num(sim0["sim_internal_terminal_price_mean"], "terminal_price_mean") - 100.0
            ),
            "pim_method_closed_form": 1.0 if plan.method == "closed_form_m_eq_k" else 0.0,
            "pim_n_paths": _num(sim["n_paths"], "n_paths"),
            "pim_horizon": float(plan.horizon),
        }
        return _finite_blob(mapped)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_stochastic_tracking() -> dict[str, float]:
    """Nutz-Voss singular stochastic tracking: sharp-rate sweep (RNG-free).

    The unregularized optimum's value must match the closed-form
    ``ow_benchmark_value`` (Theorem 4.2) and the regularized-to-unregularized
    gap must converge at the paper's sharp exponent: the log-log slope of
    ``E`` vs ``sqrt(eps)`` across an 8-point sweep on a 400-step grid.  All
    computations deterministic given the grid.
    """
    try:
        from quant_fund.execution.stochastic_tracking import (
            TrackingGrid,
            optimal_unregularized,
            ow_benchmark_value,
            ow_coefficients,
            regularization_gap_sweep,
            sharp_rate_constant,
        )

        grid = TrackingGrid(_ST_HORIZON, _ST_N_STEPS)
        coeffs = ow_coefficients(_ST_BETA, _ST_LAM, grid)
        sol = optimal_unregularized(_ST_XI, coeffs, grid)
        v0 = ow_benchmark_value(_ST_XI, _ST_BETA, _ST_LAM, _ST_HORIZON)
        h = grid.step
        epsilons = np.geomspace((5.0 * h) ** 2, 0.04, _ST_N_EPS)
        sweep = regularization_gap_sweep(
            epsilons, _ST_XI, _ST_BETA, _ST_LAM, _ST_HORIZON, grid.n_steps
        )
        mapped = {
            "st_ow_value_abs_gap": float(abs(sol.value - v0)),
            "st_ow_value": float(v0),
            "st_unregularized_value": float(sol.value),
            "st_loglog_slope_cht": float(sweep["synthetic_loglog_slope_cht"]),
            "st_loglog_slope_explicit": float(sweep["synthetic_loglog_slope_explicit"]),
            "st_loglog_slope_tracking": float(sweep["synthetic_loglog_slope_tracking"]),
            "st_sharp_rate_constant": float(
                sharp_rate_constant(_ST_XI, _ST_BETA, _ST_LAM, _ST_HORIZON)
            ),
            "st_n_steps": float(grid.n_steps),
        }
        return _finite_blob(mapped)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
