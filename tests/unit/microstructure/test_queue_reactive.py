"""Queue-reactive CTMC: MLE rates, stationary law, V(q), fail-closed."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.queue_reactive import (
    QueueTrajectory,
    RateEstimate,
    bench_queue_reactive,
    estimate_rates,
    joint_stationary,
    queue_value_curve,
    reference_rates,
    simulate_queue,
    stationary_birth_death,
    stationary_simulated,
    tv_distance,
)


def _traj(seed: int = 0, horizon: float = 50_000.0, q_max: int = 12):
    l_l, l_c, l_m = reference_rates(q_max, lam=1.0)
    return simulate_queue(l_l, l_c, l_m, horizon=horizon, seed=seed)


class TestSimulation:
    def test_states_bounded(self):
        t = _traj(seed=1)
        assert t.states.min() >= 0
        assert t.states.max() <= t.q_max

    def test_times_increasing(self):
        t = _traj(seed=2)
        assert np.all(np.diff(t.times) > 0)
        assert t.times[-1] < 50_000.0 + 1e-6

    def test_event_codes_valid(self):
        t = _traj(seed=3)
        assert set(np.unique(t.events)) <= {0, 1, 2}

    def test_deterministic_seed(self):
        a, b = _traj(seed=4), _traj(seed=4)
        np.testing.assert_array_equal(a.states, b.states)
        np.testing.assert_allclose(a.times, b.times)

    def test_different_seed_differs(self):
        a, b = _traj(seed=5), _traj(seed=6)
        assert not np.array_equal(a.states, b.states)

    def test_fail_closed_bad_rates(self):
        l_l, l_c, l_m = reference_rates(6)
        with pytest.raises(ValueError):
            simulate_queue(l_l[:3], l_c, l_m, 100.0, seed=0)
        with pytest.raises(ValueError):
            simulate_queue(-l_l, l_c, l_m, 100.0, seed=0)
        with pytest.raises(ValueError):
            simulate_queue(l_l, l_c, l_m, -1.0, seed=0)
        with pytest.raises(ValueError):
            simulate_queue(l_l, l_c, l_m, 100.0, seed=0, q0=99)


class TestRateEstimation:
    def test_recovery_declining(self):
        q_max = 10
        l_l, l_c, l_m = reference_rates(q_max, lam=1.0)
        traj = simulate_queue(l_l, l_c, l_m, horizon=200_000.0, seed=7)
        est = estimate_rates(traj)
        mask = est.dwell > 2000.0
        mask[-1] = False  # births at q_max are zero-rate by reflection
        rel = np.abs(est.lam_l[mask] - l_l[mask]) / l_l[mask]
        assert rel.mean() < 0.15

    def test_recovery_flat(self):
        q_max = 8
        l_l, l_c, l_m = reference_rates(q_max, lam=1.0, shape="flat")
        traj = simulate_queue(l_l, l_c, l_m, horizon=100_000.0, seed=8)
        est = estimate_rates(traj)
        mask = est.dwell > 1000.0
        mask[-1] = False  # births at q_max are zero-rate by reflection
        assert np.allclose(est.lam_l[mask], 1.0, atol=0.15)

    def test_counts_sum(self):
        t = _traj(seed=9)
        est = estimate_rates(t)
        assert est.counts.sum() == t.n_events

    def test_fail_closed_short(self):
        t = QueueTrajectory(
            times=np.array([1.0, 2.0]),
            states=np.array([1, 2]),
            events=np.array([0, 0]),
            q_max=5,
        )
        with pytest.raises(ValueError):
            estimate_rates(t)


class TestStationary:
    def test_product_formula_normalized(self):
        l_l, l_c, l_m = reference_rates(10)
        est = RateEstimate(
            lam_l=l_l,
            lam_c=l_c,
            lam_m=l_m,
            dwell=np.ones(11),
            counts=np.ones((11, 3)),
        )
        pi = stationary_birth_death(est)
        assert abs(pi.sum() - 1.0) < 1e-12
        assert (pi >= 0).all()

    def test_flat_rates_shape(self):
        l_l, l_c, l_m = reference_rates(8, shape="flat")
        est = RateEstimate(
            lam_l=l_l,
            lam_c=l_c,
            lam_m=l_m,
            dwell=np.ones(9),
            counts=np.ones((9, 3)),
        )
        pi = stationary_birth_death(est)
        # birth/death ratio 1/0.45 > 1 → mass concentrates at high q
        assert np.argmax(pi) == 8

    def test_analytic_vs_simulated(self):
        q_max = 10
        l_l, l_c, l_m = reference_rates(q_max, lam=1.0)
        traj = simulate_queue(l_l, l_c, l_m, horizon=200_000.0, seed=10)
        pi_a = stationary_birth_death(
            RateEstimate(
                lam_l=l_l, lam_c=l_c, lam_m=l_m, dwell=np.ones(11), counts=np.ones((11, 3))
            )
        )
        pi_s = stationary_simulated(traj)
        assert tv_distance(pi_a, pi_s) < 0.15

    def test_fail_closed_zero_death(self):
        est = RateEstimate(
            lam_l=np.ones(4),
            lam_c=np.zeros(4),
            lam_m=np.zeros(4),
            dwell=np.ones(4),
            counts=np.ones((4, 3)),
        )
        with pytest.raises(ValueError):
            stationary_birth_death(est)


class TestTV:
    def test_identical_zero(self):
        p = np.array([0.5, 0.3, 0.2])
        assert tv_distance(p, p) == 0.0

    def test_disjoint_one(self):
        assert tv_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0

    def test_unnormalized_ok(self):
        assert tv_distance(np.array([2.0, 0.0]), np.array([0.0, 2.0])) == 1.0

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            tv_distance(np.array([1.0]), np.array([1.0, 0.0]))
        with pytest.raises(ValueError):
            tv_distance(np.array([-1.0, 2.0]), np.array([0.5, 0.5]))


class TestQueueValue:
    def test_monotone_decreasing(self):
        l_l, l_c, l_m = reference_rates(10)
        est = RateEstimate(
            lam_l=l_l,
            lam_c=l_c,
            lam_m=l_m,
            dwell=np.ones(11),
            counts=np.ones((11, 3)),
        )
        v = queue_value_curve(est)
        assert np.all(np.diff(v) <= 1e-9)
        assert v[0] > v[-1]

    def test_shape(self):
        l_l, l_c, l_m = reference_rates(6)
        est = RateEstimate(
            lam_l=l_l,
            lam_c=l_c,
            lam_m=l_m,
            dwell=np.ones(7),
            counts=np.ones((7, 3)),
        )
        assert queue_value_curve(est).shape == (7,)


class TestJointStationary:
    def test_normalized_nonneg(self):
        l_l, l_c, l_m = reference_rates(6)
        est = RateEstimate(
            lam_l=l_l,
            lam_c=l_c,
            lam_m=l_m,
            dwell=np.ones(7),
            counts=np.ones((7, 3)),
        )
        pj = joint_stationary(est)
        assert pj.shape == (7, 7)
        assert abs(pj.sum() - 1.0) < 1e-9
        assert (pj >= 0).all()

    def test_fail_closed_large(self):
        n = 70
        est = RateEstimate(
            lam_l=np.ones(n),
            lam_c=np.ones(n),
            lam_m=np.ones(n),
            dwell=np.ones(n),
            counts=np.ones((n, 3)),
        )
        with pytest.raises(ValueError):
            joint_stationary(est)


class TestReferenceRates:
    def test_declining_shape(self):
        l_l, _, _ = reference_rates(10, shape="declining")
        assert np.all(np.diff(l_l) < 0)

    def test_flat_shape(self):
        l_l, l_c, l_m = reference_rates(10, shape="flat")
        assert np.all(l_l == l_l[0])

    def test_unknown_shape(self):
        with pytest.raises(ValueError):
            reference_rates(5, shape="nope")


class TestBench:
    def test_keys_finite(self):
        out = bench_queue_reactive(20260131)
        for k, v in out.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float)
            assert np.isfinite(v), k

    def test_deterministic(self):
        assert bench_queue_reactive(20260131) == bench_queue_reactive(20260131)

    def test_quality_thresholds(self):
        out = bench_queue_reactive(20260131)
        assert out["synthetic_stationary_tv"] < 0.2
        assert out["synthetic_xshift_monotonicity_violations"] == 0.0
        assert out["synthetic_determinism"] == 1.0
