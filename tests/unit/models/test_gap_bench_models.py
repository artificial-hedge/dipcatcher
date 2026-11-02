"""Unit tests for the untested ``quant_fund.models`` utility and bench modules.

Each of these modules carries a ``bench_*`` function that returns a dict of
0/1 flags. Those flags are only meaningful if the underlying maths is right,
so the tests here pin the *maths* directly — a Galois-field identity, a group
order formula, a chain-complex square that must vanish, a Lyapunov function
that must decay — rather than only asserting that the bench returns ones.

Every module here is SYNTHETIC. The flags are correctness results, not
market evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models import (
    batch_od,
    boundary_sq,
    dist_obsv,
    euler_char,
    gain_schedule,
    galois_field,
    lyap_synth,
    matrix_grp,
    odom_comp,
    pot_field,
    ramsey_bound,
    rbc_sim,
    rips_h1,
    token_ring,
)

# --------------------------------------------------------------------------- #
# galois_field
# --------------------------------------------------------------------------- #


def test_a_primitive_element_generates_every_nonzero_residue() -> None:
    assert galois_field.is_primitive(3, 7) is True
    assert galois_field.is_primitive(2, 7) is False


def test_the_discrete_log_inverts_exponentiation() -> None:
    for k in range(10):
        assert galois_field.dlog(2, pow(2, k, 11), 11) == k


def test_the_discrete_log_of_an_unreachable_element_is_minus_one() -> None:
    """4 has order 5 mod 11, so its powers never reach 2."""
    assert galois_field.dlog(4, 2, 11) == -1
    assert galois_field.is_primitive(4, 11) is False


def test_the_field_axioms_hold_for_a_small_prime() -> None:
    assert galois_field._field_axioms(7) is True


def test_the_galois_field_bench_passes() -> None:
    assert galois_field.bench_galois_field()["synthetic_gf_axioms"] == 1.0


# --------------------------------------------------------------------------- #
# matrix_grp
# --------------------------------------------------------------------------- #


def test_the_determinant_is_taken_modulo_the_prime() -> None:
    m = np.array([[2.0, 1.0], [1.0, 1.0]])
    assert matrix_grp._det2(m, 3) == 1


def test_the_group_orders_match_the_closed_form() -> None:
    p = 3
    gl, sl = matrix_grp.gl_order_enum(p)
    want_gl = (p**2 - 1) * (p**2 - p)
    assert gl == want_gl
    assert sl == want_gl // (p - 1)


def test_the_special_linear_group_is_closed_under_multiplication() -> None:
    assert matrix_grp._sl_closed(3) is True


def test_the_matrix_group_bench_passes() -> None:
    assert matrix_grp.bench_matrix_grp()["synthetic_matrix_grp"] == 1.0


# --------------------------------------------------------------------------- #
# ramsey_bound
# --------------------------------------------------------------------------- #


def test_the_five_cycle_colouring_has_no_monochromatic_triangle() -> None:
    c5 = [[0] * 5 for _ in range(5)]
    for i in range(5):
        c5[i][(i + 1) % 5] = c5[(i + 1) % 5][i] = 1
    assert ramsey_bound._has_mono(5, c5) is False


def test_a_monochromatic_triangle_is_detected() -> None:
    color = [[0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i + 1, 3):
            color[i][j] = color[j][i] = 1
    assert ramsey_bound._has_mono(3, color) is True


def test_the_ramsey_bound_bench_passes() -> None:
    assert ramsey_bound.bench_ramsey_bound()["synthetic_ramsey"] == 1.0


# --------------------------------------------------------------------------- #
# boundary_sq / euler_char / rips_h1 — complexes
# --------------------------------------------------------------------------- #


def test_the_boundary_matrix_has_one_row_per_target_face() -> None:
    faces = [(0, 1), (1, 2)]
    codom = [(0,), (1,), (2,)]
    m = boundary_sq._boundary(faces, codom)
    assert m.shape == (3, 2)
    # An edge has two endpoints, so each column has exactly two ones.
    assert m.sum(axis=0).tolist() == [2, 2]


def test_the_boundary_of_a_boundary_is_zero_modulo_two() -> None:
    edges = [(0, 1), (0, 2), (1, 2)]
    verts = [(0,), (1,), (2,)]
    tris = [(0, 1, 2)]
    d1 = boundary_sq._boundary(edges, verts)
    d2 = boundary_sq._boundary(tris, edges)
    assert np.all((d1 @ d2) % 2 == 0)


def test_the_boundary_square_bench_passes() -> None:
    assert boundary_sq.bench_boundary_sq()["synthetic_boundary_sq"] == 1.0


def test_the_euler_characteristic_bench_passes() -> None:
    assert euler_char.bench_euler_char()["synthetic_euler_betti"] == 1.0


def test_a_ring_has_a_persistent_h1_class() -> None:
    theta = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    ring = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    assert rips_h1._vr_betti1(ring, 0.8) >= 1


def test_the_rips_bench_reports_both_cells() -> None:
    result = rips_h1.bench_rips_h1()
    assert result["synthetic_rips_ring"] == 1.0
    assert result["synthetic_rips_disk"] == 1.0


# --------------------------------------------------------------------------- #
# token_ring
# --------------------------------------------------------------------------- #


def test_the_token_ring_only_transmits_from_the_holder() -> None:
    rng = np.random.RandomState(0)
    senders = token_ring.ring_run(6, 12, rng)
    assert len(senders) == 12
    assert all(0 <= sender < 6 for sender in senders)


def test_the_token_ring_is_deterministic_for_a_seed() -> None:
    first = token_ring.ring_run(6, 12, np.random.RandomState(3))
    second = token_ring.ring_run(6, 12, np.random.RandomState(3))
    assert first == second


def test_the_token_ring_bench_passes() -> None:
    assert token_ring.bench_token_ring()["synthetic_ring_safety"] == 1.0


# --------------------------------------------------------------------------- #
# lyap_synth / pot_field / odom_comp / dist_obsv — control
# --------------------------------------------------------------------------- #


def test_the_correct_control_law_decays_the_lyapunov_function() -> None:
    values = lyap_synth._run(0, correct_sign=True)
    assert values[-1] < values[0]


def test_the_wrong_sign_control_law_does_not_decay() -> None:
    """The antisymmetric controller fails to decrease V."""
    good = lyap_synth._run(0, correct_sign=True)
    bad = lyap_synth._run(0, correct_sign=False)
    assert good[-1] < bad[-1]


def test_the_lyapunov_bench_passes() -> None:
    result = lyap_synth.bench_lyap_synth()
    assert result["synthetic_lyap_decay"] == 1.0
    assert result["synthetic_lyap_order"] == 1.0


def test_the_potential_field_reaches_the_goal() -> None:
    assert pot_field._run(0, repulse=True) == pytest.approx(0.0)


def test_the_potential_field_bench_passes() -> None:
    result = pot_field.bench_pot_field()
    assert result["synthetic_pot_reach"] == 1.0
    assert result["synthetic_pot_better"] == 1.0


def test_the_fused_heading_beats_the_drifting_gyro() -> None:
    drift = odom_comp._run(0, w=None)
    fused = odom_comp._run(0, w=0.98)
    assert fused < drift


def test_the_odometry_bench_passes() -> None:
    result = odom_comp.bench_odom_comp()
    assert result["synthetic_odom_fuse"] == 1.0
    assert result["synthetic_odom_bound"] == 1.0


def test_the_disturbance_observer_removes_the_steady_state_offset() -> None:
    plain, composite = dist_obsv._run(0)
    assert composite < plain


def test_the_disturbance_observer_bench_passes() -> None:
    result = dist_obsv.bench_dist_obsv()
    assert result["synthetic_dist_obsv_reject"] == 1.0
    assert result["synthetic_dist_obsv_track"] == 1.0


# --------------------------------------------------------------------------- #
# rbc_sim / gain_schedule
# --------------------------------------------------------------------------- #


def test_the_rbc_run_returns_one_observation_per_step() -> None:
    k, a = rbc_sim.rbc_run(100, 0.9, seed=0)
    assert k.shape == (100,)
    assert a.shape == (100,)
    assert np.all(np.isfinite(k))


def test_the_rbc_capital_comoves_with_the_lagged_productivity() -> None:
    k, a = rbc_sim.rbc_run(800, 0.95, seed=0)
    best = max(
        float(np.corrcoef(k[100 + lag :], a[100 : len(a) - lag])[0, 1]) for lag in range(1, 12)
    )
    assert best > 0.8


def test_the_rbc_bench_passes() -> None:
    result = rbc_sim.bench_rbc_sim()
    assert result["synthetic_rbc_comove"] == 1.0
    assert result["synthetic_rbc_stable"] == 1.0


def test_the_gain_scheduled_controller_tracks_both_setpoints() -> None:
    track = gain_schedule.gsc_run(300)
    assert track.shape == (300,)
    assert abs(track[149] - 1.0) < 0.15
    assert abs(track[-1] - 3.0) < 0.3


def test_the_gain_schedule_bench_passes() -> None:
    assert gain_schedule.bench_gain_schedule()["synthetic_gsc_tracks"] == 1.0


# --------------------------------------------------------------------------- #
# batch_od
# --------------------------------------------------------------------------- #


def test_the_batch_estimator_recovers_the_epoch_state() -> None:
    """Gauss-Newton on the range residuals must contract the initial error."""
    rng = np.random.default_rng(0)
    rng.random()
    mu = batch_od._MU
    r0 = np.asarray([7000.0, 0.0, 0.0])
    v0 = np.asarray([0.0, np.sqrt(mu / 7000.0) * np.cos(0.4), np.sqrt(mu / 7000.0) * np.sin(0.4)])
    truth = np.concatenate([r0, v0])
    site = np.asarray([6378.0, 0.0, 0.0])
    times = np.linspace(-1500.0, 1500.0, 7)
    ranges = np.asarray(
        [
            float(
                np.linalg.norm(
                    batch_od._prop_full(
                        np.concatenate([truth, np.eye(6).ravel()]), t, mu, nstep=60
                    )[:3]
                    - site
                )
            )
            for t in times
        ]
    )
    guess = truth + np.array([30.0, -20.0, 10.0, 0.05, -0.03, 0.02])
    estimate, covariance = batch_od.batch_od(times, site, ranges, guess, itmax=6)
    assert estimate.shape == (6,)
    assert covariance.shape == (6, 6)
    assert float(np.linalg.norm(estimate - truth)) < float(np.linalg.norm(guess - truth))
