"""evalues: anytime-valid head promotion — martingale, causality, validity.

The promotion e-process bets on the *sign* of the per-origin loss diff
with a predictable Kelly plug-in fraction — valid under the conditional
median null ``P(d_i < 0 | F_{i-1}) <= P(d_i > 0 | F_{i-1})``, which every
iid stream with ``median(d) >= 0`` satisfies regardless of tails, skew,
or scale. The property tests below pin that guarantee empirically.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.evalues import LossEProcess, promotion_report


def test_identical_streams_never_promote() -> None:
    proc = LossEProcess(alpha=0.05)
    for _ in range(200):
        st = proc.update(0.5, 0.5)
    assert st.evalue == pytest.approx(1.0)
    assert st.anytime_p == pytest.approx(1.0)
    assert proc.promotion_origin is None


def test_better_challenger_promotes() -> None:
    rng = np.random.default_rng(0)
    chall = 0.5 + rng.normal(0, 0.01, size=400)
    inc = 0.5 + rng.normal(0, 0.01, size=400) + 0.005
    rep = promotion_report(chall, inc, challenger="new", incumbent="old")
    assert rep["promoted"] is True
    assert rep["anytime_p"] < 0.05
    assert rep["promotion_origin"] is not None and rep["promotion_origin"] < 400
    assert rep["mean_loss_diff"] < 0


def test_worse_challenger_never_promotes() -> None:
    """A perpetually-losing challenger collects no evidence: lam -> 0, E ~ 1."""
    proc = LossEProcess()
    for _ in range(50):
        st = proc.update(1.0, 0.5)
    assert st.evalue <= 1.0 + 1e-12
    assert st.anytime_p == pytest.approx(1.0)
    assert proc.promotion_origin is None


def test_anytime_p_is_running_max_of_wealth() -> None:
    """Ville bounds the SUPREMUM: once promoted, a dip below 1/alpha must
    leave anytime_p <= alpha (latched), not snap back above it."""
    proc = LossEProcess(alpha=0.05)
    for _ in range(60):
        st = proc.update(0.0, 1.0)  # challenger better → wealth crosses 20
    assert st.promoted
    for _ in range(55):
        st = proc.update(1.0, 0.0)  # wealth dips back under the threshold
    assert st.evalue < 20.0
    assert st.promoted  # latched
    assert st.anytime_p <= 0.05  # 1/max E, not 1/current E
    assert st.anytime_p < 1.0 / st.evalue  # strictly below the naive inverse


def test_causality_prefix_invariant() -> None:
    rng = np.random.default_rng(7)
    c = rng.normal(0.5, 0.02, size=80)
    b = rng.normal(0.5, 0.02, size=80)
    proc_a = LossEProcess()
    states_full = [proc_a.update(ci, bi) for ci, bi in zip(c, b, strict=True)]
    proc_b = LossEProcess()
    states_prefix = [proc_b.update(ci, bi) for ci, bi in zip(c[:40], b[:40], strict=True)]
    # Editing the suffix cannot change any prefix state.
    for sf, sp in zip(states_full[:40], states_prefix, strict=True):
        assert sf.evalue == pytest.approx(sp.evalue)
        assert sf.anytime_p == pytest.approx(sp.anytime_p)


def test_determinism() -> None:
    rng = np.random.default_rng(3)
    c = rng.normal(0.5, 0.05, size=60).tolist()
    b = rng.normal(0.5, 0.05, size=60).tolist()
    r1 = promotion_report(c, b)
    r2 = promotion_report(c, b)
    assert r1 == r2


def test_scale_invariance_of_bet() -> None:
    """The sign bet ignores magnitude: positive rescaling cannot change a state."""
    rng = np.random.default_rng(13)
    c = rng.normal(0.5, 0.02, size=80)
    b = rng.normal(0.5, 0.02, size=80)
    for mult in (1.0, 1e3, 1e-3):
        pa, pb = LossEProcess(), LossEProcess()
        for ci, bi in zip(c, b, strict=True):
            sa = pa.update(ci, bi)
            sb = pb.update(ci * mult, bi * mult)
            assert sa.evalue == pytest.approx(sb.evalue)
            assert sa.anytime_p == pytest.approx(sb.anytime_p)
            assert sa.promoted == sb.promoted


def test_init_scale_does_not_affect_bet() -> None:
    rng = np.random.default_rng(17)
    d = rng.normal(0.0, 0.02, size=60)
    pa, pb = LossEProcess(init_scale=1e-6), LossEProcess(init_scale=10.0)
    for di in d:
        sa = pa.update(float(di), 0.0)
        sb = pb.update(float(di), 0.0)
        assert sa.evalue == pytest.approx(sb.evalue)


def test_lam_predictable_only() -> None:
    """lam_i must use strict history: state at origin i cannot see d_i's sign."""
    # A stream where the current diff is huge negative: if lam_i peeked at it,
    # the first factor would already bet >1 — it must stay neutral.
    proc = LossEProcess()
    st = proc.update(-1e6, 0.0)
    assert st.evalue == pytest.approx(1.0)
    # after observing one win, the next step may bet
    st2 = proc.update(-1.0, 0.0)
    assert st2.evalue > 1.0


def test_martingale_validity_under_symmetric_null() -> None:
    """Under a symmetric zero-mean differential the promotion rate is <= alpha + slack."""
    promoted = 0
    trials = 200
    for seed in range(trials):
        rng = np.random.default_rng(1000 + seed)
        d = rng.normal(0.0, 0.02, size=120)
        proc = LossEProcess(alpha=0.05)
        for di in d:
            proc.update(float(di), 0.0)
        if proc.promotion_origin is not None:
            promoted += 1
    # Ville bound guarantees <=5%; e-processes are conservative in practice.
    assert promoted / trials <= 0.10


def _promotion_rate(make_stream, n_sims: int, n_origins: int, alpha: float = 0.05) -> float:
    """Fraction of runs whose evalue ever reaches 1/alpha (Ville exceedance)."""
    promoted = 0
    for seed in range(n_sims):
        rng = np.random.default_rng(seed * 7919 + 17)
        proc = LossEProcess(alpha=alpha)
        for di in make_stream(rng, n_origins):
            proc.update(float(di), 0.0)
        if proc.promotion_origin is not None:
            promoted += 1
    return promoted / n_sims


def test_ville_null_control_iid_gaussian() -> None:
    """Empirical null-control rate: fraction ever exceeding 1/alpha <= alpha + MC slack."""
    rate = _promotion_rate(lambda r, n: r.normal(0.0, 0.02, n), n_sims=1500, n_origins=150)
    # expected true rate is well below alpha for a conservative process; the
    # assertion keeps alpha + slack headroom against MC noise.
    assert rate <= 0.08


def test_ville_null_control_heavy_tail() -> None:
    rate = _promotion_rate(lambda r, n: r.standard_t(3, n) * 0.02, n_sims=1500, n_origins=150)
    assert rate <= 0.08


def test_ville_null_control_skewed_median_nonneg() -> None:
    """Skewed stream with median >= 0 — inside the sign-null — must still control."""
    # 55% mass at +0.01, 45% at -0.5: median > 0, heavy negative tail, mean < 0.
    rate = _promotion_rate(
        lambda r, n: np.where(r.random(n) < 0.55, 0.01, -0.5),
        n_sims=1500,
        n_origins=150,
    )
    assert rate <= 0.08


def test_supermartingale_increment_bound() -> None:
    """Each e-factor is a valid e-value under the null: E[e_i] <= 1 per origin.

    Pooled across origins and sims; the factors lie in [1-lam, 1+lam] so the
    MC error on the pooled mean is tiny.
    """
    n_sims, n_origins = 400, 150
    total, count = 0.0, 0
    for seed in range(n_sims):
        rng = np.random.default_rng(seed * 313 + 5)
        d = rng.normal(0.0, 0.02, size=n_origins)
        proc = LossEProcess()
        prev_log = 0.0
        for di in d:
            st = proc.update(float(di), 0.0)
            log_e = float(np.log(st.evalue))
            total += float(np.exp(log_e - prev_log))
            prev_log = log_e
            count += 1
    mean_factor = total / count
    assert mean_factor <= 1.02


def test_eprocess_mean_bound_via_factors() -> None:
    """E[M_t] <= 1 under median-null: verified through the per-step factor bound.

    The raw product mean is unusable at large T (variance of a multiplicative
    process explodes); the valid-e-value-per-step check is the tight, correct
    pin: M_t = prod e_i is a supermartingale iff every factor has E[e_i|F] <= 1.
    Skewed and heavy-tailed nulls included.
    """
    for maker in (
        lambda r, n: r.standard_t(3, n) * 0.02,
        lambda r, n: np.where(r.random(n) < 0.55, 0.01, -0.5),
        lambda r, n: r.normal(0.0, 0.02, n),
    ):
        n_sims, n_origins = 300, 120
        total, count = 0.0, 0
        for seed in range(n_sims):
            rng = np.random.default_rng(seed * 101 + 29)
            proc = LossEProcess()
            prev = 1.0
            for di in maker(rng, n_origins):
                st = proc.update(float(di), 0.0)
                total += st.evalue / prev
                prev = st.evalue
                count += 1
        assert total / count <= 1.02


def test_alternative_promotes() -> None:
    """A consistently better challenger is promoted — the process has power."""
    promoted = 0
    crossing: list[int] = []
    trials, n_origins = 100, 300
    for seed in range(trials):
        rng = np.random.default_rng(4000 + seed)
        # challenger better: win-rate ~0.64 per origin
        d = rng.normal(-0.005, 0.014, size=n_origins)
        proc = LossEProcess(alpha=0.05)
        for di in d:
            proc.update(float(di), 0.0)
        if proc.promotion_origin is not None:
            promoted += 1
            crossing.append(proc.promotion_origin)
    assert promoted / trials >= 0.9
    assert float(np.median(crossing)) <= 200


def test_fails_closed_on_bad_input() -> None:
    with pytest.raises(ValueError, match="equal length"):
        promotion_report([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="nonempty"):
        promotion_report([], [])
    with pytest.raises(ValueError, match="finite"):
        promotion_report([np.nan, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="lam"):
        LossEProcess(lam=1.5)
    with pytest.raises(ValueError, match="alpha"):
        LossEProcess(alpha=0.0)


def test_report_schema() -> None:
    rng = np.random.default_rng(11)
    rep = promotion_report(rng.normal(0.4, 0.02, 50), rng.normal(0.5, 0.02, 50), alpha=0.01)
    assert rep["kind"] == "evalue_promotion.v1"
    assert rep["alpha"] == 0.01
    assert rep["n_origins"] == 50
    assert "ville_inequality" in rep["evidence"]
    assert rep["final_evalue"] > 0
    assert 0.0 <= rep["challenger_win_rate"] <= 1.0
