"""Tests for quant_fund.models.kit_paths — KiT OHLCV candle-path generation.

References: Zhang & Li (2026, arXiv:2609.34507, cs.LG) — the K-line
Diffusion Transformer (flow matching on a DiT backbone, Eq. 3 five-channel
candle encoding, dual classifier-free guidance of Eq. 9); Peebles & Xie
(2023, DiT); Liu et al. (2023) / Lipman et al. (2023) / Esser et al.
(2024) — flow matching; Gneiting & Raftery (2007) — energy score.

All data here is SYNTHETIC (seeded two-regime candle streams and hand-built
latents) — correctness evidence for the algorithm, never market evidence;
no live-trading claims. Torch tests skip cleanly when the nn extra is
absent.
"""

from __future__ import annotations

import importlib.util
import math
import sys

import numpy as np
import pytest

from quant_fund.models import kit_paths as kp


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="KiT training requires the nn extra (torch)"
)

HL = 32.0
ALPHA = 1.0 - math.exp(-math.log(2.0) / HL)


def _stream(n: int = 128, seed: int = 0) -> np.ndarray:
    return kp.synthetic_ohlcv(n, seed, ema_halflife=HL)


# ---------------------------------------------------------------------------
# Eq. 3 encoder
# ---------------------------------------------------------------------------


class TestEncode:
    def test_first_bar_gapless_and_volume_selfseed(self):
        s = _stream(16)
        enc = kp.encode_candles(s, ema_halflife=HL)
        assert enc.states[0, 0] == pytest.approx(0.0, abs=1e-15)  # r_gap
        assert enc.states[0, 4] == pytest.approx(0.0, abs=1e-15)  # v (E_1=V_1)
        assert enc.vol_ema[0] == pytest.approx(s[0, 4])

    def test_closed_form_single_bar(self):
        o, h, lo, c, v = 100.0, 105.0, 98.0, 103.0, 1000.0
        enc = kp.encode_ohlcv(
            np.array([o]),
            np.array([h]),
            np.array([lo]),
            np.array([c]),
            np.array([v]),
            ema_halflife=HL,
            prev_close0=99.0,
            ema0=2000.0,
            volume0=1500.0,
        )
        e1 = ALPHA * 1500.0 + (1.0 - ALPHA) * 2000.0
        exp = np.array(
            [
                [
                    math.log(100.0 / 99.0),
                    math.log(103.0 / 100.0),
                    math.log(105.0 / 103.0),
                    math.log(100.0 / 98.0),
                    math.log(1001.0 / (e1 + 1.0)),
                ]
            ]
        )
        np.testing.assert_allclose(enc.states, exp, rtol=1e-12, atol=1e-12)
        assert enc.vol_ema[0] == pytest.approx(e1)
        assert enc.prev_close0 == pytest.approx(99.0)

    def test_shadows_nonneg_by_construction(self):
        enc = kp.encode_candles(_stream(64), ema_halflife=HL)
        assert np.all(enc.states[:, 2] >= 0.0)  # r_up
        assert np.all(enc.states[:, 3] >= 0.0)  # r_dn

    def test_doji_has_zero_shadows(self):
        flat = np.array([[50.0, 50.0, 50.0, 50.0, 10.0]])
        enc = kp.encode_candles(flat, ema_halflife=HL)
        np.testing.assert_allclose(enc.states[0, :4], 0.0, atol=1e-15)

    def test_volume_ema_is_strictly_causal(self):
        s = _stream(32)
        enc_a = kp.encode_candles(s, ema_halflife=HL)
        s_pert = s.copy()
        s_pert[10, 4] *= 100.0  # huge spike at bar 10
        enc_b = kp.encode_candles(s_pert, ema_halflife=HL)
        # E_t uses V_{<t}: bars <= 10 share the EMA chain exactly.
        np.testing.assert_array_equal(enc_a.vol_ema[:11], enc_b.vol_ema[:11])
        assert enc_b.vol_ema[11] != enc_a.vol_ema[11]
        assert enc_b.states[10, 4] != enc_a.states[10, 4]

    def test_ema_recursion_matches(self):
        s = _stream(32)
        enc = kp.encode_candles(s, ema_halflife=HL)
        for t in range(1, 8):
            e_t = ALPHA * s[t - 1, 4] + (1.0 - ALPHA) * enc.vol_ema[t - 1]
            assert enc.vol_ema[t] == pytest.approx(e_t, rel=1e-14)

    def test_continuation_encode_equals_slice(self):
        s = _stream(64)
        enc = kp.encode_candles(s, ema_halflife=HL)
        k = 20
        tail = kp.encode_ohlcv(
            s[k:, 0],
            s[k:, 1],
            s[k:, 2],
            s[k:, 3],
            s[k:, 4],
            ema_halflife=HL,
            prev_close0=float(s[k - 1, 3]),
            ema0=float(enc.vol_ema[k - 1]),
            volume0=float(s[k - 1, 4]),
        )
        np.testing.assert_allclose(tail.states, enc.states[k:], rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        "col,val",
        [
            (0, -1.0),  # non-positive open
            (3, 0.0),  # zero close
            (4, -0.5),  # negative volume
        ],
    )
    def test_fail_closed_bad_values(self, col, val):
        s = _stream(8)
        s[3, col] = val
        with pytest.raises(ValueError):
            kp.encode_candles(s, ema_halflife=HL)

    def test_fail_closed_inconsistent_high_low(self):
        s = _stream(8)
        s[2, 1] = s[2, 0] * 0.5  # high below max(O, C)
        with pytest.raises(ValueError, match="high"):
            kp.encode_candles(s, ema_halflife=HL)
        s = _stream(8)
        s[2, 2] = s[2, 0] * 2.0  # low above min(O, C)
        with pytest.raises(ValueError, match="low"):
            kp.encode_candles(s, ema_halflife=HL)

    def test_fail_closed_nan_and_shape(self):
        s = _stream(8)
        s[0, 0] = np.nan
        with pytest.raises(ValueError):
            kp.encode_candles(s, ema_halflife=HL)
        with pytest.raises(ValueError):
            kp.encode_candles(np.empty((0, 5)), ema_halflife=HL)
        with pytest.raises(ValueError):
            kp.encode_candles(np.ones((8, 4)), ema_halflife=HL)
        with pytest.raises(ValueError):
            kp.encode_ohlcv(np.ones(4), np.ones(3), np.ones(4), np.ones(4), np.ones(4))

    def test_encode_candles_matches_components(self):
        s = _stream(16)
        a = kp.encode_candles(s, ema_halflife=HL)
        b = kp.encode_ohlcv(s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], ema_halflife=HL)
        np.testing.assert_array_equal(a.states, b.states)


# ---------------------------------------------------------------------------
# Structural decoder
# ---------------------------------------------------------------------------


class TestDecode:
    def test_roundtrip_exact(self):
        s = _stream(96)
        enc = kp.encode_candles(s, ema_halflife=HL)
        dec = kp.decode_encoding(enc)
        np.testing.assert_allclose(dec.candles, s, rtol=1e-10, atol=1e-8)
        assert dec.last_close == pytest.approx(s[-1, 3])
        assert dec.last_volume == pytest.approx(s[-1, 4])
        assert dec.last_ema == pytest.approx(enc.vol_ema[-1])

    @pytest.mark.parametrize("rectifier", ["relu", "softplus"])
    def test_arbitrary_latents_decode_legal(self, rectifier):
        rng = np.random.default_rng(7)
        lat = rng.normal(0.0, 2.0, (4, 24, 5))
        lat[..., 4] = rng.normal(-10.0, 20.0, (4, 24))  # extreme volume draws
        dec = kp.decode_kit_states(
            lat, 100.0, ema_halflife=HL, ema0=1e4, volume0=1e4, rectifier=rectifier
        )
        rep = kp.candle_consistency_report(dec.candles)
        assert rep["violation_rate"] == 0.0
        assert rep["n_violations"] == 0.0

    def test_relu_clamps_shadow_to_body_edge(self):
        lat = np.array([[0.0, 0.0, -5.0, -7.0, 0.0]])  # negative shadow latents
        dec = kp.decode_kit_states(
            lat, 100.0, ema_halflife=HL, vol_ema=np.array([10.0]), rectifier="relu"
        )
        o, h, lo, c, _v = dec.candles[0]
        assert h == pytest.approx(max(o, c), rel=1e-14)
        assert lo == pytest.approx(min(o, c), rel=1e-14)

    def test_softplus_strictly_above_body(self):
        lat = np.array([[0.0, 0.0, -5.0, -7.0, 0.0]])
        dec = kp.decode_kit_states(
            lat, 100.0, ema_halflife=HL, vol_ema=np.array([10.0]), rectifier="softplus"
        )
        o, h, lo, c, _v = dec.candles[0]
        assert h > max(o, c)
        assert lo < min(o, c)

    def test_volume_floored_at_zero(self):
        lat = np.array([[0.0, 0.0, 0.0, 0.0, -40.0]])
        dec = kp.decode_kit_states(lat, 100.0, ema_halflife=HL, vol_ema=np.array([10.0]))
        assert dec.candles[0, 4] == 0.0

    def test_close_chain_coherence(self):
        s = _stream(24)
        enc = kp.encode_candles(s, ema_halflife=HL)
        dec = kp.decode_encoding(enc)
        for t in range(1, dec.candles.shape[0]):
            assert dec.candles[t, 0] == pytest.approx(
                dec.candles[t - 1, 3] * math.exp(enc.states[t, 0]), rel=1e-12
            )

    def test_batched_decode_equals_looped(self):
        rng = np.random.default_rng(3)
        lat = rng.normal(0.0, 1.0, (5, 12, 5))
        dec = kp.decode_kit_states(lat, 50.0, ema_halflife=HL, ema0=1e3, volume0=1e3)
        assert dec.candles.shape == (5, 12, 5)
        for m in range(5):
            d1 = kp.decode_kit_states(lat[m], 50.0, ema_halflife=HL, ema0=1e3, volume0=1e3)
            np.testing.assert_allclose(dec.candles[m], d1.candles, rtol=0, atol=0)
        assert dec.last_close.shape == (5,)

    def test_continuation_anchors_reproduce_tail(self):
        s = _stream(64)
        enc = kp.encode_candles(s, ema_halflife=HL)
        k = 20
        dec = kp.decode_kit_states(
            enc.states[k:],
            float(s[k - 1, 3]),
            ema_halflife=HL,
            ema0=float(enc.vol_ema[k - 1]),
            volume0=float(s[k - 1, 4]),
        )
        np.testing.assert_allclose(dec.candles, s[k:], rtol=1e-10, atol=1e-8)

    def test_fail_closed_missing_anchor(self):
        lat = np.zeros((4, 5))
        with pytest.raises(ValueError, match="anchor"):
            kp.decode_kit_states(lat, 100.0, ema_halflife=HL)
        with pytest.raises(ValueError):
            kp.decode_kit_states(lat, 100.0, ema_halflife=HL, ema0=1.0)
        with pytest.raises(ValueError):
            kp.decode_kit_states(lat, 100.0, ema_halflife=HL, vol_ema=np.ones(3))
        with pytest.raises(ValueError):
            kp.decode_kit_states(
                lat, 100.0, ema_halflife=HL, vol_ema=np.ones(4), ema0=1.0, volume0=1.0
            )
        with pytest.raises(ValueError, match="rectifier"):
            kp.decode_kit_states(lat, 100.0, ema_halflife=HL, vol_ema=np.ones(4), rectifier="bogus")
        with pytest.raises(ValueError):
            kp.decode_kit_states(lat, -1.0, ema_halflife=HL, vol_ema=np.ones(4))

    def test_fail_closed_extreme_latent_overflow(self):
        lat = np.full((3, 5), 700.0)
        with pytest.raises(ValueError, match="not finite"):
            kp.decode_kit_states(lat, 100.0, ema_halflife=HL, vol_ema=np.ones(3))


# ---------------------------------------------------------------------------
# Robust scaler (median/MAD + tanh)
# ---------------------------------------------------------------------------


class TestScaler:
    def test_fit_values_closed_form(self):
        x = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [2.0, 3.0, 0.0, 0.0, 0.0],
                [4.0, 9.0, 0.0, 0.0, 0.0],
            ]
        )
        sc = kp.KitStateScaler.fit(x)
        np.testing.assert_allclose(sc.median[:2], [2.0, 3.0])
        # MAD of [0,2,4] = 2 -> scale 2.9652; MAD of [1,3,9] = 2.
        np.testing.assert_allclose(sc.scale[:2], 1.4826 * 2.0)
        np.testing.assert_allclose(sc.scale[2:], 1.0)  # MAD=0 -> 1

    def test_transform_bounds_and_median_maps_zero(self):
        enc = kp.encode_candles(_stream(64), ema_halflife=HL)
        sc = kp.KitStateScaler.fit(enc.states)
        w = sc.transform(enc.states)
        assert np.all(np.abs(w) < 1.0)
        row0 = sc.transform(sc.median[None, :])
        np.testing.assert_allclose(row0, 0.0, atol=1e-15)

    def test_roundtrip_moderate_range(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 0.01, (200, 5))
        sc = kp.KitStateScaler.fit(x)
        rt = sc.inverse(sc.transform(x))
        np.testing.assert_allclose(rt, x, rtol=1e-9, atol=1e-10)

    def test_degenerate_column_passthrough(self):
        x = np.tile(np.array([0.0, 1.0, 0.0, 0.0, 3.0]), (10, 1))
        sc = kp.KitStateScaler.fit(x)
        w = sc.transform(x)
        np.testing.assert_allclose(w, 0.0, atol=1e-15)
        np.testing.assert_allclose(sc.inverse(w), x, atol=1e-12)

    def test_fit_needs_two_rows(self):
        with pytest.raises(ValueError):
            kp.KitStateScaler.fit(np.zeros((1, 5)))

    def test_fail_closed_shapes(self):
        sc = kp.KitStateScaler.fit(np.random.default_rng(0).normal(size=(10, 5)))
        with pytest.raises(ValueError):
            sc.transform(np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------


class TestWindows:
    def test_shapes_and_alignment(self):
        st = np.arange(40 * 5, dtype=float).reshape(40, 5) / 100.0
        ctx, tgt = kp.kit_windows(st, context=8, horizon=4)
        assert ctx.shape == (40 - 8 - 4 + 1, 8, 5)
        assert tgt.shape == (40 - 8 - 4 + 1, 4, 5)
        np.testing.assert_array_equal(ctx[0], st[:8])
        np.testing.assert_array_equal(tgt[0], st[8:12])
        np.testing.assert_array_equal(ctx[1], st[1:9])
        np.testing.assert_array_equal(tgt[-1], st[-4:])

    def test_fail_closed(self):
        st = np.zeros((10, 5))
        with pytest.raises(ValueError, match="context"):
            kp.kit_windows(st, 8, 4)
        with pytest.raises(ValueError, match="single"):
            kp.kit_windows(np.zeros((2, 10, 5)), 8, 4)

    def test_generic_feature_dim(self):
        # calendar conditioning windows a (n, d) sequence with d != 5
        st = np.arange(30 * 3, dtype=float).reshape(30, 3) / 100.0
        ctx, tgt = kp.kit_windows(st, context=8, horizon=4)
        assert ctx.shape == (30 - 8 - 4 + 1, 8, 3)
        assert tgt.shape == (30 - 8 - 4 + 1, 4, 3)
        np.testing.assert_array_equal(ctx[0], st[:8])
        np.testing.assert_array_equal(tgt[0], st[8:12])
        np.testing.assert_array_equal(tgt[-1], st[-4:])
        bad = st.copy()
        bad[5, 1] = np.nan
        with pytest.raises(ValueError, match="finite"):
            kp.kit_windows(bad, 8, 4)
        with pytest.raises(ValueError, match="non-empty"):
            kp.kit_windows(np.zeros((10, 0)), 8, 4)


# ---------------------------------------------------------------------------
# Consistency diagnostics
# ---------------------------------------------------------------------------


class TestConsistency:
    def test_clean_stream_zero(self):
        rep = kp.candle_consistency_report(_stream(64))
        assert rep["violation_rate"] == 0.0
        assert rep["n_bars"] == 64.0

    def test_detects_each_axiom(self):
        base = np.array([[100.0, 105.0, 98.0, 103.0, 10.0]])
        assert kp.candle_consistency_report(base)["violation_rate"] == 0.0
        bad_high = base.copy()
        bad_high[0, 1] = 90.0
        rep = kp.candle_consistency_report(bad_high)
        assert rep["high_violation_rate"] == 1.0 and rep["violation_rate"] == 1.0
        bad_low = base.copy()
        bad_low[0, 2] = 110.0
        rep = kp.candle_consistency_report(bad_low)
        assert rep["low_violation_rate"] == 1.0
        bad_vol = base.copy()
        bad_vol[0, 4] = -1.0
        rep = kp.candle_consistency_report(bad_vol)
        assert rep["volume_violation_rate"] == 1.0
        bad_px = base.copy()
        bad_px[0, 0] = 0.0
        rep = kp.candle_consistency_report(bad_px)
        assert rep["price_violation_rate"] == 1.0

    def test_batched_counting(self):
        ok = np.tile(np.array([[100.0, 105.0, 98.0, 103.0, 10.0]]), (2, 4, 1))
        ok[1, 2, 4] = -1.0
        rep = kp.candle_consistency_report(ok)
        assert rep["n_bars"] == 8.0
        assert rep["violation_rate"] == pytest.approx(1.0 / 8.0)

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            kp.candle_consistency_report(np.zeros((4, 4)))
        s = _stream(8)
        s[0, 0] = np.nan
        with pytest.raises(ValueError):
            kp.candle_consistency_report(s)


# ---------------------------------------------------------------------------
# Proper path scores (state space)
# ---------------------------------------------------------------------------


class TestPathScores:
    def _obs(self) -> np.ndarray:
        return kp.encode_candles(_stream(48), ema_halflife=HL).states[-8:]

    def test_degenerate_ensemble_zero_loss(self):
        obs = self._obs()
        ens = np.tile(obs[None, :, :], (16, 1, 1))
        rep = kp.path_score_report(ens, obs)
        assert rep["energy_score"] == pytest.approx(0.0, abs=1e-12)
        assert rep["crps_marginal_mean"] == pytest.approx(0.0, abs=1e-12)
        assert rep["crps_terminal_ret"] == pytest.approx(0.0, abs=1e-12)
        assert rep["pinball_50_terminal_ret"] == pytest.approx(0.0, abs=1e-12)
        assert rep["coverage_ret"] == 1.0
        assert rep["width_ret"] == pytest.approx(0.0, abs=1e-12)

    def test_shifted_ensemble_pinball_closed_form(self):
        obs = self._obs()
        shift = 0.1
        ens = np.tile(obs[None, :, :], (16, 1, 1))
        ens[..., 0] += shift  # r_gap shifts each step -> terminal ret by L*s
        rep = kp.path_score_report(ens, obs)
        # deterministic ensemble -> CRPS = |L s|, median pinball = 0.5 L s
        assert rep["crps_terminal_ret"] == pytest.approx(8 * shift, rel=1e-9)
        assert rep["pinball_50_terminal_ret"] == pytest.approx(0.5 * 8 * shift, rel=1e-9)
        assert rep["coverage_ret"] == 0.0  # realized sits below the band
        assert rep["energy_score"] > 0.0

    def test_terminal_ret_is_cumsum_gap_plus_body(self):
        obs = self._obs()
        ens = np.tile(obs[None, :, :], (8, 1, 1))
        ens[..., 1] += 0.02  # body channel also moves the terminal return
        rep = kp.path_score_report(ens, obs)
        s = 8 * 0.02  # horizon L=8 -> terminal shift L*0.02
        assert rep["pinball_50_terminal_ret"] == pytest.approx(0.5 * s, rel=1e-9)

    def test_fail_closed(self):
        obs = self._obs()
        ens = np.tile(obs[None, :, :], (4, 1, 1))
        with pytest.raises(ValueError):
            kp.path_score_report(ens[:, :4, :], obs)
        with pytest.raises(ValueError):
            kp.path_score_report(ens, obs, interval=1.5)
        with pytest.raises(ValueError):
            kp.path_score_report(ens[:, :, :4], obs)


# ---------------------------------------------------------------------------
# SYNTHETIC fixture + samplers
# ---------------------------------------------------------------------------


class TestSyntheticAndSamplers:
    def test_synthetic_deterministic_and_legal(self):
        a = kp.synthetic_ohlcv(64, 11, ema_halflife=HL)
        b = kp.synthetic_ohlcv(64, 11, ema_halflife=HL)
        c = kp.synthetic_ohlcv(64, 12, ema_halflife=HL)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)
        assert kp.candle_consistency_report(a)["violation_rate"] == 0.0
        assert np.all(a[:, :4] > 0.0)

    def test_synthetic_fail_closed(self):
        for kwargs in ({"n_bars": 0},):
            with pytest.raises(ValueError):
                kp.synthetic_ohlcv(**kwargs, seed=0)
        with pytest.raises(ValueError):
            kp.synthetic_ohlcv(10, 0, s0=-1.0)
        with pytest.raises(ValueError):
            kp.synthetic_ohlcv(10, 0, vol_persist=1.5)

    def test_bootstrap_membership_and_determinism(self):
        pool = np.random.default_rng(0).normal(size=(9, 4, 5))
        a = kp.bootstrap_horizon(pool, 32, 5)
        b = kp.bootstrap_horizon(pool, 32, 5)
        np.testing.assert_array_equal(a, b)
        assert a.shape == (32, 4, 5)
        for row in a:
            assert any(np.array_equal(row, p) for p in pool)
        with pytest.raises(ValueError):
            kp.bootstrap_horizon(np.zeros((4, 5)), 4, 0)

    def test_logit_normal_range_and_determinism(self):
        t = kp.logit_normal_times(512, 3)
        assert np.all((t > 0.0) & (t < 1.0))
        np.testing.assert_array_equal(t, kp.logit_normal_times(512, 3))
        with pytest.raises(ValueError):
            kp.logit_normal_times(0, 0)

    def test_bench_keys_and_consistency(self):
        rep = kp.bench_kit_paths(n_train_bars=192, context=16, horizon=4, n_samples=32, seed=0)
        assert rep["synthetic_violation_rate"] == 0.0
        assert rep["synthetic_source_violation_rate"] == 0.0
        assert rep["synthetic_roundtrip_max_abs_err"] < 1e-6
        assert math.isfinite(rep["synthetic_energy_score"])
        assert math.isfinite(rep["synthetic_crps_marginal_mean"])
        assert 0.0 <= rep["synthetic_coverage_ret"] <= 1.0
        assert rep["synthetic_claim"] == "research_metric_only"
        assert all(k.startswith("synthetic_") for k in rep)


# ---------------------------------------------------------------------------
# Torch gating (no torch needed for these)
# ---------------------------------------------------------------------------


class TestTorchGating:
    def test_torch_helper_raises_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(ImportError, match="nn"):
            kp._torch()

    def test_validation_precedes_torch(self):
        gen = kp.KitPathGenerator()
        bad = _stream(40)
        bad[0, 0] = -1.0
        with pytest.raises(ValueError):
            gen.fit_ohlcv(bad, context=8, horizon=4)
        with pytest.raises(ValueError):
            gen.fit_ohlcv(_stream(10), context=8, horizon=4)  # too few bars
        with pytest.raises(RuntimeError, match="not fitted"):
            gen.sample_ohlcv(_stream(16))

    def test_config_validation(self):
        with pytest.raises(ValueError):
            kp.KitConfig(d_model=30, n_heads=4)  # not divisible
        with pytest.raises(ValueError):
            kp.KitConfig(d_model=28, n_heads=4)  # odd head dim (RoPE)
        with pytest.raises(ValueError):
            kp.KitConfig(w_id=0.5)
        with pytest.raises(ValueError):
            kp.KitConfig(cond_cardinalities=(4,), unk_signals=(2,))

    def test_condition_dropout_slots(self):
        cfg = kp.KitConfig(
            cond_cardinalities=(3, 5), unk_signals=(1,), p_unk=1.0, p_null=0.0, p_hist=1.0
        )
        ids = np.tile(np.array([1, 2]), (6, 1))
        out, hist = kp._apply_condition_dropout(ids, cfg, 6, np.random.default_rng(0))
        assert np.all(hist)  # p_hist = 1
        assert np.all(out[:, 1] == 5 + 1)  # [UNK] = card + 1
        assert np.all(out[:, 0] == 1)
        cfg2 = kp.KitConfig(cond_cardinalities=(3, 5), p_null=1.0)
        out2, _ = kp._apply_condition_dropout(ids, cfg2, 6, np.random.default_rng(0))
        np.testing.assert_array_equal(out2, np.tile(np.array([3, 5]), (6, 1)))


# ---------------------------------------------------------------------------
# torch lane (skipped without the nn extra)
# ---------------------------------------------------------------------------


@requires_torch
class TestKitTorch:
    @staticmethod
    def _cfg(**kw) -> kp.KitConfig:
        base = dict(
            d_model=32,
            n_heads=4,
            n_layers=2,
            n_register=2,
            epochs=5,
            batch_size=32,
            n_euler=8,
            seed=0,
        )
        base.update(kw)
        return kp.KitConfig(**base)

    @staticmethod
    def _fitted(**kw) -> kp.KitPathGenerator:
        cfg = kp.KitPathGenerator(kw.pop("cfg", TestKitTorch._cfg(**kw)))
        cfg.fit_ohlcv(_stream(220)[:200], context=16, horizon=8)
        return cfg

    def test_zero_init_velocity(self):
        torch = kp._torch()
        model = kp._build_kit_backbone(torch, self._cfg(), context_len=6, horizon_len=4)
        ctx = torch.randn(2, 6, 5)
        z = torch.randn(2, 4, 5)
        t = torch.rand(2)
        out = model(ctx, z, t)
        assert out.shape == (2, 4, 5)
        np.testing.assert_allclose(out.detach().numpy(), 0.0, atol=1e-7)

    def test_forward_shape_and_finite(self):
        torch = kp._torch()
        cfg = self._cfg(cond_cardinalities=(3,), calendar_dim=2)
        model = kp._build_kit_backbone(torch, cfg, context_len=6, horizon_len=4)
        ctx = torch.randn(3, 6, 5)
        z = torch.randn(3, 4, 5)
        t = torch.rand(3)
        ids = torch.randint(0, 3, (3, 1))
        cal_c = torch.randn(3, 6, 2)
        cal_t = torch.randn(3, 4, 2)
        out = model(ctx, z, t, cond_ids=ids, cal_ctx=cal_c, cal_tgt=cal_t)
        assert out.shape == (3, 4, 5)
        assert torch.isfinite(out).all()

    def test_init_loss_is_vstar_mse(self):
        torch = kp._torch()
        model = kp._build_kit_backbone(torch, self._cfg(), context_len=6, horizon_len=4)
        rng = np.random.default_rng(0)
        ctx = torch.as_tensor(rng.normal(size=(4, 6, 5)), dtype=torch.float32)
        tgt = torch.as_tensor(rng.normal(size=(4, 4, 5)), dtype=torch.float32)
        t = torch.as_tensor(rng.random(4), dtype=torch.float32)
        eps = torch.as_tensor(rng.normal(size=(4, 4, 5)), dtype=torch.float32)
        loss = kp._kit_flow_matching_loss(torch, model, ctx, tgt, None, None, None, t, eps, None)
        v_star = (eps - tgt).numpy()
        assert float(loss.detach()) == pytest.approx(float(np.mean(v_star**2)), rel=1e-5)

    def test_fit_and_sample_consistency(self):
        gen = self._fitted()
        assert gen.is_fitted
        info = gen.fit_info
        assert info is not None and info.n_windows > 0
        assert math.isfinite(info.final_loss)
        assert info.loss_curve[-1] <= info.loss_curve[0] * 1.5
        out = gen.sample_ohlcv(_stream(220)[184:200], n_samples=8, seed=1)
        assert out.paths.shape == (8, 8, 5)
        assert out.consistency["violation_rate"] == 0.0
        assert np.all(out.paths[:, :, :4] > 0.0)
        np.testing.assert_allclose(kp.candle_consistency_report(out.paths)["violation_rate"], 0.0)

    def test_sample_determinism_and_seed_variation(self):
        gen = self._fitted()
        ctx = _stream(220)[184:200]
        a = gen.sample_ohlcv(ctx, n_samples=6, seed=42)
        b = gen.sample_ohlcv(ctx, n_samples=6, seed=42)
        np.testing.assert_array_equal(a.paths, b.paths)
        c = gen.sample_ohlcv(ctx, n_samples=6, seed=43)
        assert not np.array_equal(a.paths, c.paths)

    def test_dual_cfg_changes_output(self):
        gen = self._fitted()
        ctx = _stream(220)[184:200]
        base = gen.sample_states(ctx, n_samples=4, seed=9, w_id=1.0, w_hist=1.0)
        guided = gen.sample_states(ctx, n_samples=4, seed=9, w_id=2.0, w_hist=1.5)
        assert not np.allclose(base, guided)
        assert base.shape == guided.shape == (4, 8, 5)

    def test_fit_validation_contract(self):
        gen = kp.KitPathGenerator(self._cfg(cond_cardinalities=(3,)))
        s = _stream(200)
        with pytest.raises(ValueError, match="cond_ids required"):
            gen.fit_ohlcv(s, context=16, horizon=8)
        n_win = 200 - 16 - 8 + 1
        bad_ids = np.full((n_win, 1), 9, dtype=np.int64)
        with pytest.raises(ValueError, match="out of range"):
            gen.fit_ohlcv(s, context=16, horizon=8, cond_ids=bad_ids)
        ok_ids = np.zeros((n_win, 1), dtype=np.int64)
        gen.fit_ohlcv(s, context=16, horizon=8, cond_ids=ok_ids)
        with pytest.raises(ValueError, match="cond_ids required"):
            gen.sample_states(_stream(16))

    def test_sample_contract(self):
        gen = self._fitted()
        with pytest.raises(ValueError):
            gen.sample_ohlcv(_stream(8))  # wrong context length
        with pytest.raises(ValueError):
            gen.sample_ohlcv(_stream(220)[184:200], n_samples=0)
        with pytest.raises(ValueError):
            gen.sample_states(_stream(220)[184:200], w_id=0.5)
        bad = _stream(16)
        bad[0, 1] = bad[0, 0] * 0.5
        with pytest.raises(ValueError):
            gen.sample_ohlcv(bad)

    def test_fit_determinism_same_seed(self):
        # backbone Linear/Embedding init draws must be seeded by cfg.seed;
        # before the fix they consumed the unseeded global torch stream.
        s = _stream(200)
        ctx = _stream(220)[184:200]
        ga = kp.KitPathGenerator(self._cfg(seed=11)).fit_ohlcv(s, context=16, horizon=8)
        gb = kp.KitPathGenerator(self._cfg(seed=11)).fit_ohlcv(s, context=16, horizon=8)
        assert ga.fit_info is not None and gb.fit_info is not None
        np.testing.assert_array_equal(ga.fit_info.loss_curve, gb.fit_info.loss_curve)
        np.testing.assert_array_equal(
            ga.sample_states(ctx, n_samples=4, seed=3),
            gb.sample_states(ctx, n_samples=4, seed=3),
        )

    def test_calendar_conditioning_fit_and_sample(self):
        # calendar_dim != 5 must reach the model: previously kit_windows
        # rejected any non-5 trailing dim, so calendar conditioning never ran.
        gen = kp.KitPathGenerator(self._cfg(calendar_dim=3))
        rng = np.random.default_rng(5)
        cal = rng.normal(size=(200, 3))
        gen.fit_ohlcv(_stream(200), context=16, horizon=8, calendar=cal)
        assert gen.is_fitted
        out = gen.sample_states(
            _stream(16),
            n_samples=4,
            calendar_context=rng.normal(size=(16, 3)),
            calendar_target=rng.normal(size=(8, 3)),
            seed=2,
        )
        assert out.shape == (4, 8, 5)
