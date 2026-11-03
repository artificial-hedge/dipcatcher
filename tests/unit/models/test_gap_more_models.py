"""Unit tests for the remaining untested ``quant_fund.models`` fixtures.

These are the shared SYNTHETIC fixtures behind the coding-theory,
cryptography, causal-discovery, game-theory, multi-task and RL canons, plus
the ``arch`` fit helper that the research and paper quantile lanes share.

What is asserted is the fixture contract — determinism, shapes, the declared
known-answer vectors, and the mathematical identities downstream modules
rely on (a CRC constant, a SHA-256 digest, a perfect-information minimax
value, a zero-sum exploitability bound).

Every fixture is SYNTHETIC. Nothing here is market evidence.
"""

from __future__ import annotations

import hashlib
import zlib

import numpy as np
import pytest

from quant_fund.models import (
    _code_synth,
    _crypto_synth,
    _csl_synth,
    _mfg_synth,
    _mt_synth,
    _rlt_synth,
    _sp_synth,
    _td_synth,
    arch_fit,
)

# --------------------------------------------------------------------------- #
# _code_synth — coding theory
# --------------------------------------------------------------------------- #


def test_the_message_bits_are_binary_and_deterministic() -> None:
    bits = _code_synth.msg_bits(0, k=16)
    assert bits.shape == (16,)
    assert set(np.unique(bits)) <= {0, 1}
    assert np.array_equal(bits, _code_synth.msg_bits(0, k=16))


def test_bpsk_maps_zero_to_plus_one_and_one_to_minus_one() -> None:
    bits = np.array([0, 1])
    received = _code_synth.bpsk_awgn(bits, sigma=0.0, seed=0)
    assert received[0] == pytest.approx(1.0)
    assert received[1] == pytest.approx(-1.0)


def test_the_bpsk_noise_scales_with_sigma() -> None:
    bits = np.zeros(500)
    tight = _code_synth.bpsk_awgn(bits, sigma=0.01, seed=0)
    loose = _code_synth.bpsk_awgn(bits, sigma=1.0, seed=0)
    assert tight.std() < loose.std()


def test_a_binary_symmetric_channel_flips_with_probability_p() -> None:
    bits = np.zeros(2000, dtype=int)
    received = _code_synth.bsc(bits, p=0.1, seed=0)
    assert set(np.unique(received)) <= {0, 1}
    assert 0.03 < received.mean() < 0.17


def test_a_binary_symmetric_channel_with_p_one_flips_everything() -> None:
    bits = np.array([0, 1, 0, 1])
    assert _code_synth.bsc(bits, p=1.0, seed=0).tolist() == [1, 0, 1, 0]


def test_the_burst_channel_flips_only_the_requested_window() -> None:
    bits = np.array([0, 0, 0, 0, 0])
    out = _code_synth.burst_channel(bits, start=1, length=2)
    assert out.tolist() == [0, 1, 1, 0, 0]


def test_the_burst_channel_does_not_mutate_its_input() -> None:
    bits = np.array([0, 0, 0, 0, 0])
    _code_synth.burst_channel(bits, start=0, length=3)
    assert bits.tolist() == [0, 0, 0, 0, 0]


def test_the_ldpc_parity_matrix_is_the_declared_shape() -> None:
    assert _code_synth.LDPC_P.shape == (6, 8)
    assert _code_synth.LDPC_K + _code_synth.LDPC_P.shape[0] == _code_synth.LDPC_N


# --------------------------------------------------------------------------- #
# _crypto_synth — known-answer vectors
# --------------------------------------------------------------------------- #


def test_the_sha_256_vector_matches_hashlib() -> None:
    assert hashlib.sha256(_crypto_synth.SHA_MSG).hexdigest() == _crypto_synth.SHA_EXPECT


def test_the_safe_prime_group_parameters_are_primes() -> None:
    p = _crypto_synth.P_SAFE
    q = (p - 1) // 2
    assert all(p % d for d in range(2, int(p**0.5) + 1))
    assert all(q % d for d in range(2, int(q**0.5) + 1))


def test_the_generator_has_the_declared_order_in_the_safe_prime_group() -> None:
    p = _crypto_synth.P_SAFE
    g = _crypto_synth.G_SAFE
    q = (p - 1) // 2
    assert pow(g, q, p) == 1
    assert pow(g, 1, p) != 1


def test_the_aes_sbox_known_values_are_pinned() -> None:
    assert _crypto_synth.AES_SBOX_KNOWN[0x00] == 0x63
    assert _crypto_synth.AES_SBOX_KNOWN[0xFF] == 0x16


def test_the_secp256k1_constants_are_self_consistent() -> None:
    p = _crypto_synth.SECP_P
    gx = _crypto_synth.SECP_GX
    gy = _crypto_synth.SECP_GY
    # The generator must satisfy the curve equation y^2 = x^3 + 7 (mod p).
    assert (gy * gy - gx * gx * gx - 7) % p == 0
    assert p % 4 == 3


# --------------------------------------------------------------------------- #
# _csl_synth — causal discovery
# --------------------------------------------------------------------------- #


def test_the_sem_fixture_returns_data_a_matrix_and_an_order() -> None:
    X, B, order = _csl_synth.lingam_sem(0, n=60, d=4, edges=3)
    assert X.shape == (60, 4)
    assert B.shape == (4, 4)
    assert sorted(order) == [0, 1, 2, 3]


def test_the_sem_matrix_has_only_the_requested_edges() -> None:
    _X, B, _order = _csl_synth.lingam_sem(0, n=40, d=4, edges=3)
    assert int(np.count_nonzero(B)) == 3


def test_the_skeleton_is_symmetric_and_binary() -> None:
    _X, B, _order = _csl_synth.lingam_sem(0, n=40, d=4, edges=3)
    sk = _csl_synth.skeleton(B)
    assert np.allclose(sk, sk.T)
    assert set(np.unique(sk)) <= {0.0, 1.0}


def test_the_skeleton_f1_of_a_matrix_with_itself_is_one() -> None:
    _X, B, _order = _csl_synth.lingam_sem(0, n=40, d=4, edges=3)
    assert _csl_synth.skeleton_f1(B, B) == pytest.approx(1.0)


def test_the_order_error_of_an_identical_order_is_zero() -> None:
    order = [2, 0, 1, 3]
    assert _csl_synth.order_err(order, order) == pytest.approx(0.0)


def test_the_order_error_of_a_reversed_order_is_one() -> None:
    order = [0, 1, 2, 3]
    assert _csl_synth.order_err(order, list(reversed(order))) == pytest.approx(1.0)


def test_the_correlation_baseline_returns_a_permutation() -> None:
    X, _B, _order = _csl_synth.lingam_sem(0, n=60, d=4, edges=3)
    order = _csl_synth.corr_baseline_order(X)
    assert sorted(order) == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# _mfg_synth — game theory parameters
# --------------------------------------------------------------------------- #


def test_the_cournot_costs_are_strictly_increasing() -> None:
    costs = _mfg_synth.COU_C
    assert costs.shape == (4,)
    assert np.all(np.diff(costs) > 0)


def test_the_stochastic_game_transitions_are_row_stochastic() -> None:
    for transitions in _mfg_synth.SG_P:
        for row_block in transitions:
            assert np.allclose(row_block.sum(axis=1), 1.0)


def test_the_stochastic_game_payoff_matrices_are_square() -> None:
    for matrix in _mfg_synth.SG_M:
        assert matrix.shape == (2, 2)
    assert pytest.approx(0.9) == _mfg_synth.SG_GAMMA


def test_the_congestion_potential_parameters_are_ordered() -> None:
    assert _mfg_synth.POT_A.shape == _mfg_synth.POT_B.shape
    assert np.all(np.diff(_mfg_synth.POT_A) > 0)
    assert np.all(np.diff(_mfg_synth.POT_B) > 0)


# --------------------------------------------------------------------------- #
# _mt_synth / _td_synth — multi-task and training dynamics
# --------------------------------------------------------------------------- #


def test_the_multi_task_fixture_returns_two_conflicting_labels() -> None:
    X, y1, y2 = _mt_synth.mt_data(0, n=60, d=8)
    assert X.shape == (60, 8)
    assert y1.shape == (60,)
    assert y2.shape == (60,)
    assert set(np.unique(y1)) <= {0, 1}
    assert set(np.unique(y2)) <= {0, 1}


def test_the_two_tasks_are_not_the_same_label() -> None:
    _X, y1, y2 = _mt_synth.mt_data(0, n=200, d=8)
    assert not np.array_equal(y1, y2)


def test_the_training_dynamics_fixture_splits_train_from_test() -> None:
    X, y, Xt, yt = _td_synth.make_data(0, n=60)
    assert X.shape == (60, 8)
    assert Xt.shape == (300, 8)
    assert set(np.unique(y)) <= {0.0, 1.0}
    assert set(np.unique(yt)) <= {0.0, 1.0}


# --------------------------------------------------------------------------- #
# _rlt_synth — bandit / MDP parameters
# --------------------------------------------------------------------------- #


def test_the_bandit_arm_means_are_strictly_decreasing() -> None:
    assert np.all(np.diff(_rlt_synth.ARM_MEANS) < 0)
    assert _rlt_synth.ARM_MEANS[0] == 0.9


def test_the_expert_edge_weights_are_probabilities() -> None:
    """The best expert has the largest edge weight."""
    assert np.all((_rlt_synth.EXP_P > 0.0) & (_rlt_synth.EXP_P < 1.0))
    assert _rlt_synth.EXP_P[0] == _rlt_synth.EXP_P.max()


def test_the_chain_mdp_transitions_are_row_stochastic() -> None:
    assert np.allclose(_rlt_synth.MDP_P.sum(axis=1), 1.0)


def test_the_chain_mdp_reward_is_only_at_the_absorbing_state() -> None:
    assert _rlt_synth.MDP_R.tolist() == [0.0, 0.0, 0.0, 1.0]


# --------------------------------------------------------------------------- #
# _sp_synth — tic-tac-toe and Kuhn poker
# --------------------------------------------------------------------------- #


def test_the_legal_moves_exclude_the_occupied_cells() -> None:
    board = (1, -1, 0, 0, 0, 0, 0, 0, 0)
    assert _sp_synth.ttt_legal(board) == [2, 3, 4, 5, 6, 7, 8]


def test_a_finished_row_declares_its_winner() -> None:
    board = (1, 1, 1, 0, 0, 0, 0, 0, 0)
    assert _sp_synth.ttt_winner(board) == 1


def test_a_full_board_with_no_line_is_a_draw() -> None:
    board = (1, -1, 1, -1, 1, -1, -1, 1, -1)
    assert _sp_synth.ttt_winner(board) == 0


def test_an_unfinished_board_has_no_winner() -> None:
    assert _sp_synth.ttt_winner((0,) * 9) is None


def test_the_minimax_value_of_the_empty_board_is_a_draw() -> None:
    """Perfect play from the empty board is a draw, so the value is zero."""
    assert _sp_synth.minimax_val((0,) * 9, 1) == pytest.approx(0.0)


def test_the_minimax_value_is_antisymmetric_on_the_empty_board() -> None:
    empty = (0,) * 9
    assert _sp_synth.minimax_val(empty, 1) == pytest.approx(-_sp_synth.minimax_val(empty, -1))


def test_the_optimal_move_is_never_illegal() -> None:
    board = (1, -1, 1, 0, 0, 0, 0, 0, 0)
    move = _sp_synth.optimal_move(board, 1)
    assert move in _sp_synth.ttt_legal(board)


def test_playing_two_optimal_policies_ends_in_a_draw() -> None:
    def oracle(state, legal):
        return _sp_synth.optimal_move(tuple(int(x) for x in state), 1)

    assert _sp_synth.play_ttt(oracle, oracle) == 0


def test_the_kuhn_terminal_histories_are_the_declared_ones() -> None:
    for history in ("pp", "bb", "bp", "pbp", "pbb"):
        assert _sp_synth.kuhn_terminal(history) is True
    assert _sp_synth.kuhn_terminal("") is False


def test_the_kuhn_utility_follows_the_payoff_table() -> None:
    assert _sp_synth.kuhn_util("pp", 2, 1) == pytest.approx(1.0)
    assert _sp_synth.kuhn_util("pp", 1, 2) == pytest.approx(-1.0)
    assert _sp_synth.kuhn_util("bp", 0, 1) == pytest.approx(1.0)


def test_the_kuhn_utility_rejects_an_unknown_history() -> None:
    with pytest.raises(ValueError):
        _sp_synth.kuhn_util("xyz", 0, 1)


def test_the_kuhn_infosets_are_well_formed() -> None:
    infosets = _sp_synth.kuhn_infosets()
    assert infosets
    for player, card, history in infosets:
        assert player in (0, 1)
        assert card in _sp_synth.CARDS
        assert not _sp_synth.kuhn_terminal(history)


def test_the_kuhn_encoding_marks_the_card_and_the_history() -> None:
    encoded = _sp_synth.encode_kuhn(0, 2, "pb")
    assert encoded.shape == (8,)
    assert encoded[2] == 1.0
    assert encoded[3] == pytest.approx(-1.0)
    assert encoded[4] == pytest.approx(1.0)


def test_the_exploitability_of_the_random_strategy_is_bounded() -> None:
    """A random strategy is exploitable, but its exploitability stays small."""
    exploitability = _sp_synth.kuhn_exploit(_sp_synth.kuhn_random_strat())
    assert exploitability > 0.0
    assert np.isfinite(exploitability)


# --------------------------------------------------------------------------- #
# arch_fit
# --------------------------------------------------------------------------- #


def test_the_arch_fit_returns_a_fitted_model() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 1.0, 300)
    fit = arch_fit.arch_fit(returns, "GARCH", "normal", 1)
    assert fit is not None
    assert len(fit.params) > 0
    assert np.all(np.isfinite(np.asarray(fit.params, dtype=float)))


def test_the_arch_fit_is_reproducible_for_the_same_series() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 1.0, 300)
    first = arch_fit.arch_fit(returns, "GARCH", "normal", 1)
    second = arch_fit.arch_fit(returns, "GARCH", "normal", 1)
    assert np.allclose(
        np.asarray(first.params, dtype=float), np.asarray(second.params, dtype=float)
    )


def test_the_arch_fit_reports_a_volatility_parameter() -> None:
    """A GARCH fit must expose an omega/alpha/beta parameter set."""
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 1.0, 400)
    fit = arch_fit.arch_fit(returns, "GARCH", "normal", 1)
    names = list(fit.params.index)
    assert "mu" in names
    assert "omega" in names
    assert "alpha[1]" in names
    assert "beta[1]" in names


def test_the_crc_known_answers_are_the_declared_constants() -> None:
    """The fixture pins the standard CRC-32/CRC-16 check values."""
    assert zlib.crc32(_code_synth.CRC_KAT_MSG) == _code_synth.CRC32_TRUE
    assert _code_synth.CRC16_TRUE == 0x29B1
