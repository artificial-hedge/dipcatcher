"""Latent-skill rating systems: Elo, Glicko-2, and Bradley-Terry.

Pairwise-comparison outcomes (win/loss/draw over ``(i, j, outcome)`` logs)
are mapped to latent player strengths. Three estimators:

- **Elo** (Elo 1978): logistic expected-score update with step ``K``.
- **Glicko-2** (Glickman): per-player ``(mu, phi, sigma)`` state — rating,
  rating deviation, volatility — updated through the closed-form
  steps of the Glicko-2 system.
- **Bradley-Terry** (Zermelo/Bradley-Terry): MLE of strengths
  ``w_i`` with ``P(i beats j) = w_i / (w_i + w_j)`` via Hunter's MM
  minorization iteration.

All three order players on an interval scale; Glicko-2 additionally
carries a per-player uncertainty that shrinks with evidence.

Functions
---------
- :func:`elo_update`, :func:`elo_fit` — Elo.
- :func:`glicko2_update`, :func:`glicko2_fit` — Glicko-2.
- :func:`bradley_terry_mle`, :func:`predict_matrix` — Bradley-Terry.
- :func:`synth_games` — synthetic tournament generator.
- :func:`bench_skill_ratings` — SYNTHETIC telemetry blob.

References
----------
- Elo (1978). *The Rating of Chessplayers, Past and Present* (book).
- Glickman (1999/2013). The Glicko-2 system (Boston University technical
  report — the update equations implemented verbatim).
- Zermelo (1929); Bradley & Terry (1952). Rank analysis of incomplete
  block designs. *Biometrika* 39.
- Hunter (2004). MM algorithms for generalized Bradley-Terry models.
  *Annals of Statistics* 32 — the fixed-point iteration used here.
- Luce (1959). *Individual Choice Behavior* — choice axiom under BT.

Honesty
-------
Benches run on SYNTHETIC tournaments only. Skill ratings order latent
abilities under a stated outcome model — they are not performance or
PnL claims; never emit headline P&L/Sharpe tokens.

Composition notes
-----------------
- Ranking-family evaluators consume pairwise outcomes through the
  ``ranking`` scorecard family; this module supplies the latent-skill
  layer (Elo/Glicko-2/BT) that a tournament of model candidates can use.
- ``metrics/bootstrap.py`` / ``metrics/stationary_bootstrap.py``:
  uncertainty over rating estimates can be built by resampling the game
  log — composition point, not implemented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _v_outcomes(outcomes: IntArray, n_players: int) -> IntArray:
    g = np.asarray(outcomes, dtype=np.int64)
    if g.ndim != 2 or g.shape[1] != 3 or g.shape[0] < 1:
        raise ValueError("outcomes must be (n_games, 3): [i, j, outcome]")
    if g[:, 0].min() < 0 or g[:, 1].min() < 0:
        raise ValueError("player indices must be >= 0")
    if g[:, 0].max() >= n_players or g[:, 1].max() >= n_players:
        raise ValueError("player index out of range")
    if not np.isin(g[:, 2], [0, 1]).all():
        raise ValueError("outcome must be 0 (j wins) or 1 (i wins); draws unsupported")
    return g


def elo_update(
    ratings: FloatArray,
    i: int,
    j: int,
    outcome: float,
    k: float = 32.0,
    base: float = 10.0,
    scale: float = 400.0,
) -> FloatArray:
    """One Elo update: returns a NEW rating vector (input unchanged)."""
    r = np.asarray(ratings, dtype=float).copy()
    if k <= 0 or scale <= 0 or base <= 1:
        raise ValueError("need k>0, scale>0, base>1")
    if not 0.0 <= outcome <= 1.0:
        raise ValueError("outcome must be in [0, 1]")
    e_i = 1.0 / (1.0 + base ** ((r[j] - r[i]) / scale))
    e_j = 1.0 - e_i
    r[i] += k * (outcome - e_i)
    r[j] += k * ((1.0 - outcome) - e_j)
    return r


def elo_fit(
    outcomes: IntArray,
    n_players: int,
    k: float = 32.0,
    passes: int = 1,
    seed: int = 0,
) -> FloatArray:
    """Iterate Elo over a game log; ``passes`` replays the log."""
    if n_players < 2 or passes < 1:
        raise ValueError("need n_players >= 2, passes >= 1")
    g = _v_outcomes(outcomes, n_players)
    r = np.full(n_players, 1500.0)
    rng = np.random.default_rng(seed)
    order = np.arange(g.shape[0])
    for _ in range(passes):
        perm = rng.permutation(order)
        for t in perm:
            r = elo_update(r, int(g[t, 0]), int(g[t, 1]), float(g[t, 2]), k=k)
    return r


_Q = math.log(10.0) / 400.0


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * _Q * _Q * phi * phi / (math.pi * math.pi))


def _exp_score(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def glicko2_update(
    rating: float,
    rd: float,
    vol: float,
    opp_rating: float,
    opp_rd: float,
    outcome: float,
    tau: float = 0.5,
) -> tuple[float, float, float]:
    """Single-opponent Glicko-2 update → ``(mu', phi', sigma')`` scale.

    ``rating``/``rd`` are on the 1500/350 scale; internally converted to
    ``mu = (rating-1500)*q`` / ``phi = rd*q``. Follows the 8-step
    Glickman algorithm including the Illinois-method volatility root
    solve.
    """
    if rd <= 0 or opp_rd <= 0 or tau <= 0:
        raise ValueError("need rd>0, opp_rd>0, tau>0")
    if not 0.0 <= outcome <= 1.0:
        raise ValueError("outcome must be in [0, 1]")
    mu = (rating - 1500.0) * _Q
    phi = rd * _Q
    sig = vol
    mu_j = (opp_rating - 1500.0) * _Q
    phi_j = opp_rd * _Q
    g_j = _g(phi_j)
    e = _exp_score(mu, mu_j, phi_j)
    # Glickman step 3/4: v and delta carry no q factor — q enters only
    # in the mu update (step 7)
    v = 1.0 / (g_j * g_j * e * (1.0 - e))
    delta = v * g_j * (outcome - e)
    a = math.log(sig * sig)
    eps = 1e-6

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta * delta - phi * phi - v - ex)
        den = 2.0 * (phi * phi + v + ex) ** 2
        return num / den - (x - a) / (tau * tau)

    big_a = a
    if delta * delta > phi * phi + v:
        big_b = math.log(delta * delta - phi * phi - v)
    else:
        k_iter = 1
        while f(a - k_iter * tau) < 0:
            k_iter += 1
        big_b = a - k_iter * tau
    f_a, f_b = f(big_a), f(big_b)
    for _ in range(100):
        c = big_a + (big_a - big_b) * f_a / (f_b - f_a)
        f_c = f(c)
        if f_c * f_b < 0:
            big_a, f_a = big_b, f_b
        else:
            f_a /= 2.0
        big_b, f_b = c, f_c
        if abs(big_b - big_a) < eps:
            break
    sig_new = math.exp(big_a / 2.0)
    phi_star = math.sqrt(phi * phi + sig_new * sig_new)
    phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_new = mu + _Q * phi_new * phi_new * g_j * (outcome - e)
    return (1500.0 + mu_new / _Q, phi_new / _Q, sig_new)


@dataclass(frozen=True)
class Glicko2State:
    """Per-player Glicko-2 state after processing a game log."""

    rating: FloatArray  # ~1500 scale
    rd: FloatArray  # rating deviation (~350 start)
    vol: FloatArray  # volatility (~0.06)


def glicko2_fit(
    outcomes: IntArray,
    n_players: int,
    tau: float = 0.5,
    seed: int = 0,
) -> Glicko2State:
    """Glicko-2 over a game log — one rating-period update.

    The Glicko-2 system aggregates ALL of a player's games in a rating
    period into a single ``(v, delta)`` pair before the phi/sigma
    updates — the per-game application of ``glicko2_update`` is kept
    for single-opponent periods, but applying it per-game over a whole
    log inflates ``phi*`` each step and RD never shrinks. Here the
    whole log is treated as one period (the documented Glickman
    procedure).
    """
    if n_players < 2:
        raise ValueError("need n_players >= 2")
    g = _v_outcomes(outcomes, n_players)
    rating = np.full(n_players, 1500.0)
    rd = np.full(n_players, 350.0)
    vol = np.full(n_players, 0.06)
    mu = (rating - 1500.0) * _Q
    phi = rd * _Q
    # seed retained in signature for API stability; the period update is
    # order-invariant (all games aggregate into one (v, delta) pair)
    del seed
    # accumulate v and delta per player over the whole period
    # (Glickman steps 3/4 — no q factor in the aggregates)
    inv_v = np.zeros(n_players)  # sum g^2 e(1-e)  -> v = 1/inv_v
    dsum = np.zeros(n_players)  # sum g (s - e)
    for t in range(g.shape[0]):
        i, j, s = int(g[t, 0]), int(g[t, 1]), float(g[t, 2])
        for p, q_idx, out in ((i, j, s), (j, i, 1.0 - s)):
            g_j = _g(phi[q_idx])
            e = 1.0 / (1.0 + math.exp(-g_j * (mu[p] - mu[q_idx])))
            inv_v[p] += g_j * g_j * e * (1.0 - e)
            dsum[p] += g_j * (out - e)
    v = 1.0 / np.maximum(inv_v, 1e-300)
    delta = v * dsum
    active = inv_v > 0
    for p in range(n_players):
        if not active[p]:
            continue
        # volatility solve identical to glicko2_update's Illinois method
        a = math.log(vol[p] * vol[p])
        d2 = delta[p] * delta[p]
        p2 = phi[p] * phi[p]
        vv = v[p]

        def f(x: float, d2: float = d2, p2: float = p2, vv: float = vv, a: float = a) -> float:
            ex = math.exp(x)
            return ex * (d2 - p2 - vv - ex) / (2.0 * (p2 + vv + ex) ** 2) - (x - a) / (tau * tau)

        big_a = a
        if d2 > p2 + vv:
            big_b = math.log(d2 - p2 - vv)
        else:
            k_it = 1
            while f(a - k_it * tau) < 0:
                k_it += 1
            big_b = a - k_it * tau
        f_a, f_b = f(big_a), f(big_b)
        for _ in range(100):
            c = big_a + (big_a - big_b) * f_a / (f_b - f_a)
            f_c = f(c)
            if f_c * f_b < 0:
                big_a, f_a = big_b, f_b
            else:
                f_a /= 2.0
            big_b, f_b = c, f_c
            if abs(big_b - big_a) < 1e-6:
                break
        vol[p] = math.exp(big_a / 2.0)
        phi_star = math.sqrt(p2 + vol[p] * vol[p])
        phi[p] = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / vv)
        # step 7: mu' = mu + q phi'^2 sum_j g_j (s_j - e_j)
        mu[p] += _Q * phi[p] * phi[p] * dsum[p]
    rating = 1500.0 + mu / _Q
    rd = phi / _Q
    return Glicko2State(rating=rating, rd=rd, vol=vol)


def bradley_terry_mle(
    outcomes: IntArray,
    n_players: int,
    n_iter: int = 200,
    tol: float = 1e-10,
) -> FloatArray:
    """Bradley-Terry strength MLE via Hunter's MM iteration.

    ``w_i <- wins_i / sum_{games i played} 1/(w_i + w_j)`` — monotone
    ascent on the Luce-choice log-likelihood. Players with zero games
    are pinned at eps (documented degenerate handling).
    """
    if n_players < 2 or n_iter < 1 or tol <= 0:
        raise ValueError("need n_players >= 2, n_iter >= 1, tol > 0")
    g = _v_outcomes(outcomes, n_players)
    wins = np.zeros(n_players)
    played = np.zeros(n_players)
    pairs_i = g[:, 0]
    pairs_j = g[:, 1]
    np.add.at(wins, pairs_i, g[:, 2])
    np.add.at(wins, pairs_j, 1 - g[:, 2])
    np.add.at(played, pairs_i, 1)
    np.add.at(played, pairs_j, 1)

    w = np.full(n_players, 1.0 / n_players)
    for _ in range(n_iter):
        denom_i = np.zeros(n_players)
        denom_j = np.zeros(n_players)
        for t in range(g.shape[0]):
            i, j = pairs_i[t], pairs_j[t]
            s = w[i] + w[j]
            denom_i[i] += 1.0 / s
            denom_j[j] += 1.0 / s
        w_new = np.empty(n_players)
        active = played > 0
        w_new[active] = wins[active] / np.maximum(denom_i[active] + denom_j[active], 1e-300)
        w_new[~active] = 1e-300
        # floor: zero-win players approach 0 but loglik/predict need w > 0
        w_new = np.maximum(w_new, 1e-300)
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def bradley_terry_loglik(outcomes: IntArray, w: FloatArray) -> float:
    """BT log-likelihood of a strength vector on a game log."""
    g = np.asarray(outcomes, dtype=np.int64)
    wv = np.asarray(w, dtype=float)
    if g.ndim != 2 or g.shape[1] != 3 or g.shape[0] < 1:
        raise ValueError("outcomes must be (n_games, 3)")
    if wv.ndim != 1 or (wv <= 0).any():
        raise ValueError("w must be positive 1-d")
    ll = 0.0
    for t in range(g.shape[0]):
        i, j, o = int(g[t, 0]), int(g[t, 1]), int(g[t, 2])
        p = wv[i] / (wv[i] + wv[j])
        ll += math.log(p) if o == 1 else math.log(1.0 - p)
    return ll


def predict_matrix(w: FloatArray) -> FloatArray:
    """BT pairwise win-probability matrix ``P[i, j] = w_i/(w_i+w_j)``."""
    wv = np.asarray(w, dtype=float)
    if wv.ndim != 1 or wv.size < 2 or (wv <= 0).any():
        raise ValueError("w must be a positive 1-d vector")
    s = wv[:, None] + wv[None, :]
    out = np.asarray(wv[:, None] / s, dtype=np.float64)
    np.fill_diagonal(out, 0.5)
    return out


def _spearman(a: FloatArray, b: FloatArray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(ra @ rb)
    nrm = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return d / nrm if nrm > 0 else 0.0


def synth_games(
    n_players: int,
    n_games: int,
    spread: float = 2.0,
    seed: int = 0,
) -> tuple[IntArray, FloatArray]:
    """Synthetic tournament: true strengths ~ N(0, spread), games ~ BT.

    Returns ``(outcomes, true_strengths)`` where outcomes are
    ``(n_games, 3)`` rows ``[i, j, outcome]`` and outcome=1 means i won.
    """
    if n_players < 2 or n_games < n_players or spread <= 0:
        raise ValueError("need n_players>=2, n_games>=n_players, spread>0")
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, spread, n_players)
    w = np.exp(theta - theta.max())
    w /= w.sum()
    i_idx = rng.integers(0, n_players, n_games)
    j_idx = rng.integers(0, n_players, n_games)
    same = i_idx == j_idx
    j_idx[same] = (j_idx[same] + 1) % n_players
    p = w[i_idx] / (w[i_idx] + w[j_idx])
    out = (rng.random(n_games) < p).astype(np.int64)
    outcomes = np.stack([i_idx, j_idx, out], axis=1).astype(np.int64)
    return outcomes, theta


def bench_skill_ratings(seed: int = 20260205) -> dict[str, float]:
    """SYNTHETIC bench: rating-recovery quality on a seeded tournament."""
    out: dict[str, float] = {}
    n_players, n_games = 8, 400
    outcomes, theta = synth_games(n_players, n_games, spread=2.0, seed=seed)
    true_order = np.argsort(theta)

    w = bradley_terry_mle(outcomes, n_players)
    out["synthetic_spearman_recovery"] = _spearman(np.log(w + 1e-300), theta)
    out["synthetic_top1_hit"] = float(np.argmax(w) == int(true_order[-1]))
    ll_uniform = bradley_terry_loglik(outcomes, np.full(n_players, 1.0 / n_players))
    ll_hat = bradley_terry_loglik(outcomes, w)
    out["synthetic_bt_loglik_improve"] = float(ll_hat - ll_uniform)

    r = elo_fit(outcomes, n_players, k=32.0, passes=3, seed=seed)
    out["synthetic_elo_spearman"] = _spearman(r, theta)

    g2 = glicko2_fit(outcomes, n_players, seed=seed)
    out["synthetic_glicko_spearman"] = _spearman(g2.rating, theta)
    out["synthetic_glicko_rd_shrink"] = float(1.0 - g2.rd.mean() / 350.0)

    # determinism across identical calls
    w2 = bradley_terry_mle(outcomes, n_players)
    out["synthetic_determinism"] = float(np.allclose(w, w2))
    return out
