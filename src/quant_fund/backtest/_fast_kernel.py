"""Optional numba-compiled inner loop for ``run_backtest_fast``.

Bit-identical contract: this kernel reproduces ``run_backtest`` /
``run_backtest_fast`` semantics statement-for-statement, including
CPython 3.12 ``sum()`` behaviour — exact-``float`` prefixes accumulate with
Neumaier compensation and the first ``np.float64`` term commits the
compensated partial and degrades the rest to a naive left fold (verified
empirically against CPython 3.12.14). ``np.float64`` propagation is tracked
via per-asset/per-scalar "np64" flags: values are IEEE-754 float64 either
way, so flags only choose the summation path, never the arithmetic.

The kernel is selected only when numba is importable; without numba the
pure-Python loop in ``fast_replay`` runs (identical outputs, slower).
"""

from __future__ import annotations

import numpy as np
import polars as pl

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised only on numba-less envs
    HAVE_NUMBA = False

    def njit(*_a, **_k):  # type: ignore[no-redef]
        def _deco(fn):
            return fn

        if len(_a) == 1 and callable(_a[0]) and not _k:
            return _a[0]
        return _deco


@njit(cache=True)
def _neumaier(v: np.ndarray) -> float:
    """CPython 3.12 float fast-path sum: Neumaier compensated accumulation."""
    hi = 0.0
    lo = 0.0
    for j in range(v.shape[0]):
        x = v[j]
        t = hi + x
        if abs(hi) >= abs(x):
            lo += (hi - t) + x
        else:
            lo += (x - t) + hi
        hi = t
    return hi + lo


@njit(cache=True)
def _csum(v: np.ndarray, f: np.ndarray) -> float:
    """``sum()`` over terms with per-term ``np.float64`` flags.

    Exact-float prefix -> Neumaier; first np.float64 term commits the
    compensated partial (hi + lo) and the remaining terms naive-fold.
    """
    n = v.shape[0]
    k0 = n
    for j in range(n):
        if f[j]:
            k0 = j
            break
    if k0 == n:
        return _neumaier(v)
    hi = 0.0
    lo = 0.0
    for j in range(k0):
        x = v[j]
        t = hi + x
        if abs(hi) >= abs(x):
            lo += (hi - t) + x
        else:
            lo += (x - t) + hi
        hi = t
    acc = hi + lo
    for j in range(k0, n):
        acc += v[j]
    return acc


@njit(cache=True)
def _order_costs(
    d: float,
    px: float,
    adv_d: float,
    vol_d: float,
    frictionless: bool,
    commission_bps: float,
    half_spread_bps: float,
    impact_y: float,
    bps_per_turnover: float,
) -> tuple[float, float, float, float]:
    # Caller validates ``d``/``px`` first — identical ordering to the
    # reference ``total_cost`` (validation precedes the frictionless return).
    if frictionless:
        return 0.0, 0.0, 0.0, 0.0
    nt = abs(d) * px
    comm = abs(nt) * commission_bps / 1e4
    spr = abs(nt) * half_spread_bps / 1e4
    part = nt / max(adv_d, 1e-12)
    imp = float(nt * impact_y * vol_d * np.sqrt(part))
    bpt = abs(nt) * bps_per_turnover / 1e4
    return comm, spr, imp, float(comm + spr + imp + bpt)


@njit(cache=True)
def replay_kernel(
    exec_px: np.ndarray,  # (T,A) f64 — exec source (open next_open / close)
    close_px: np.ndarray,  # (T,A) f64 — decision close + mark fallback
    ctr: np.ndarray,  # (T,A) f64 — preferred mark
    adv: np.ndarray,  # (T,A) f64
    vol: np.ndarray,  # (T,A) f64
    w_mat: np.ndarray,  # (T,A) f64 — carried target grid already applied
    market_vols: np.ndarray,  # (T,)  f64 — value only valid where has_mv
    has_mv: np.ndarray,  # (T,)  bool
    use_next_open: bool,
    ref_carries: bool,
    commission_bps: float,
    half_spread_bps: float,
    impact_y: float,
    bps_per_turnover: float,
    borrow_bps_per_year: float,
    participation_limit: float,
    frictionless: bool,
    stale_price_bars: int,
    max_order_notional: float,
    max_gross: float,
    max_net: float,
    max_name: float,
    max_participation: float,
    max_predicted_vol: float,
    kill_blocks: bool,
    initial_nav: float,
    navs_out: np.ndarray,  # (T,4) f64
    navs_t: np.ndarray,  # (T,) i64
    f_qty: np.ndarray,  # (M,) f64
    f_px: np.ndarray,  # (M,) f64
    f_fee: np.ndarray,  # (M,) f64
    f_spr: np.ndarray,  # (M,) f64
    f_imp: np.ndarray,  # (M,) f64
    f_asset: np.ndarray,  # (M,) i64
    f_et: np.ndarray,  # (M,) i64
    f_st: np.ndarray,  # (M,) i64
    f_dec: np.ndarray,  # (M,) f64
    f_decok: np.ndarray,  # (M,) bool
    stale_assets: np.ndarray,  # (A,) i64
    stale_ages: np.ndarray,  # (A,) i64
    stale_ever: np.ndarray,  # (A,) bool
    term_v: np.ndarray,  # (A,) f64 work buffer
    term_f: np.ndarray,  # (A,) bool work buffer
) -> tuple[int, int, int, int, int, int, float, float, float, int, int]:
    """Returns (status, n_navs, n_fills, rejects, cash_rejects, halts,
    commission_sum, spread_sum, impact_sum, order_seq, nstale).

    status: 0 ok (incl. nav<=0 early break), 1 stale-valuation (stale_* out
    params filled in the reference's two-pass order), 2 non-finite
    delta/price.
    """
    n_dates = exec_px.shape[0]
    n_assets = exec_px.shape[1]

    shares = np.zeros(n_assets, dtype=np.float64)
    share_np64 = np.zeros(n_assets, dtype=np.bool_)
    pos_seq = np.empty(n_assets, dtype=np.int64)
    npos = 0
    in_book = np.zeros(n_assets, dtype=np.bool_)
    last_mark = np.zeros(n_assets, dtype=np.float64)
    ever_marked = np.zeros(n_assets, dtype=np.bool_)
    mark_age = np.zeros(n_assets, dtype=np.int64)
    npv = np.zeros(n_assets, dtype=np.float64)
    marked_today = np.zeros(n_assets, dtype=np.bool_)

    cash = float(initial_nav)
    cash_np64 = False
    cost_comm = 0.0
    cost_spr = 0.0
    cost_imp = 0.0
    reject_count = 0
    cash_reject_count = 0
    halt_count = 0
    order_seq = 0
    n_navs = 0
    n_fills = 0

    last_i = n_dates - 1 if use_next_open else n_dates
    for i in range(last_i):
        exec_t = i + 1 if use_next_open else i

        # --- mark update ------------------------------------------------
        for a in range(n_assets):
            c_ok = np.isfinite(ctr[exec_t, a]) and ctr[exec_t, a] > 0
            cl_ok = np.isfinite(close_px[exec_t, a]) and close_px[exec_t, a] > 0
            if c_ok:
                last_mark[a] = ctr[exec_t, a]
                mark_age[a] = 0
                ever_marked[a] = True
                marked_today[a] = True
            elif cl_ok:
                last_mark[a] = close_px[exec_t, a]
                mark_age[a] = 0
                ever_marked[a] = True
                marked_today[a] = True
            else:
                mark_age[a] += 1
                marked_today[a] = False

        # --- stale-valuation fail-closed on held positions --------------
        # Two passes, matching the reference's dict comprehension order:
        # never-marked held names first (detail "unknown"), then over-limit.
        nstale = 0
        for j in range(npos):
            a = pos_seq[j]
            if abs(shares[a]) > 1e-12 and not ever_marked[a]:
                stale_assets[nstale] = a
                stale_ages[nstale] = mark_age[a]
                stale_ever[nstale] = False
                nstale += 1
        for j in range(npos):
            a = pos_seq[j]
            if abs(shares[a]) > 1e-12 and ever_marked[a] and mark_age[a] > stale_price_bars:
                stale_assets[nstale] = a
                stale_ages[nstale] = mark_age[a]
                stale_ever[nstale] = True
                nstale += 1
        if nstale:
            return (
                1,
                n_navs,
                n_fills,
                reject_count,
                cash_reject_count,
                halt_count,
                cost_comm,
                cost_spr,
                cost_imp,
                order_seq,
                nstale,
            )

        # --- nav at execution marks --------------------------------------
        for a in range(n_assets):
            e_ok = np.isfinite(exec_px[exec_t, a]) and exec_px[exec_t, a] > 0
            if e_ok:
                npv[a] = exec_px[exec_t, a]
            elif ever_marked[a]:
                npv[a] = last_mark[a]
            else:
                npv[a] = 0.0
        any_share_np64 = False
        for j in range(npos):
            a = pos_seq[j]
            term_v[j] = shares[a] * npv[a]
            term_f[j] = share_np64[a]
            if share_np64[a]:
                any_share_np64 = True
        pos_val = _csum(term_v[:npos], term_f[:npos])
        nav = cash + pos_val
        if nav <= 0:
            break
        nav_safe = max(nav, 1e-12)
        nav_np64 = cash_np64 or any_share_np64

        traded_turn = 0.0
        has = has_mv[i]
        mv = market_vols[i]
        mv_bad = has and (not np.isfinite(mv) or mv < 0.0)

        for a in range(n_assets):
            if not (np.isfinite(exec_px[exec_t, a]) and exec_px[exec_t, a] > 0):
                continue
            price = exec_px[exec_t, a]
            tw = w_mat[i, a] if (not ref_carries or marked_today[a]) else 0.0
            desired = tw * nav / price
            current = shares[a]
            delta = desired - current
            if abs(delta) * price < 1.0:
                continue
            adv_a = adv[exec_t, a]
            adv_eff = adv_a if np.isfinite(adv_a) and adv_a > 0 else 1.0
            vol_a = vol[exec_t, a]
            vol_eff = vol_a if np.isfinite(vol_a) and vol_a > 0 else 0.02
            delta_np64 = nav_np64 or share_np64[a]

            # ``total_cost`` validates before the frictionless early return —
            # a non-finite delta fails closed rather than trading.
            if not np.isfinite(delta) or not np.isfinite(price) or price <= 0:
                return (
                    2,
                    n_navs,
                    n_fills,
                    reject_count,
                    cash_reject_count,
                    halt_count,
                    cost_comm,
                    cost_spr,
                    cost_imp,
                    order_seq,
                    0,
                )
            comm, spr, imp, total = _order_costs(
                delta,
                price,
                adv_eff,
                vol_eff,
                frictionless,
                commission_bps,
                half_spread_bps,
                impact_y,
                bps_per_turnover,
            )
            max_qty = participation_limit * (adv_eff / price)
            if abs(delta) > max_qty > 0:
                delta = np.sign(delta) * max_qty
                delta_np64 = True
                comm, spr, imp, total = _order_costs(
                    delta,
                    price,
                    adv_eff,
                    vol_eff,
                    frictionless,
                    commission_bps,
                    half_spread_bps,
                    impact_y,
                    bps_per_turnover,
                )

            if kill_blocks:
                halt_count += 1
                continue

            # --- risk gate ----------------------------------------------
            current_w = current * npv[a] / nav_safe
            k = 0
            for b in range(n_assets):
                if b == a:
                    p = (current + delta) * npv[a]
                    fl = share_np64[a] or delta_np64
                else:
                    p = shares[b] * npv[b]
                    fl = share_np64[b]
                if p != 0.0:
                    term_v[k] = p
                    term_f[k] = fl
                    k += 1
            abs_v = np.empty(k, dtype=np.float64)
            for j in range(k):
                abs_v[j] = abs(term_v[j])
            gross_after = _csum(abs_v, term_f[:k]) / nav_safe
            net_after = _csum(term_v[:k], term_f[:k]) / nav_safe
            participation = abs(delta) * price / max(adv_eff, 1e-12)
            order_seq += 1
            gate_vol = mv if has else vol_eff
            rejected = (
                mv_bad
                or not np.isfinite(nav)
                or not np.isfinite(price)
                or not np.isfinite(current_w)
                or not np.isfinite(gross_after)
                or not np.isfinite(net_after)
                or not np.isfinite(participation)
                or not np.isfinite(vol_eff)
                or nav <= 0.0
                or price <= 0.0
                or gross_after < 0.0
                or participation < 0.0
                or vol_eff < 0.0
                or not np.isfinite(delta)
                or delta == 0.0
                or abs(delta) * price > max_order_notional
                or abs(current_w + (delta * price) / nav_safe) > max_name
                or gross_after > max_gross
                or abs(net_after) > max_net
                or participation > max_participation
                or gate_vol > max_predicted_vol
            )
            if rejected:
                reject_count += 1
                continue

            notional = delta * price
            if delta > 0 and cash < notional + total:
                cash_reject_count += 1
                continue
            cash -= notional + total
            if delta_np64:
                cash_np64 = True
            if not in_book[a]:
                in_book[a] = True
                pos_seq[npos] = a
                npos += 1
            shares[a] = current + delta
            if delta_np64:
                share_np64[a] = True
            traded_turn += abs(notional) / nav_safe
            cost_comm += comm
            cost_spr += spr
            cost_imp += imp
            f_qty[n_fills] = delta
            f_px[n_fills] = price
            f_fee[n_fills] = comm
            f_spr[n_fills] = spr
            f_imp[n_fills] = imp
            f_asset[n_fills] = a
            f_et[n_fills] = exec_t
            f_st[n_fills] = i
            d_ok = np.isfinite(close_px[i, a]) and close_px[i, a] > 0
            f_decok[n_fills] = d_ok
            f_dec[n_fills] = close_px[i, a] if d_ok else 0.0
            n_fills += 1

        # --- mark to close -----------------------------------------------
        for j in range(npos):
            a = pos_seq[j]
            term_v[j] = shares[a] * last_mark[a]
            term_f[j] = share_np64[a]
        pos_close = _csum(term_v[:npos], term_f[:npos])
        nav_close = cash + pos_close

        any_short_np64 = False
        for j in range(npos):
            a = pos_seq[j]
            term_v[j] = (-shares[a] if shares[a] < 0.0 else 0.0) * last_mark[a]
            term_f[j] = share_np64[a] and shares[a] <= 0.0
            if term_f[j]:
                any_short_np64 = True
        short_notional = _csum(term_v[:npos], term_f[:npos])

        for j in range(npos):
            a = pos_seq[j]
            term_v[j] = abs(shares[a] * last_mark[a])
            term_f[j] = share_np64[a]
        gross_v = _csum(term_v[:npos], term_f[:npos])
        net_v = pos_close

        borrow = short_notional * (borrow_bps_per_year / 1e4) / 252.0
        if not frictionless:
            cash -= borrow
            if any_short_np64:
                cash_np64 = True
            nav_close -= borrow
        navs_out[n_navs, 0] = nav_close
        navs_out[n_navs, 1] = gross_v / max(nav_close, 1e-12)
        navs_out[n_navs, 2] = net_v / max(nav_close, 1e-12)
        navs_out[n_navs, 3] = traded_turn
        navs_t[n_navs] = exec_t
        n_navs += 1

    return (
        0,
        n_navs,
        n_fills,
        reject_count,
        cash_reject_count,
        halt_count,
        cost_comm,
        cost_spr,
        cost_imp,
        order_seq,
        0,
    )


def kernel_or_none():
    """Return the compiled kernel when numba is importable, else ``None``."""
    return replay_kernel if HAVE_NUMBA else None


def replay_driver(
    *,
    exec_px: np.ndarray,
    close_px: np.ndarray,
    ctr: np.ndarray,
    adv: np.ndarray,
    vol: np.ndarray,
    w_mat: np.ndarray,
    market_vols: list | None,
    use_next_open: bool,
    ref_carries: bool,
    costs_cfg,
    gate,
    kill_blocks: bool,
    initial_nav: float,
    dates: list,
    sids: list[str],
):
    """Allocate buffers, run the compiled kernel, materialise result frames.

    Returns ``(navs, fill_rows, cost_sum, reject_count, cash_reject_count,
    halt_count, order_seq)`` where ``navs``/``fill_rows`` are polars frames
    with identical columns/dtypes to ``_build_result``'s dict transpose — or
    ``None`` when numba is unavailable. Raises ``StaleValuationError`` /
    ``ValueError`` with the reference's exact messages on fail-closed paths.
    """
    if not HAVE_NUMBA:
        return None

    n_dates = exec_px.shape[0]
    n_assets = exec_px.shape[1]
    mv_arr = np.full(n_dates, np.nan)
    has_mv = np.zeros(n_dates, dtype=np.bool_)
    if market_vols is not None:
        for j, v in enumerate(market_vols):
            if v is not None:
                has_mv[j] = True
                mv_arr[j] = float(v)

    max_fills = max(n_dates * n_assets, 1)
    navs_out = np.empty((n_dates, 4))
    navs_t = np.empty(n_dates, dtype=np.int64)
    f_qty = np.empty(max_fills)
    f_px = np.empty(max_fills)
    f_fee = np.empty(max_fills)
    f_spr = np.empty(max_fills)
    f_imp = np.empty(max_fills)
    f_asset = np.empty(max_fills, dtype=np.int64)
    f_et = np.empty(max_fills, dtype=np.int64)
    f_st = np.empty(max_fills, dtype=np.int64)
    f_dec = np.empty(max_fills)
    f_decok = np.empty(max_fills, dtype=np.bool_)
    stale_assets = np.empty(n_assets, dtype=np.int64)
    stale_ages = np.empty(n_assets, dtype=np.int64)
    stale_ever = np.empty(n_assets, dtype=np.bool_)
    term_v = np.empty(n_assets)
    term_f = np.empty(n_assets, dtype=np.bool_)

    (
        status,
        n_navs,
        n_fills,
        reject_count,
        cash_reject_count,
        halt_count,
        cost_comm,
        cost_spr,
        cost_imp,
        order_seq,
        nstale,
    ) = replay_kernel(
        np.ascontiguousarray(exec_px),
        np.ascontiguousarray(close_px),
        np.ascontiguousarray(ctr),
        np.ascontiguousarray(adv),
        np.ascontiguousarray(vol),
        np.ascontiguousarray(w_mat),
        mv_arr,
        has_mv,
        use_next_open,
        ref_carries,
        float(costs_cfg.commission_bps),
        float(costs_cfg.half_spread_bps),
        float(costs_cfg.impact_y),
        float(costs_cfg.bps_per_turnover),
        float(costs_cfg.borrow_bps_per_year),
        float(costs_cfg.participation_limit),
        bool(costs_cfg.frictionless),
        int(gate.stale_price_bars),
        float(gate.max_order_notional),
        float(gate.max_gross),
        float(gate.max_net),
        float(gate.max_name),
        float(gate.max_participation),
        float(gate.max_predicted_vol),
        bool(kill_blocks),
        float(initial_nav),
        navs_out,
        navs_t,
        f_qty,
        f_px,
        f_fee,
        f_spr,
        f_imp,
        f_asset,
        f_et,
        f_st,
        f_dec,
        f_decok,
        stale_assets,
        stale_ages,
        stale_ever,
        term_v,
        term_f,
    )

    if status == 1:
        from quant_fund.backtest.engine import StaleValuationError

        details = ", ".join(
            f"{sids[stale_assets[k]]}={int(stale_ages[k]) if stale_ever[k] else 'unknown'}"
            for k in range(nstale)
        )
        raise StaleValuationError(
            "held position valuation is stale beyond the configured limit: " + details
        )
    if status == 2:
        raise ValueError("quantity must be finite and price must be finite and positive")

    # Column-oriented frames — same columns/dtypes/order as the interpreted
    # path's dict transpose in ``_build_result`` (event_time -> Datetime from
    # the same datetime objects, f64 value columns, decision_price nullable).
    navs: pl.DataFrame | list = (
        pl.DataFrame(
            {
                "event_time": [dates[int(t)] for t in navs_t[:n_navs]],
                "nav": navs_out[:n_navs, 0],
                "gross": navs_out[:n_navs, 1],
                "net": navs_out[:n_navs, 2],
                "turnover": navs_out[:n_navs, 3],
            }
        )
        if n_navs
        else []
    )
    if n_fills:
        # NaN is a float value, not a polars null — the dict path stores
        # ``None`` for fills whose decision bar lacked a valid close, so map
        # masked decision prices to real nulls (``drop_nulls`` parity).
        dec_vals = np.where(f_decok[:n_fills], f_dec[:n_fills], np.nan)
        dec_ser = pl.Series("decision_price", dec_vals).fill_nan(None)
        fill_rows: pl.DataFrame | list = pl.DataFrame(
            {
                "fill_time": [dates[int(t)] for t in f_et[:n_fills]],
                "signal_time": [dates[int(t)] for t in f_st[:n_fills]],
                "security_id": pl.Series(np.asarray(sids)[f_asset[:n_fills]]),
                "quantity": f_qty[:n_fills],
                "price": f_px[:n_fills],
                "fee": f_fee[:n_fills],
                "spread_cost": f_spr[:n_fills],
                "impact_cost": f_imp[:n_fills],
                "decision_price": dec_ser,
            }
        )
    else:
        fill_rows = []
    cost_sum = {"commission": cost_comm, "spread": cost_spr, "impact": cost_imp}
    return (
        navs,
        fill_rows,
        cost_sum,
        reject_count,
        cash_reject_count,
        halt_count,
        order_seq,
    )
