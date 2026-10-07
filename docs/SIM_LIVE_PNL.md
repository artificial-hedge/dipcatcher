# Simulated Live-PnL — `sim-live` lane

**SIMULATED ONLY — no live-PnL claim.** Every number below is a causal
replay of real collected Binance bars through the paper loop /
`run_backtest` with modeled fees (2bps commission + 3bps half-spread +
impact, 10% participation cap), the risk gate, and the kill switch. No
broker connectivity exists on this path.

Pipeline: causal quantile panels (`fhs|evt|garch_t|egarch_l|empirical|
ewma_emp{,94,99}|vincent(a+b)|agree(a,b)`, `@hN` horizon variants) →
`QuantilePolicy` (mu-bps or edge-z gate with hysteresis, `edge`/`risk`
sizing, book-vol target, tail gate, entry/exit persistence, leadership
gate, breadth-scaled gross, signal-Sharpe meta-gate, accel gate,
rebalance cadence, market-disp breaker, top-k concentration, target
EWMA, deadband carry, caps) → sparse weight panel → event loop
(next-open fills). Receipts in `.dsh-24x7/lane-simlive/`.

## Headline book (daily, 5 majors, 2020-08→2026-09, shared calendar)

`ewma_emp` long-flat, `gate_on=edge z≥0.15`, `sizing=risk`
(`w = κ·edge/σ`, κ=0.15), deadband 1%, name cap 25%, gross ≤1.0:

| measure | bench (`run_backtest`) | paper loop (ledger) |
|---|---|---|
| total return | +2,628.8% | +2,494.2% |
| Sharpe (simulated) | +1.785 | +1.769 |
| max drawdown | −22.1% | −22.2% |
| ann vol | 33.4% | 33.2% |
| fills | 1,721 | 1,721 |

Second flagship — `vincent(ew94+97+99)` + xp3 + bvt=0.03 g2 (no leader):
paper loop +2,769.8% / SR 1.850 / DD −27.0% / 1,947 fills / 1,014 gate
rejects / 0 halts vs bench +2,752.5% / SR 1.849 — bench↔loop parity
holds within participation-cap noise on the leveraged book too.

Third flagship — full stack `vincent(3-lam)` + xp3 + **BTC-lead** +
bvt=0.03 g2: paper loop **+2,923.5% / SR 1.970 / DD −27.0%** / 1,893
fills / 857 gate rejects / 0 halts vs bench +2,905.2% / SR 1.970.

Benchmarks on identical bars/costs: equal-weight buy-hold 5 majors
+2,201% at Sharpe 1.16, **maxDD −72.3%**, vol 59.5%. The book beats
buy-hold on Sharpe (1.79 vs 1.16) and drawdown (−22% vs −72%), not on
raw return scale.

## Out-of-sample (panels on full history; book runs last 800 days only)

| book | OOS return | OOS Sharpe | OOS maxDD |
|---|---|---|---|
| **xp3 + BTC-lead + breadth-gross** | **+50.5%** | **+1.04** | −15.2% |
| **vincent(3-lam) + xp3 + BTC-lead + bvt** | +49.5% | +1.01 | **−12.8%** |
| champion + xp=3 + BTC-lead | +51.1% | +1.05 | −15.3% |
| ewriskz (champion) | +45.4% | +1.03 | −12.2% |
| vincent(3-lam) + xp=3 + bvt=0.03 | +48.2% | +0.99 | −12.8% |
| champion + xp=3 | +46.5% | +0.97 | −15.3% |
| BTC-lead only | +40.4% | +0.97 | −12.2% |
| ewma_emp mu-gate 10bps | +17.3% | +1.05 | −4.9% |
| vincent(ewma_emp+empirical) risk | +25.7% | +0.89 | −9.3% |
| persist=2 + bvt=0.02 | +30.0% | +0.81 | −14.5% |

In-sample Sharpe 1.79 → OOS 1.03: real degradation, disclosed. The
signal survives but does not replicate the 2020-21 regime's magnitude.

## Return/drawdown frontier (bench, same bars/costs)

`book_vol_target` scales the whole book toward a per-bar vol target
(both directions; gross cap bounds the upside). `persist_bars` requires
consecutive gate-passing dates before (re-)entry.

| config | return | Sharpe | maxDD | ann vol |
|---|---|---|---|---|
| ewma_emp@h5 (5d horizon) | +3,486.1% | +1.62 | −39.8% | 41.7% |
| **xp3 + BTC-lead + breadth-gross** | **+3,309.8%** | +1.85 | −27.0% | 34.4% |
| **vincent(3-lam) + xp3 + BTC-lead + bvt=0.03 g2** | +2,905.2% | **+1.97** | −27.1% | 30.7% |
| champion + xp=3 + BTC-lead | +2,942.1% | +1.71 | −27.7% | 36.5% |
| champion + xp=3 | +2,882.4% | +1.71 | −30.7% | 36.3% |
| vincent(3-lam) + xp=3 + bvt=0.03 g2 | +2,752.5% | +1.85 | −27.1% | 32.5% |
| champion (no bvt) | +2,628.8% | +1.79 | −22.1% | 33.4% |
| vincent(ew94+97+99) bvt=0.03 g2 | +1,543.7% | +1.83 | −21.9% | 27.1% |
| bvt=0.02 | +672.8% | +1.82 | −16.6% | 19.4% |
| persist=2 + bvt=0.02 | +630.2% | +1.81 | **−13.2%** | 19.0% |
| bvt=0.015 | +447.7% | +1.85 | **−13.1%** | 15.7% |

Vol targeting is a clean efficient frontier — Sharpe holds ~1.8 while
drawdown halves at bvt 0.015–0.02; chasing return past ~1,500% costs
drawdown faster than it buys Sharpe.

`exit_persist` (slow exit — hold through a single gate-failing bar)
lifts both return and Sharpe at scale: xp=3 on the vincent blend is the
best risk-adjusted large book (SR 1.85), at the cost of deeper DD
(−27.1%). `leader_sid=BTCUSDT` (alts trade only while BTC's edge > 0)
stacks on xp=3 for the best raw return, **+2,942%**, and the best OOS
slice yet (+51.1% / SR 1.045 / DD −15.3%). `top_k` concentration
(2–3 names) lowers return and Sharpe — diversification across the
5-name book is load-bearing. `@hN` multi-horizon specs amplify edge by
~√N but *only* through a looser effective gate — at matched selectivity
(z·√N) h3/h5/h10 are all worse than the daily champion (SR ≤1.70), so
the horizon axis is disclosed as rejected. `mkt_edge_min` (flat book
when mean cross-asset edge sags) and `w_alpha` (target EWMA) are both
~Sharpe-neutral: the edge filter cuts into trend profits, and smoothing
inflates turnover 6× (10.6k fills) without DD benefit — rejected.
`breadth_gross` (gross cap × fraction of names with positive edge)
*stacks*: on xp3+BTC-lead it lifts return to +3,310% at SR 1.85 —
breadth-proportional exposure is the rare soft market filter that pays.
Rejected in the same sweep: `rebal_every` cadence (SR ≤1.75, turnover
down but exposure timing worse), `meta_min` signal-Sharpe gate (DD cut
but lower return; meta=0.5 starves the book), `accel_min` (negative),
`gate_out` hysteresis (SR 1.39 — loose exits hold losers).

## Rejected experiments (honest negatives)

- **`tail_gate`** (enter only when the forecast's own adverse-tail bound
  is benign): too strict — crypto left tails are fat; kills the book.
- **`mkt_disp_cut`** (market-wide vol breaker; flat when median disp
  > cut): every cut level *lowers* Sharpe — the biggest trend-follow
  gains occur inside high-vol regimes. Feature retained, disclosed.
- **Leverage gross 2–3× without vol targeting**: +2,938% at SR 1.46 /
  DD −42.6% in-sample; **OOS collapses to SR 0.16** — leverage amplifies
  chop, not edge.
- **4h interval**: costs × 2,191 bars/yr compress the edge to +10.8%
  (SR 0.60) at best.

## Regime dependence (the honest limit)

| window | book | Sharpe |
|---|---|---|
| 2020-08→2026-09 (5 majors) | champion | 1.79 |
| 2020-08→2026-09 (5 majors) | **vin3+xp3+BTC-lead+bvt** | **1.97** |
| last 800d only | champion | 1.03 |
| last 800d only | vin3+xp3+BTC-lead+bvt | 1.01 |
| last 500d only | *all variants* | 0.2–0.85 |
| 2018-05→2026-09 (BTC/ETH/XRP, 8.3y) | champion | ~1.01 |
| 2018-05→2026-09 (3 majors) | **xp3+BTC-lead+bg** | **1.14** |
| 998d breadth book (9 majors) | champion | 1.04 |

The edge is real but regime-dependent: ~1.0 Sharpe outside the trending
2020-24 regime — the +2,900% headline is regime compounding, not a
universal multiplier. The full stack keeps ~1.1 Sharpe over 8.3y
bear-inclusive history (vs 1.01 for the plain champion), so the layers
generalize — but no configuration escapes the chop regime.

**Rejected as regime-fitting**: `leader_edge_min > 0` looked great
in-sample (SR 2.07 at le=0.10) but folds disagree — le=0 wins 800d,
le=0.15 wins 500d, le=0.20 wins 1200d. No stable optimum → pure noise
axis; le=0 kept. `fund_cut` (BTC 7d funding-crowding breaker) is
Sharpe-negative at every level (p99 cut → SR 1.27): crowded-long
regimes carry the biggest trends. Same verdict shape as the vol
breaker — the signal's best days are exactly the scary ones.

## Chop-regime robustness sweep — PRE-REGISTRATION (first written 2026-10-07T16:47+05:30)

**PRE-REGISTRATION — first written 2026-10-07T16:47+05:30 (Asia/Calcutta),
before any sweep run.** Provenance (three checkable anchors, plus one
disclosed disagreement and one disclosed gap):

1. **Sweep driver (durable anchor).** `research/chop_robustness/run_sweep.py`
   encodes exactly this declaration — the mechanisms and their single fixed
   parameterizations, the four windows, and the decision rules — and has
   filesystem mtime **2026-10-07T16:50:24+05:30**, sha256
   `c8dcaddae5adc09965fee5ecf4e0f026cc273d7d7873a8b5b6499519ca833740`.
   It predates every outcome.
2. **Sweep receipts (outcome times/hashes).** All four window receipts were
   written strictly after: W1 `fca29eb5beac1ede9ff7be84a1cb240df8cb7dd50c5171539229841deb59fce3`
   (16:51:43+05:30), W2 `d88420a191e8c6475edefb026a056e78ec8dd0e5a09cb67b0a756dcb16ca209e`
   (16:52:05), W3 `8312d27c6162bbea02e4b586091bbb9e59b4a7ff6edc3ee25521afa4f28ef885`
   (16:52:30), W4 `18a1537ed3ee59d97ca5c5ede9de284b2da1e6fe2ca50e2ad73fd207ea09f1fc`
   (16:53:02).
3. **Commit timestamp.** This block and the driver are committed together,
   immediately after this insertion; the commit time corroborates the
   ordering story in the agent transcript.

Disclosed gap: the original in-doc insertion (first written ~16:47+05:30)
was deleted at ~16:48+05:30 by the second of two destructive external tree
resets (~16:41, ~16:48), and it left **no** surviving filesystem or patch
artifact — the exported `stash@{0}` patch contains zero hunks of this file
(verified by the Lead). Only the agent transcript retains the original prose.
The SUBSTANCE (mechanisms, parameterizations, windows, decision rules) is
independently anchored by the driver in (1), which predates all outcomes.
Disclosed disagreement: the self-recollection for the driver's write time was
~16:49+05:30; the measured mtime is 16:50:24+05:30. The measurement is
authoritative; nothing is smoothed. The mechanism/parameterization/window/
decision-rule text below is verbatim unchanged from the original pre-run
insertion. This block is never edited after the fact; any deviation from it
is itself a finding and must be recorded as such. Everything in this sweep is
**retrospective analytics** — simulated books through `run_backtest` on
collected Binance USDT spot bars with the lane's modeled costs, risk gate and
kill switch — carried as `live_pnl_claim=false`. **Binance USDT spot majors
only; not market evidence; not a live-trading claim.** The Sharpe/return/
drawdown figures here are book-reporting analytics under this label; they are
not research scorecards (which remain proper-scores-only).

**Base book (fixed, not re-tuned).** The flagship full stack above: spec
`vincent(ewma_emp+ewma_emp94+ewma_emp99)` (Vincentized mean of three fixed
EWMA memories 0.94/0.97/0.99) with policy
`mode=long_flat, kappa=0.15, gross_target=2.0, name_cap=0.5, cost_gate=0.15,
gate_on=edge, sizing=risk, deadband=0.01, band_lo=0.05, band_hi=0.95,
book_vol_target=0.03, exit_persist=3, leader_sid=BTCUSDT,
leader_edge_min=0.0, w_alpha=1.0, persist_bars=1, meta_lookback=60,
rebal_every=1` and all remaining gates off (`tail_gate, mkt_disp_cut, top_k,
mkt_edge_min, gate_out, meta_min, accel_min, fund_cut = null`,
`breadth_gross=false`). This is byte-for-byte the `vin3stack` policy recorded
in `.dsh-24x7/lane-simlive/sim_live_bench-1d-deep3-stack.json`. Each
mechanism below is exactly ONE change from this base; the base is evaluated
on every window as the control.

**Mechanisms (at most three, exactly ONE fixed parameterization each, chosen
a priori — no grid, no second value will be run):**

1. **`vol_scale_002` — one fixed-target vol scale.** `book_vol_target=0.02`
   (replacing 0.03). Chosen a priori as the round per-bar book-dispersion
   target one third below the base's own 0.03 — the hypothesis being tested is
   that a fixed, slightly conservative dispersion target mechanically
   de-risks exactly in chop regimes, where realized dispersion outruns trend.
   Not fitted to any window in the cross-window table below.
2. **`breadth_gross` — one fixed breadth-proportional exposure rule.**
   `breadth_gross=true` (gross cap scaled by the fraction of names with
   positive edge). A binary rule with NO free parameter to tune; the
   hypothesis is that in chop, fewer names carry edge, so exposure shrinks
   mechanically instead of being cut by a fitted threshold.
3. **`trend_gate_010` — one fixed trend-strength threshold.**
   `mkt_edge_min=0.10` (flat the book when mean cross-asset edge z ≤ 0.10).
   Chosen a priori as two-thirds of the base book's own fixed entry gate
   (`edge z ≥ 0.15`) — the only anchor used is the base's existing scale.
   Declared influence: `mkt_edge_min` was previously judged ~Sharpe-neutral on
   the full window in this doc; that prior judgment is disclosed here and no
   other level will be tried.

**Evaluation windows (declared up front; all four are locally available — the
8.3y deep bars from 2017-08 are present for BTC/ETH/XRP, so no window is
blocked):**

| id | window | universe | implementation |
|---|---|---|---|
| W1 | 2020-08→2026-09 (full) | 5 majors (BNB, BTC, ETH, SOL, XRP) | no tail cut |
| W2 | last 800 days | 5 majors | `eval_tail_bars=800` (panels see full history) |
| W3 | last 500 days | 5 majors | `eval_tail_bars=500` (panels see full history) |
| W4 | 2018-05→2026-09 (8.3y, bear-inclusive) | BTC/ETH/XRP | no tail cut on deep bars (3 majors) |

**Metrics per cell (analytics, `live_pnl_claim=false`):** `sharpe_simulated`
(primary), `total_return`, `max_drawdown`, `n_fills`, all from the same
`run_backtest` fill semantics on identical bars. Every mechanism × every
window cell is published below, losers included; nothing is dropped.

**Pre-declared decision rules (fixed before running):**

- **Chop-lift success** (the goal) requires ALL of:
  (a) `Δsharpe_simulated(W3) > 0` versus base — W3 (last 500d) is the
  pre-declared worst window (the chop regime);
  (b) `Δsharpe_simulated(W1) ≥ 0` versus base — the full window is not hurt;
  (c) no REGIME-FITTING trigger below fires.
- **REGIME-FITTING (mandatory rejection):** if the mechanism helps in any one
  window and hurts in any other (`Δsharpe_simulated > 0` in one and `< 0` in
  another), it is recorded as **REGIME-FITTING and rejected**, in the same
  verdict shape as `leader_edge_min` / `fund_cut` above — even if it satisfies
  (a) and (b). No materiality band: any sign reversal counts.
- If no mechanism satisfies all rules, the recorded verdict is **none of the
  pre-registered mechanisms lifts the worst window without hurting the full
  window** — a first-class, expected-possible outcome.
- Any attractive parameter value discovered by this sweep is NOT run. It
  would require a new dated pre-registration and is out of scope here.

## Chop-regime robustness sweep — RESULTS (run 2026-10-07T16:51–16:53+05:30)

**Retrospective analytics, `live_pnl_claim=false`. Binance USDT spot majors
only; not market evidence; not a live-trading claim.** Every pre-registered
mechanism × every pre-registered window is published; losers included,
nothing dropped. `sharpe_simulated` / total return / max drawdown / fills via
`run_backtest`, identical bars per window.

| window | base | `vol_scale_002` | `breadth_gross` | `trend_gate_010` |
|---|---|---|---|---|
| W1 full (5 majors) | 0.39 / +3.6% / −3.4% / 151 fills | **1.22 / +15.7% / −3.0% / 931** | = base (no effect) | 0.36 / +3.3% / −3.5% / 216 |
| W2 last 800d | no trades (0 fills) | 0.13 / +0.10% / −0.39% / 25 | no trades | no trades |
| W3 last 500d | no trades (0 fills) | 0.83 / +0.05% / −0.02% / 6 | no trades | no trades |
| W4 8.3y (3 majors) | no trades (0 fills) | −0.03 / −0.05% / −0.40% / 25 | no trades | no trades |

**Run-environment anomaly (disclosed, unresolved).** The base book here does
NOT reproduce the lane's historical bench receipts: on W4's exact bars
(sha256-identical) and byte-identical `vin3stack` policy, the historical
receipt reports SR 1.127 / +652% / 1,078 fills; this run records 0 fills and
a flat book on 3 of 4 windows. The divergence is therefore in the run
configuration, not the declaration: this sweep used `AppConfig()` defaults
where the lane's bench runs used a different config object. A corrective
re-run pinning `data.source=binance_public_data` produced byte-identical
outcomes, so the data label is not the cause. Within this round the
environment could not be reconciled to the historical bench config.

**Pre-declared verdicts (applied mechanically):** `vol_scale_002` — NOT
chop-lift success (criterion (a) not evaluable: base degenerate on W2/W3/W4;
known cells help W1 strongly); `breadth_gross` — NOT success (zero effect in
every evaluable cell); `trend_gate_010` — NOT success (criterion (b) fails:
ΔW1 = −0.026). REGIME-FITTING is **not evaluable** — the sign-reversal test
needs a live base in every window and the control book is degenerate in three
of four. Recorded verdict, in the `leader_edge_min`/`fund_cut` verdict shape:
**none of the pre-registered mechanisms lifts the worst window without
hurting the full window**, and the sweep is additionally recorded as a
**NON-RESULT pending run-environment reconciliation** — it must not be read
as out-sample or forward evidence (the 2025 holdout is spent;
`forward_2026H2` is not yet collected). Per the pre-declaration, no
parameter value discovered here will be tried; any future sweep needs a new
dated pre-registration.

## Per-asset attribution (same policy, single-name books)

BNB +195% (SR 0.95) · BTC +112% (0.99) · ETH +100% (0.75) ·
SOL +274% (1.48) · XRP +69% (0.61). Broad-based — no single-name luck.
The multi-asset book exceeds the naive sum because gross normalization
reallocates capital to whichever names currently pass the gate.

## Robustness checks

- **EWMA memory (the real axis — `window` saturates for lam=0.97)**:
  lam 0.94 → SR 1.59–1.74; 0.97 → 1.79; 0.99 → 0.87–1.17. Degrades
  gracefully, not knife-edge.
- **Deadband** 1%→3%: +2,488% / SR 1.775 (turnover drag only).
- **Name cap** 15%: +691% SR 1.68 · 40%: +567% SR 1.20 — 25% optimal.
- **4h interval** (harder: 2191 bars/yr × costs): best book
  `ewma_emp` z0.2/risk +10.8% SR 0.60 over ~2y; fhs 15bps conviction
  book +0.83% SR 0.97 on 154 fills. Daily dominates 4h.
- **Fail-closed verified**: stale-mark `StaleValuationError`,
  fingerprint-bound resume (80→160-step continuation, NAV seam parity,
  cumulative fills, no duplicate orders), monotone-quantile rejection.

## What this does NOT establish

- Live profitability. Slippage, venue latency, borrow/funding, and
  market-impact regimes past the modeled caps are unmeasured.
- Cross-venue or cross-class generality — Binance USDT spot majors only.
- The OOS window overlaps the strongest trend regimes; a flat market
  would compress these numbers further.
