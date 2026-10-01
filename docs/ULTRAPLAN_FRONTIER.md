# ULTRAPLAN — Frontier upgrades for dipcatcher

**Status:** active. **Created:** 2026-09-22. **Owner lane:** frontier expansion —
supersedes nothing; absorbs the open tails of [MEGAPLAN_SOTA.md](MEGAPLAN_SOTA.md) (phases A–G),
[MEGAPLAN_SHARPE5.md](MEGAPLAN_SHARPE5.md) (performance lane), and the `.dsh-24x7` industry-grade bar
into one ordered program.

## Mission

Make dipcatcher the most *rigorously evidenced* open quant research lab: the
largest set of independently-verifiable PROVEN claims across four bars.

| bar | current status | ultraplan target |
|---|---|---|
| Forecasting SOTA | PROVEN (scoped: Binance 1d+4h next-bar CRPS vs 4 TSFMs) | Wider & deeper: more cells, horizons, targets, published protocols, second domain |
| Industry-grade | NOT PROVEN (vbt parity+reliability win, ~1.8–5.4× latency loss; qlib parity+104× win; zipline not-fair) | PROVEN: close/argue latency, add NautilusTrader lane, UX+security evidence |
| Strategy performance | NOT PROVEN (carry dev Sharpe 1.7–2.1, holdout ~0; Sharpe-5 honestly failed) | Highest *honest* result achievable; multi-sleeve + vol-target + basis-carry lanes, locked holdout |
| Scientific infra | strong (receipts, hashing, MCS/SPA/DM, fail-closed) | frontier-grade: CI repro, mutation testing of money paths, adversarial self-audits as routine |

**Honesty contract (unchanged, absolute):** every claim traces to a receipt with
embedded hashes; negative results are recorded, not hidden; no live-PnL claim;
locked holdouts are never retuned. "Beating every hedge fund" is not a
verifiable claim — the plan instead maximizes *scoped, auditable* proofs a
skeptic cannot dismiss.

## Research basis (2026-09-22 scan)

Forecasting targets — the field moved since the current slate was frozen:
- **Moirai-2.0** (Salesforce, arXiv:2511.11698): decoder-only, quantile output,
  #1 non-leaking on GIFT-Eval MASE; `Salesforce/moirai-2.0-R-small` on HF.
- **TiRex-2** (NX-AI): xLSTM, zero-shot SOTA claims, ships **decontaminated
  checkpoints** (TiRex-2-g/-f) — lets us answer the contamination objection.
- **Sundial** (THU-MT, flow-matching generative), **Toto** (Datadog),
  **YingLong** (Alibaba), **TabPFN-TS** (PriorLabs, 11M, tabular-PFN
  formulation, strong covariate-informed results).
- Community benchmark framing: GIFT-Eval + fev-bench; TIME benchmark
  (arXiv:2602.12147) argues for task-centric strict zero-shot cells — aligns
  with our protocol.
- **"Heads, not backbones"** (arXiv:2606.30037): on fat-tailed returns at
  short horizons, output *distribution head* dominates architecture —
  Gaussian→mixture adds ~2.4% CRPS, largest in high-vol regimes. Directly
  motivates a mixture challenger: cheap, causal, exactly our lane.
- **Two-stage quantile-NN + spline CDF** (arXiv:2408.07497): SOTA-claiming
  distributional forecaster for stock returns; adaptable as a challenger.
- Conformal frontier: online-CP-via-online-optimization (Areces et al. ICML
  2025), KOWCPI kernel-weighted CP (ICLR 2025), O2CP cross-horizon CP,
  ResCP reservoir-reweighted CP — all extend our existing conformal stack.

Strategy lane — external evidence *confirms* our honest negative:
- BitMEX (Jan 2026): funding carry "post-yield" — compressed below ~4% on
  majors; edge migrated to TradFi-perp and CEX–DEX spreads.
- Funding-carry falsification studies (public, pre-registered): base trade
  net-negative OOS on BTC/ETH/SOL after costs; "the rule that barely trades
  is the one that works" — validates our hysteresis design choice.
- Survivors worth trying: **quarterly-futures cash-and-carry basis**
  (settlement-anchored, ~3%/yr unlevered, every settled contract positive in
  published in-sample), cross-venue funding (needs second venue's data —
  scope-dependent), cross-sectional **reversal** (crypto four-factor models
  find reversal > momentum), **time-series momentum** (stronger evidence than
  cross-sectional per SSRN 4675565).

Incumbent lane:
- **NautilusTrader** is the serious execution-realism incumbent (Rust core,
  backtest/live parity, order-book first-class) — the right third matched
  workload after vectorbt/Qlib. Conformance-replay framing (decision-matrix
  replay) is exactly our existing methodology.

## Phase structure & item ledger

Ordered roughly by unblocking value. Item IDs are stable — cite them in
receipts/progress. ✅/🔄/⬜ track state; ❌ = attempted, honestly failed.

### P0 — Close open evidence debt (integrity-critical; nothing else claims finality until this lands)

- [ ] P0.1 Poll remote fleets to completion: `h4f_*` (bnb/btc/xrp partial),
      `tfmfix_*` (d1 landed; deep cells in flight), `native nd_*/nh_*`
      (eth/xrp 4h pending), `s11_*` (not started). Respawn dead shards via
      `scripts/spawn_*.ps1`; capture stderr for any that die twice.
- [ ] P0.2 TimesFM contract-v2 splice: `scripts/splice_timesfm_fix.py` on all
      v4-daily + h4f cells; assert challenger/target non-TimesFM columns
      bit-identical; archive pre-splice matrices.
- [ ] P0.3 Contract-v2 re-merge all cells (`--bars-root data/raw/sources`),
      n_boot 2000, fresh receipts; update [EVAL_REPORT_SOTA.md](EVAL_REPORT_SOTA.md) §4.1 table.
- [ ] P0.4 Native-protocol crossover merge (`sota_eval_native.py --merge-parts
      .dsh-24x7/native/nd_*`) → §4.3 ordering table (path RankIC, H-step
      RankIC, vol MAE/R²) with disclosed deviations.
- [ ] P0.5 `s11_*` + `s23_*` corrected-pairing seed replication fleet →
      §4.4/B1 closure.
- [ ] P0.6 Full-coverage 4h merge replaces complete-case receipt; student-t
      coverage row updated (was 63% NaN → hardened fitter 0 NaN claim must be
      *receipted*, not just probed).
- [ ] P0.7 Update PROOF.md SOTA section, HANDOFF.md, PROGRESS.md; flip
      EVAL_REPORT status from DRAFT when verdict checklist fills.

### P1 — Challenger frontier (beat targets by more, in more cells)

Each new challenger: causal-only, deterministic seed, honest-NaN on failure,
added to `BASELINES`/`dip_challengers`, unit-tested against known cases, then
a `v5_*` fleet column on existing shards' protocol.

Swarm harnesses landed: `research/fleet_eval.py` + `dipcatcher fleet` (nine
seeded SYNTHETIC shards — iid_gaussian/bimodal_mixture/heavy_tail/left_skew/
regime_switch/garch_cluster + vol_break/gjr_leverage/ar1_lagged_x — scoring
every registered head with proper scores only into sealed receipts), and
`research/identity_sweep.py` + `dipcatcher verify-identities` (38 curated
estimator identities proven over seeded SyntheticBundle draws with
hash-sealed receipts, exits non-zero on any violation).

Fleet coverage is now the full landed head slate — all 13 registry entries
(unconditional + conditional/series via fleet adapters) scored on all 9
shards, sealed in `receipts/fleet_eval_5ddf15b0dc7d3ca1.json` (117 rows,
0 errors; proper scores only — pinball/CRPS/PIT-KS/coverage; head versions +
per-shard seeds embedded). This closes every "Fleet cell still open" note
below. Adapter details: `qar` is scored on its native one-step grid (eval row
`i` forecasts from the observed lag `y[n_train+i-1]` — frozen coefficients,
no refit, no lookahead); `hstep` enters as `hstep_t`/`hstep_emp`, the two
`h=1` construction blocks of `HStepScaledDistribution` — honestly scorable
on the 1-step trailing slice; the longer-horizon blocks stay outside the
fleet contract.

- [x] P1.1 `dip_gmm_k` — Gaussian-mixture density head (K∈{2,3,4}, EM on
      trailing returns, BIC or fixed-K; the "heads not backbones" result
      predicts +2–4% CRPS in high-vol regimes). Closed-form mixture CRPS
      (Grimit et al. 2006 identity) — no sampling noise.
      Head wired: `train distribution --model gmm` (`GMMDistribution`,
      BIC over K∈{2,3,4} or fixed) + `gaussian_mixture_crps_1d` in
      `models/mixture.py`. Fleet cell closed: `gmm` scored on all 9 shards
      (receipt `fleet_eval_5ddf15b0dc7d3ca1`).
- [x] P1.2 `dip_skt` — Hansen/Fernández–Steel skew-t MLE (captures asymmetry
      that symmetric-t misses on crypto).
      Head wired: `train distribution --model skew_t` (`SkewTDistribution`
      wraps `models/skew_t.py` MLE + ppf). Fleet cell closed: `skew_t` scored
      on all 9 shards (receipt `fleet_eval_5ddf15b0dc7d3ca1`).
- [x] P1.3 `dip_qar` — quantile autoregression (Koenker–Xiao) direct per-τ
      fit; monotone-quantile enforced.
      Head landed: `QARDistribution` in `models/qar.py` — single-series,
      one-step-ahead head (`predict` accepts exactly 1 row; `fit` requires
      an all-finite series). Deliberately NOT in the `train_distribution`
      panel catalog — the panel interface has no honest row-order contract;
      usable via direct construction. Fleet cell closed: `qar` scored on
      all 9 shards via the `_QarOneStepHead` adapter — per-row forecasts
      conditioned on the *observed* lag-1 return, frozen coefficients, no
      refit (receipt `fleet_eval_5ddf15b0dc7d3ca1`).
- [x] P1.4 `dip_conf_t` — conformalized Student-t: parametric base +
      weighted-online-conformal recalibration of residuals →
      distribution-free coverage correction (bridges our conformal stack
      into the distributional lane).
      Head wired: `train distribution --model conf_t`
      (`ConformalTDistribution` — Hansen skew-t MLE on leading 2/3, CQR
      additive shift per τ on trailing slice, `ceil((n+1)·τ)` order
      statistic). Fleet cell closed: `conf_t` scored on all 9 shards
      (receipt `fleet_eval_5ddf15b0dc7d3ca1`).
- [x] P1.5 `dip_regime` — 2-state vol-regime mixture (existing `regime.py`
      HMM or Markov-switching): per-state empirical, state-prob mixed.
      Head wired: `train distribution --model regime`
      (`RegimeDistribution` — filtered `GaussianHMMRegime` posteriors on
      |y|, last-obs weights mixing per-state empirical CDFs via generalized
      inverse; VolThreshold + single-state fallbacks disclosed in
      metadata). Fleet cell closed: `regime` scored on all 9 shards incl.
      the planted-break `vol_break` shard (receipt
      `fleet_eval_5ddf15b0dc7d3ca1`).
- [x] P1.6 `dip_fhs_skew` — FHS variant on skew-filtered residuals + GJR
      asymmetry already present; quantile-level tail check.
      Head wired: `train distribution --model fhs_skew`
      (`FhsSkewDistribution` — fixed-coefficient GJR(1,1) σ-path with
      moment-matched ω, skew-t MLE on standardized residuals, one-step
      σ forecast; `train_distribution` enforces a single strictly-ordered
      series). Fleet cell closed: `fhs_skew` scored on all 9 shards incl.
      `gjr_leverage` (strong γ + skew-t innovations; receipt
      `fleet_eval_5ddf15b0dc7d3ca1`).
- [x] P1.7 `dip_isotonic` — isotonic-recalibrated empirical (PIT-based
      recalibration on trailing window; cheap calibration challenger).
      Head wired: `train distribution --model isotonic`
      (`IsotonicPitDistribution` — PIT-quantile map recalibrating a
      Gaussian base; empirical-base variant is vacuous, so the calibrated
      parametric base carries the challenger role). Fleet cell closed:
      `isotonic` scored on all 9 shards (receipt
      `fleet_eval_5ddf15b0dc7d3ca1`).
- [x] P1.8 `dip_lgbm_q2` — LightGBM quantiles v2: richer causal feature set
      (realized-vol term structure, OHLC range, amount), Dask-free, ≤30
      features; keep warmup disclosure.
      Head wired: `train distribution --model lgbm_q2`
      (`LGBMQ2Distribution` — per-τ `LGBMRegressor(objective="quantile")`
      on the full causal PIT design, rearranged monotone; warmup disclosed
      in metadata). Fleet cell closed: `lgbm_q2` scored on all 9 shards; the
      `ar1_lagged_x` shard supplies a real causal feature frame
      (`[y_{t-1}, |y_{t-1}|]`), the rest exercise the inert-x path
      (receipt `fleet_eval_5ddf15b0dc7d3ca1`).
- [x] P1.9 Blend-search policy: `dip_blend` is empirical+parametric concat;
      add `dip_stack` — weights fit by *trailing-window* CRPS minimization
      (causal stacking, no lookahead).
      Head wired: `train distribution --model stack` (`StackedDistribution` —
      per-τ convex weights via SLSQP pinball minimization on the trailing
      slice; empirical + Gaussian + skew-t bases; rearranged monotone).
      Fleet cell closed: `stack` scored on all 9 shards (receipt
      `fleet_eval_5ddf15b0dc7d3ca1`).
- [~] P1.10 h-step challengers for Phase D: vol-scaled h-bar distributions
      (σ√h + EWMA term-structure + Student-t tails; empirical h-day
      overlapping bootstrap) — honest constructions only.
      Head landed: `HStepScaledDistribution` (per-h unit-variance
      Student-t iid-sum construction plus empirical overlapping-bootstrap;
      `2·T·H` column layout disclosed in metadata). It requires one
      strictly chronological series and remains outside the generic
      `train distribution` catalog: that one-step evaluator expects
      `len(taus)` columns, while this head emits separate horizon
      blocks that need horizon-aligned targets. Fleet cell closed for the
      1-step slice: `hstep_t`/`hstep_emp` adapters slice the `h=1`
      `student_t`/`empirical` blocks — the only horizon honestly scorable
      on the fleet's 1-step trailing slice — on all 9 shards (receipt
      `fleet_eval_5ddf15b0dc7d3ca1`). Multi-horizon fleet scoring remains
      open pending horizon-aligned targets.

### P2 — New published targets (make the claim harder to dismiss)

Each: pinned artifact + sha256, zero-shot, native output honored
(quantile/sample/path), per-model coverage disclosed.

- [~] P2.1 `moirai2` — Salesforce/moirai-2.0-R-small via uni2ts; quantile
      head maps directly onto our CRPS/pinball path. Adapter landed:
      `Moirai2Distribution` (`models/moirai2.py`) — lazy fail-closed
      import, `availability()` gate, causal-window `predict_from_history`,
      registered in `FLEET_HEAD_REGISTRY`. Dep evidence: `uv add uni2ts`
      fails resolution — every published uni2ts (1.1.0–2.0.0) pins
      `scipy>=1.11.3,<1.12.dev0` and `numpy~=1.26.0` against pinned
      `scipy>=1.14` / `numpy>=2.0` (upstream main pins the same ranges, so
      git install does not help either), plus `gluonts~=0.14.3` → `toolz<1`
      vs `exchange-calendars==4.13.2` → `toolz>=1`. Lane stays fail-closed
      until upstream loosens. Fleet cell open pending a resolvable dep.
- [ ] P2.2 `tirex2` — NX-AI TiRex-2; prefer a decontaminated checkpoint for
      the fev-bench/GIFT overlap question; sample-path → distribution.
- [ ] P2.3 `sundial` — THU-MT flow-matching; sample paths → empirical dist.
- [ ] P2.4 `toto` — Datadog Toto if public weights resolve; else document
      unavailable.
- [~] P2.5 `tabpfn_ts` — PriorLabs tabpfn-time-series (CPU-feasible, 11M).
      Adapter landed: `TabpfnTsDistribution` (`models/tabpfn_ts.py`) — lazy
      fail-closed import, causal-window `predict_from_history`, registered in
      `FLEET_HEAD_REGISTRY`. Dep evidence: `tabpfn-time-series` transitively
      pins `toolz<1` (via gluonts) while `exchange-calendars==4.13.2` requires
      `toolz>=1` — unsatisfiable in uv.lock, so the lane stays fail-closed
      until upstream loosens. Fleet cell open pending a resolvable dep.
- [x] P2.6 `kronos_base` fleet head — wraps the real Kronos adapter,
      candle-envelope → 5/50/95 quantiles (#377). v5-fleet-scale cell
      remains remote-gated like the rest of the fleet.
- [~] P2.7 Classical neural baselines: N-BEATS / N-HiTS / DLinear via a small
      harness (Darts or direct) — closes the "only foundation models"
      objection.
      Heads landed: `NBeatsDistribution` / `NHiTsDistribution` in
      `models/nbeats.py` — direct deterministic CPU torch implementations
      (no Darts), doubly-residual N-BEATS blocks and multi-rate N-HiTS
      pooling, pinball loss on the scoring tau grid, seeded + single-thread.
      Wired into `FLEET_HEAD_REGISTRY` (`fleet --models nbeats,nhits`);
      predict emits the last-window one-step quantile vector tiled per row,
      warmup (`lookback`) disclosed in metadata. DLinear already exists as
      `models/dlinear.py` (sota_protocol path baseline). Fleet cell still
      open.
- [~] P2.8 Patch-Transformer reference line per arXiv:2602.06909 finding
      (generic transformer ~SOTA when pretrained at scale) — likely
      infeasible to pretrain; document as bounded.

### P3 — New cells & domains (widen the scope claim)

- [ ] P3.1 Hourly (1h) cell: deep bars exist remotely; same walk-forward
      protocol; watch microstructure-noise caveat (disclose).
- [~] P3.2 Multi-horizon: h∈{1,5,20} daily + {1,6} 4h on identical origins
      (megaplan Phase D); targets emit native paths, challengers use P1.10.
- [x] P3.2 Multi-horizon (daily half): `research/multih_fleet.py` +
      `quant multih-fleet` — identical origins, h∈{1,5,20}; 1-step heads
      extended via `iid_sqrt` (μ×h, σ×√h) and causal `empirical_ratio`
      (trailing h-sum/1-step dispersion), `hstep_*` scored on native blocks;
      proper scores on realized h-step sums; sealed `multih_fleet_eval`
      receipt. Result: empirical_ratio wins clustered/break shards, native
      hstep wins where horizon structure matters — the construction
      discriminates as designed. 4h {1,6} remains remote-gated (P3.1 bars).
- [x] P3.3 Second domain: US equity dailies — the committed 424-name
      `data/file_us_wide` yahoo corpus ran through `real_benchmark`'s
      preregistered two-phase harness (protocol hash-pins the parquet;
      prepare audits splits, then validation scores before test).
      Verdict: honest negative — on 495 validation dates / 431 test
      dates, no baseline (ridge, rolling_mean_20, historical_mean) beat
      the zero-return forecast on equal-weight MSE; `promote: false`,
      `claim: fixed_split_forecast_diagnostic`. Sealed receipts:
      `receipts/real_benchmark_us_wide_{manifest,validation,test}.json`.
- [x] P3.4 Cross-sectional lane: rank-IC eval vs targets on the panel
      (existing ranking bench + northset) — a different claim axis.
      Harness landed: `research/cross_sectional.py` + `dipcatcher rankic` —
      5 seeded planted-signal panels (linear, rank-preserving cubic,
      pure-noise null, mid-sample regime flip, weak edge) x 5 challenger
      transforms (identity / noisy / lagged / shuffled / inverted), Spearman
      rank-IC per date + Newey-West mean-IC t-stat per horizon (1/5/20),
      sealed receipt `receipts/rankic_eval_*.json`. Property tests:
      asset-permutation invariance, inversion sign-flip, ~nominal null
      rejection on the shuffled challenger.
- [x] P3.5 Volatility-forecast cell: QLIKE on next-bar/h-step realized vol —
      `dip_garch_t` already near-top CRPS; formal vol bench vs published
      vol baselines (HAR, realized-GARCH).
      Harness landed: `research/vol_bench.py` + `dipcatcher vol-bench`
      (seeded SYNTHETIC shards — garch_vol / rough_vol / break_vol — scoring
      HAR-RV, realized-GARCH, dip_garch_t and RV baselines with QLIKE/MSE on
      cumulative h-step realized variance, NW loss diffs vs `har`, sealed
      receipts). Real-data vol cells still open.
- [x] P3.9 Selection-concordance lane: does "head X wins" survive the choice
      of multiple-comparison correction? `research/concordance.py`
      (`quant_fund.research.concordance.run_concordance_eval`) runs MCS /
      Romano–Wolf StepM / pairwise DM on
      the same pinball loss tensor per shard — eliminated-set Jaccard,
      Kendall-τ on elimination confidence, SPA/Reality-Check decisiveness on
      differentials-vs-best. A head MCS keeps but StepM rejects is flagged:
      dependence-fragile selection, not evidence.
- [x] P3.7 Distributional coherence bench
      (`quant_fund.research.coherence.run_coherence`) —
      `research/coherence.py` reconciles per-name marginal quantile grids
      to the aggregate distribution on SYNTHETIC correlated panels
      (gauss/independent/heavy-tail/regime-break copulas). Methods:
      direct aggregate fit, naive sum-of-quantiles (comonotone bound),
      independent MC convolution, and a Gaussian copula MC fit on
      in-sample PIT z-scores. Proper scores only; sealed receipt.v2.

### P3b — Sequential inference suite (new statistical layer)

- [x] Anytime-valid head promotion: `research/evalues.py` `LossEProcess`
      (betting e-process, Ville/Ramdas) wired into `vol_bench` — #380.
- [x] Sequential fleet elimination: `research/fleet_race.py` + `dipcatcher
      race` (two e-processes per head vs fixed incumbent) — #381.
- [x] Corpus-level inference: `research/corpus_inference.py` harvests all
      committed receipts → pooled BH-FDR + e-value product — #382.
- [x] Online FDR over the receipt stream: `research/online_fdr.py`
      Foster–Stine alpha-investing — #383.
- [x] Verifier contracts for the family: `research/evalue_contracts.py`
      deep-checks the kinds — #384, #398 (extended).
- [x] Winner's-curse correction: `research/winner_curse.py` bootstrap
      selection-bias + split-half honest control — #385.
- [x] Anytime-valid drift alarms: `research/drift_alarm.py` level-shift
      e-process + Page–Hinkley diagnostic — #386.
- [x] Composite verdict: `research/honest_verdict.py` — #387.
- [x] Registry completeness ratchet (no orphan heads) — #388.
      See `docs/SEQUENTIAL_INFERENCE.md` for the architecture.

### P4 — Industry-grade bar (the open one)

- [x] P4.1 Profile `run_backtest` on the 11-asset workload (cProfile +
      allocation trace); classify remaining 5.4× gap: interpreter loop vs
      per-order gate cost vs polars overhead.
      Landed: `docs/PERF_SWEEP.md` — top-20 tables, classification
      (~80% per-order gate/cost, ~15% loop body, ~5% marshalling), and
      bit-identical event-loop vectorizations (1.41× event-loop speedup).
- [x] P4.2 Implement `run_backtest_fast` vectorized replay path for the
      *matched-workload class* (fixed rules: target-percent orders, next-open,
      no limits/stops) behind an explicit flag; must produce bit-identical
      NAV/fees on the conformance suite before use in any receipt.
      Landed: `backtest/fast_replay.py` (numba kernel + interpreted
      fallback), explicit `run_backtest(..., fast=True|False|None)` flag with
      fail-closed refusal of unsupported workload classes, byte-identical
      property suite `tests/property/test_fast_replay_byte_identity.py`,
      scope/gap analysis `docs/FAST_REPLAY_P42.md`, receipt
      `receipts/fast_replay_p42_conformance_20260928.json`.
      `receipts/legacy-unsealed/fast_replay_p42_conformance_20260927.json`.
- [x] P4.3 If fast path can't reach ≤1× honestly, write the argument:
      per-order risk gates + fail-closed semantics are the product; vectorbt
      is a vectorized reducer without them; show latency decomposition
      table + the 3/3 fault-injection wins.
- [x] P4.4 NautilusTrader conformance replay attempt (third incumbent):
      same bars/panel/costs — attempted and sealed not-fair/attempted
      verdict: `receipts/nautilus_conformance_*.json` (#232).
- [~] P4.5 UX evidence: `dipcatcher doctor` self-check output, error-message
      quality suite, `--help` coverage vs incumbent CLIs/APIs.
- [x] P4.6 Security evidence: sealed `receipts/deps_security_hygiene_*.json`
      (pin audit + `uv audit`), gitleaks in CI with allowlist ratchet,
      no-`eval`/no-`pickle` sweep clean, SHA-pinned actions.
- [ ] P4.7 Write the industry-grade verdict in PROOF.md only after P4.1–P4.6.

### P5 — Strategy performance (highest honest result; locked holdout)

Dev-window tuning only; the holdout stays locked. Negative results recorded.

- [~] P5.1 Multi-sleeve dev study: carry + time-series momentum + x-sectional
      reversal (Kakushadze-style BTC-factor residual mean-reversion);
      sleeve-level risk-parity / vol-target overlay.
- [x] P5.2 Vol-targeting overlay (`research/capacity_overlay.py::
      vol_target_scales`): delay-1 trailing/EWMA σ estimate -> clip(
      target/σ, 0, max_leverage); warmup neutral, unmeasurable vol
      flattens. Dev-only evidence via `dipcatcher capacity --dev`.
- [ ] P5.3 Quarterly-futures cash-and-carry lane: collect Binance delivery
      futures (`collect_perp_universe.py` extension); settlement-anchored
      basis capture — the one structural edge with positive published OOS.
- [~] P5.4 Cost-side improvements: maker-fill assumption variant (limit-at-
      touch model already in SimulatedBroker — measure fee drag delta),
      hysteresis parameter robustness surface (not retuned on holdout).
- [x] P5.5 Cross-venue funding/basis: OKX resolved as second venue
      (Binance geo-blocked HTTP 451, Bybit 403); `research/crossvenue_basis.py`
      + `dipcatcher xvenue-basis` (kraken-okx preset) -> spot-vs-futures
      basis + daily funding differential across venues; sealed
      `crossvenue_basis.v1` receipt (descriptive stats only).
- [x] P5.6 Capacity analysis (`research/capacity_overlay.py::
      run_capacity_bench` + `dipcatcher capacity --dev`): 4 seeded
      SYNTHETIC books × AUM grid -> feasible-date share, days-to-trade,
      sqrt-impact bps under participation cap; sealed
      `capacity_overlay_eval` receipt (dev-only, SYNTHETIC).

### P6 — Full code audit ("every file can be made better" — verify or fix)

Audit order = blast radius. Each finding → fix + regression test, or written
waiver in the audit log. Output: [AUDIT_FRONTIER.md](AUDIT_FRONTIER.md) ledger.

- [x] P6.1 Money paths: `simulated_broker.py`, `carry_engine.py`,
      `perp_engine.py`, `engine.py`, `sleeves.py`, `risk_gate.py`,
      `costs.py`, `implementation_shortfall.py`, `pnl_attribution.py`.
- [x] P6.2 Statistical core: `scoring.py`, `inference.py`, `snooping.py`,
      `hac.py`, `evalues.py`, `conformal.py`, `multiple_testing.py`,
      `cpcv.py`, `purging.py`, `embargo.py`, `walk_forward.py`, `fdr.py`,
      `gates.py`.
      Partially done: metrics layer audited (`metrics/{inference,snooping,
      bootstrap,scoring,calibration_tests}.py`) — `docs/AUDIT_P62_STATS.md`
      + `tests/unit/metrics/test_stats_audit.py` (43 KATs); 3 proven bugs
      fixed (PW2004 block length, NaN coverage masking, degenerate-sd
      Sharpe CI). Validation/`purging`/`walk_forward` layers still open.
- [x] P6.3 Data integrity: `ingest.py`, `point_in_time.py`, `universe.py`,
      `corporate_actions.py`, `security_master.py`, `sources/`, `lake.py`,
      `calendars.py`.
- [x] P6.4 Model layer: every file in `models/` audited line-by-line
      vs cited behavior; `pipeline/` causal gates verified.
      Completion evidence: `quality/audit_coverage.json` marks
      Completion evidence: `quality/audit_coverage.json` +
      `quality/audit_coverage_fx1.json` mark
      `models/` and the named dirs `audited` under CI enforcement
      + `docs/AUDIT_LEDGER.md` per-directory findings (#334+).
- [x] P6.5 Exec/microstructure: `almgren_chriss.py`, `microstructure/*`,
      `northset/*` estimators (Kyle λ, Roll, VPIN, OFI).
- [x] P6.6 Infra: `paper/*` (ledger atomicity, resume), `registry/`,
      `monitoring/` (drift, kill_switch), `api/app.py`, `cli/main.py`,
      `reporting/tearsheet.py`, `utils/*` (hashing, seeds, reproducibility).
- [x] P6.7 Scale hygiene: `research/catalog.py` was 10.7k LOC — split into
      the `research/catalog/` package by theme (`_helpers`, `constants`,
      `predicates`, `session`/`candle`/`kyle`/`northset` honesty checkers,
      `consistency`, `families`); `__init__.py` re-exports all 444 public
      names so `from quant_fund.research.catalog import X` is unchanged.
- [x] P6.8 Perf sweep: cProfile top-20 hot paths across engine, features,
      scoring; fix only where semantics bit-identical.
- [x] P6.9 Test-quality audit: mutation spot-checks on money-path
      conditionals (#221, #372, #397 — survivors pinned), property tests
      for accounting identities, verifier mutation-fuzz (#368).
- [x] P6.10 Dependency hygiene: sealed `receipts/deps_security_hygiene_*.json`
      (pin audit + `uv audit` + license scan), verifier contract
      re-derives it (#349).

### P7 — Frontier infrastructure upgrades

- [x] P7.1 CI reproduction job: `.github/workflows/reproduce_sota.yml` —
      the *native* leg runs unconditionally (both committed parts reproduce
      their merged receipts bit-exact via `scripts/check_sota_reproduction.py`;
      volatile timestamp keys dropped, implementation-hash drift reported as
      warnings). The kronos leg stays gated on the repo-policy decision:
      `--bars-root data/raw/sources` is gitignored — commit the bar
      parquets or set `SOTA_ARTIFACT_URI`; the workflow notices-and-skips
      until then.
- [~] P7.2 Receipt v2 schema: unified envelope across eval/incumbent/
      carry/paper lanes (dataset hash, code hash, params, environment,
      `live_pnl_claim`, verdict).
      Partially landed: `research/receipt_v2.py` defines `receipt.v2`
      (pydantic + published `receipt_v2.schema.json`); `dipcatcher
      verify-receipt` validates v1/v2; `fleet_eval`, `vol_bench`,
      `capacity_overlay`, `cross_sectional` emit v2 behind
      `--receipt-version 2`. Remaining v1 writers migrating on an
      in-flight sweep.
- [~] P7.3 Experiment registry hardening: mlflow.db exists locally — wire
      fleet runs into it or document why not. (PR #218 open.)
- [~] P7.4 Determinism sweep: every `receipt.v2` envelope carries an
      `environment` block (python/numpy/polars/scipy versions, BLAS/LAPACK
      build from `np.__config__.CONFIG`, threadpools via threadpoolctl)
      with a `fingerprint_sha256` digest. Cross-process determinism proven:
      `sim_live` receipts are byte-identical under different PYTHONHASHSEED
      (`tests/unit/determinism/`). Still open: cross-machine fingerprint
      sweeps.
- [x] P7.5 Remote-fleet ops: consolidated into `scripts/fleet_spawn.ps1`
      (JSON-manifest WMI launcher, Win32_Process + `cmd /c` redirect,
      dry-run + spawn receipt) + `scripts/fleet_watchdog.ps1` (PID liveness,
      output staleness, bounded respawn, `.dsh-24x7/fleet_heartbeat.json`)
      + `scripts/fleet_manifest_sota.ps1`.
- [x] P7.6 `AGENTS.md` refresh: remote conventions landed — PowerShell-only,
      WMI spawn survives ssh teardown, parametrized launcher + watchdog,
      `.dsh-24x7` durable paths, thread-pinning env block, Defender
      exclusions.
- [x] P7.7 Evidence chain-of-custody: `research/evidence_audit.py` +
      `dipcatcher verify-all` — set-level receipt audit (filename↔digest
      binding, duplicate-seal detection, unsealed-legacy accounting,
      evidence-index freshness via byte-compared regen) emitting a sealed
      `evidence_audit` receipt. Also fixed `receipts-reverify` dispatch:
      v2 envelopes and sealed v1 receipts now route to `verify-receipt`
      instead of the notebook-schema `verify-research`, which had never
      verified a sealed receipt correctly. The 7 committed pre-envelope
      artifacts moved to `receipts/legacy-unsealed/` — retained for
      provenance, outside the seal-verified set.

## Execution rules

1. **Receipts or it didn't happen.** Every checkmark needs a receipt file +
   embedded hashes + rerun command, per `.dsh-24x7` convention.
2. **No silent regressions.** Optimization must be bit-identical or the
   difference is disclosed and justified.
3. **Locked holdouts.** Nothing in P5 retunes on holdout. Ever.
4. **Honest negatives are deliverables.** A falsified lane closes with a
   documented falsification, not silence.
5. **One claim per scope.** Pooled mixed-interval inference stays refused by
   design; per-interval claims only.
6. **No repo-policy decisions for the agent.** Committing data/matrices,
      publishing, or external posting needs the user.

## Ordering

```
P0 (evidence debt) ──▶ P4 (industry-grade) ──▶ P3/P2 (wider SOTA)
P1 (challengers) runs parallel — v5 fleet waits on P0.3
P5 (strategy) runs parallel — independent compute
P6 (audit) interleaved — findings feed all lanes
P7 throughout
```

First executable tranche (this session): P0.1 polling loop; P4.1 engine
profile; P1.1 GMM challenger locally; P6.1 money-path audit start.

## Status annotations (2026-09-28)

Checkboxes synced to main. `[~]` = shipped code on an open PR:

- P1.10: Multi-horizon fleet eval landing in #233 (identical origins, iid_sqrt/empirical_ratio/native constructions).
- P2.7: Fleet cell now wired: `nbeats`/`nhits` in FLEET_HEAD_REGISTRY; the committed fleet receipt predates them — refreshes on the next sealed fleet run.
- P2.8: In flight: #217 (PatchTST quantile head, bounded reference).
- P3.2: In flight: #233 (`quant multih-fleet` — h∈{1,5,20} identical origins, sealed multih_fleet_eval receipt).
- P3.6: In flight: #226 (fleet significance — DM matrix + Hansen MCS).
- P3.7: In flight: #227 (distributional coherence — copula-MC aggregate reconciliation).
- P3.8: In flight: #229 (mixture stability — bootstrap CI on expert weights).
- P3.9: In flight: #230 (selection concordance — MCS/StepM/DM agreement).
- P4.1: In flight: #210 (cProfile + allocation-trace sweep).
- P4.3: In flight: #216 (fast-path latency argument doc).
- P4.4: In flight: #232 (NautilusTrader conformance lane + sealed verdict).
- P4.5: In flight: #205 (clean config-error UX + doctor evidence).
- P4.6: In flight: #207 (dep-hygiene + security evidence).
- P5.1: In flight: #222 (multi-sleeve dev study) + #225 (residual_mr_weights sleeve).
- P5.4: In flight: #220 (cost surface — maker/taker delta + hysteresis robustness).
- P6.10: In flight: #207 (dep-hygiene + uv audit).
- P6.2: Validation layer in flight: #211 (22 KATs); metrics audit merged via #206.
- P6.4: Forecast/fusion layer in flight: #214; dist/vol families merged via #172 + model-layer leak fix #212.
- P6.8: In flight: #210 (perf sweep).
- P6.9: In flight: #221 (mutation spot-checks on money paths).
- P7.1: In flight: #219 (reproduce-sota workflow; still gated on artifact-store policy).
- P7.3: In flight: #218 (MLflow registry keyed to sealed receipts).
- P7.4: In flight: #203 (v2 adoption in capacity/rankic/vol-bench lanes).
- P7.5: In flight: #224 (parametrized fleet launcher + watchdog).
- P7.6: In flight: #224 (AGENTS.md remote conventions).
- P7.7: In flight: #231 (verify-all custody audit + reverify dispatch fix).

## Hardening wave (coverage + ratchets, 2026-09-28 later)

Second audit pass over the full-suite coverage map (85.49% on main).
Coverage-invisible lanes excluded (numba `@njit` bodies in
`backtest/fast_replay.py` + `backtest/_kernels.py`, Rust dispatch in
`native/__init__.py`, integration-gated orchestrators). Open PRs:

- #263 — narrow 3 `except Exception` lazy-import guards to `ImportError`;
  ratchet ceiling 75 → 72.
- #264 — robustness closed-form KATs (Clopper–Pearson, Gelbrich tangent,
  smoothing radius, distributional-bound status contract, threat operators)
  + simtest swarm invariant/failure-path tests.
- #265 — `# type: ignore` ratchet at 218.
- #266 — covariance catalog KATs (DCC-family recovery, PSD repairs,
  trailing-window fail-closed, spec-alias table).
- #267 — `fetch_yahoo_panel` fail-closed contracts + `prepare_bars`
  causality (adv/vol_20 strictly-past windows).
- #268 — `mc_engine/tails.py` KATs (Wilson, spectral/weighted ES,
  batch-means interval, POT/GPD exceedance fit).
- #260 — cross-process determinism (PYTHONHASHSEED, pipeline bitwise).
- #259 — TabPFN-TS zero-shot head shipped as fail-closed adapter (dep
  conflict `toolz<1` vs `toolz>=1` documented; P2.5 `[~]`).

## Type-discipline completion (2026-09-28 latest)

- #270 — promoted 248 strict-clean modules (allowlist 397 → 645).
- #271 — fixed the 23-file strict tail: `mypy --strict` clean on **all
  668** `src/quant_fund` modules; `STRICT_MODULE_FLOOR = 668` makes strict
  coverage total — any new non-strict module fails CI at birth.
- #274 — `src/fx1` strict-clean (65/65) + `mypy --strict` added to the
  fx1 Types step; both packages now strict-locked.
- #273 — coverage ratchets: global floor 80→81, per-package floors +1
  (pit/proof/reality/proofcore 91, leakage 86), fx1 lane 60→80.
- #272 — macOS shard fix (`mapfile` → while-read; runners ship bash 3.2).
- #269 — union-merge drivers for append-only ledgers (`.gitattributes`).
