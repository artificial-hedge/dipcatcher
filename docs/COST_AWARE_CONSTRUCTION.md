# Cost-aware portfolio construction (step 3)

The net-return tournament now supports an optional convex allocator. It uses
actual filled positions, forecast and uncertainty proxies, covariance and
modeled costs to choose target weights. It can keep cash or leave positions
unchanged when the expected benefit of trading is too small.

This is an engineering implementation, not measured market uplift. The tracked
real-price Parquet snapshot can be recovered from a normal Git checkout.
Generated-price tests verify mechanics; they do not establish profitability.
The existing vendor universe has survivorship bias and an already-inspected
holdout. A rejected CLARABEL solve records its status, iteration count and
solve time when available, with `weights_accepted=false`. The allocator tries
only mathematically equivalent formulations and never substitutes rejected or
merely feasible weights for an optimal solve.

The 2026-09-25 frozen snapshot run is a blocked validation: both cost-aware
variants returned `optimal_inaccurate`. Its sealed failure and control ledgers
are in `data/metadata/research/phase1_20260925.md`, linked by
`data/metadata/research/phase1_evidence_index.json`. There is no selected
candidate and no test receipt. The test command below applies only after a
new frozen run has selected a validation candidate.

**2026-10-07 re-run verdict (receipt `receipts/cost_aware_rerun_20261007.json`,
seal `618fadcd…`).** The conditioning repair and the fail-closed solver chain
are mechanically proven (see *Numerical solve protocol*), but the matched
validation re-run on the tracked 424-name snapshot under the committed chain
**selects nothing again** — `selected: null`, `complete: false`. Some
cost-aware decisions still end in `optimal_inaccurate`, which the chain
records as a diagnostic and refuses, exactly as designed. Said loudly and
clearly: **no candidate is selected; no test receipt exists;
`economic_evidence_gate` is `false`; `selected_holdout_adjusted_rejection`
never turned true because no test phase ever ran.** A superseded run under
the pre-rewrite 314-line repair did select `momentum_20_cost_aware`, but
forensic byte-hash evidence shows that run used the older, looser-acceptance
code (`058afffc…`), not this chain; its own test phase was refused by the
run's code lock after external tree wipes. The disagreement between the two
runs is disclosed in the receipt and is not reconciled: the fail-closed
refusal is binding, and the earlier selection is not claimed as evidence of
a conditioning-fix success.

## Run a matched comparison

After preparing the benchmark manifest as described in [REAL_DATA_BENCHMARK.md](REAL_DATA_BENCHMARK.md):

```bash
PYTHONPATH=src python -m quant_fund.research.net_tournament prepare \
  --benchmark-run data/metadata/real_benchmark \
  --spec configs/cost_aware_tournament.json \
  --output data/metadata/cost_aware_tournament
PYTHONPATH=src python -m quant_fund.research.net_tournament run \
  --run data/metadata/cost_aware_tournament --phase validation
PYTHONPATH=src python -m quant_fund.research.net_tournament run \
  --run data/metadata/cost_aware_tournament --phase test
```

The sample slate freezes momentum-20 and reversal-1, each with a rank control
and cost-aware variant, plus the equal-weight benchmark. A cost-aware candidate
must have an otherwise identical rank control. The default configuration is an
illustrative protocol, not a fitted optimum. Every parameter change is a new
research trial and requires a new frozen run; record external experiments too.

All strategies use the same universe and history requirement (the largest
lookback/risk window across the slate), sessions, next-open execution, costs,
initial capital and exposure limits. Adding a longer risk window can change the
common universe, so compare within the same frozen run. The optimized allocator
can change exposure, selection persistence and turnover; this is a construction
ablation, not a risk-matched or factor-neutral alpha estimate.

Each scenario includes `allocation_ablations`: descriptive differences in mean
daily net return, total return, maximum drawdown (negative convention), aggregate
costs and traded notional. Differences are candidate minus control. Pairwise
comparisons do not claim statistical significance. The existing full-slate
RC/SPA/StepM procedure still compares every candidate with equal weight;
validation-only selection and failure/terminal-liquidation gates still apply.
A failed allocator remains a failed candidate and blocks selection. Nothing in
this runner promotes a strategy or enables live trading.

## Model and units

Weights are fractions of signal-close NAV. Previous weights come from filled
shares marked at that close, including drift and prior fees, not yesterday's
target. With `d = |w - previous|`, the allocator maximizes:

```
alpha @ w - risk_aversion * w.T @ covariance @ w
          - uncertainty_aversion * uncertainty @ abs(w)
          - linear_cost * sum(d) - impact @ d**1.5 - holding_cost(w)
```

| Input | Construction / units |
|---|---|
| Return proxy | Daily geometric rate of the same trailing price ratio used to rank names, multiplied by frozen `alpha_scale`; negated for reversal. Unselected existing positions get zero alpha. This is not a trained return forecast. |
| Covariance | Sample covariance of simple close returns over `risk_window`, shrunk toward its diagonal with a frozen coefficient. No future bars or labels. |
| Uncertainty | `sqrt(diag(covariance) / lookback)`. A dispersion proxy with an independence approximation, not a calibrated confidence interval. |
| Linear costs | `(commission_bps + half_spread_bps) / 10000` per traded NAV fraction. |
| Impact coefficient | `impact_y * known_20_session_volatility * sqrt(NAV / known_ADV_dollars)`. Multiplied by `d**1.5`, this matches the replay's square-root participation impact at planning prices. |
| Holding costs | Short borrow plus financing, with cash return expressed as opportunity cost. APR / 252 is the fixed one-session planning approximation; replay charges actual calendar time on its cash/positions ledger. Funding APR must be at least cash APR. |

Forecast magnitudes and uncertainty are explicit proxies. Before making an
investment claim they need train-only calibration, separate validation and
independent forward evidence. The public `allocate` function also accepts
externally generated as-of forecasts and uncertainty arrays, but timestamp and
training provenance for those inputs remain the caller's responsibility.

## Constraints and execution

- Total absolute change is capped by `turnover_limit`; each change is capped by
  `participation_limit * known_ADV / signal_NAV`.
- Buffered name/gross limits include estimated transaction fees in their NAV
  denominator. Unlevered portfolios retain the declared cash buffer after
  estimated trading fees. Holding fees and future price gaps are approximated,
  not guaranteed by the planning constraints.
- New exposure is allowed only in the names and directions selected by the
  control's rank signal. Existing unselected positions may stay or shrink,
  allowing gradual exits when capacity or costs bind. There is no mandatory
  full investment or net-neutrality constraint.
- Existing holdings need available marks, risk history and liquidity. Missing
  inputs fail the candidate. If drift, liquidity or turnover constraints make
  the problem infeasible, it fails without relaxing limits or substituting
  target/previous weights.
- CLARABEL must return `optimal`. Returned weights are checked against every
  original constraint with a maximum absolute residual of `1e-7` NAV fraction.
  The solver's own constraint expressions and the unscaled objective are checked
  again after accepting a candidate. Tiny
  changes below `1e-8` are snapped to previous weights and rechecked. These are
  numerical tolerances, not an economic minimum-order policy.
- Next-open prices can differ from planning prices. Replay still caps fills and
  checks realized cash/exposures, recording rejects and unfilled amounts. The
  next solve uses actual filled holdings. Terminal exits bypass the optimizer's
  turnover budget but still obey execution participation and risk/cash checks;
  residual positions are marked and prevent the evidence gate from passing.
- The double-impact scenario increases realized execution costs; planning
  coefficients stay frozen. Later decisions can differ because costs changed
  cash/holdings. This stresses the same policy under worse implementation costs.

The `allocations` ledger records security mapping, planning NAV, prior/target
weights, alpha/uncertainty/capacity, predicted risk and costs, turnover, solver
status and constraint residual. Tournament receipts hash allocator code and
record CVXPY, CLARABEL and SciPy versions in addition to the prior runtime stamp.
Old receipts require their original code/runtime; create a new run for this
implementation.

## Numerical solve protocol

The frozen September 25 receipt remains a failed run. The 2026-10-07 repair
fixes problem conditioning and adds a fail-closed multi-solver chain; the
acceptance bar is not lowered anywhere. Solvers are the installed set only
(CLARABEL, OSQP, SCS, HiGHS/SCIPY; ECOS is absent and no dependency is
added).

**Solver chain and capability gating.** Each solve attempt runs
CLARABEL → OSQP → SCS → HIGHS in order, skipping solvers that cannot
represent the problem: the 3/2-power impact cone (`imp·δ^1.5`) is supported
by CLARABEL and SCS only; OSQP is QP-only (eligible when `impact=0`); HIGHS
is LP-only and is never eligible because the quadratic risk term is always
present — recorded honestly as `not_attempted_unsupported_cone`. Every
attempt records solver name, normalized status, iteration count and
wall-clock solve time; wall-clock is stripped from the hashed ledger so
allocations stay deterministic across runs.

**Status normalization (fail-closed dispatch table).** Only a genuinely
`optimal`, constraint-satisfying solution is selectable.
`optimal_inaccurate`, `suboptimal`, `feasible`, `user_limit`, `infeasible`,
`unbounded`, `numerical_error`, `solver_error` and `indeterminate` are
recorded as diagnostics with `selection_verdict: not_selectable` and can
never supply weights; an unrecognized status string maps to `("unknown",
not_selectable)`. The independent recomputation gate is unchanged: capacity,
turnover, buffered exposure and cash limits must hold and the recomputed
objective must agree within `1e-7` before any weight is accepted.

**Formulation ladder (mathematically equivalent only).** Fixed order, fresh
CVXPY variables per attempt:

| Formulation | Exact change from the original model |
|---|---|
| `original` | Original problem with the conditioning fixes below. |
| `bounded_capacity` | Limit each nonnegative trade variable by `min(capacity, turnover_limit)`. `sum(delta) <= turnover_limit` already implies each bound. |
| `factored_risk` | Write the same PSD quadratic risk as a squared eigenfactor norm. |
| `scaled_cost` | Express the same transaction-cost epigraph in nondimensional units (`cost_unit`), divided out in objective and constraints. |

**Conditioning diagnosis (why CLARABEL returned `optimal_inaccurate`).** The
September formulation amplified the objective by 100 while keeping fixed
absolute solver tolerances — effectively demanding a `1e-10` gap in original
units on a badly scaled cost-epigraph variable (~`1e-4`), next to redundant
capacity bounds of order `1e3`. On the market-derived `momentum_factor359`
fixture a one-change-at-a-time flip experiment isolates the cause: objective
scale 100→1 alone → `optimal` (41 iterations) instead of
`optimal_inaccurate` (58); bound presolve alone → still `optimal_inaccurate`
(90); cost-epigraph nondimensionalization alone → `optimal` (118); all three
→ `optimal` (18). The repair therefore rescales and reformulates the problem
— deterministic uniform objective scaling `1/s_obj` (the objective is
degree-1 homogeneous, so this is exact), cost-epigraph nondimensionalization,
bound presolve `min(capacity, turnover_limit)`, and the eigenfactor risk form
— and applies uniform original-unit tolerances: `GAP_TOL_ORIGINAL = 1e-10`
(the strictest gap any historical formulation effectively demanded —
tightened, never loosened) and `FEAS_TOL_ORIGINAL = 1e-8`.

**Equivalence proof.** `tests/unit/backtest/test_cost_solver_chain.py`
proves scaled and unscaled problems return the same solution (uniform
scaling `k ∈ {1e-2, 1e-1, 1e1, 1e2}`, `atol=1e-6`) and that every
exactly-equivalent formulation agrees (`atol=1e-5`, ~0.001% NAV; measured
agreement ~3e-6). A status-matrix test pins that only `optimal` is
selectable; forced non-selectable statuses and injected constraint
violations never provide weights. Four market-derived numerical fixtures in
`tests/fixtures/cost_allocation/` exercise the previously failing decisions
as a numerical regression only: their success measures solver reliability,
not a market edge.

## Validation and research basis

Focused tests cover analytical risk optima, cost-induced no-trade regions,
impact/uncertainty response, borrow/cash costs, liquidity/turnover/post-fee
exposures, invalid inputs, solver/infeasibility failures, causal decisions,
actual-fill feedback, matched ablations and frozen validation/test execution.
A flat generated market demonstrates avoided round-trip costs; it is a
controlled accounting example rather than evidence of a market edge.

The objective follows the single-period convex construction framework in
[Boyd et al., Multi-Period Trading via Convex Optimization (2017)](https://stanford.edu/~boyd/papers/cvx_portfolio.html),
including linear and 3/2-power transaction costs and holding costs. See the
[authors' slides](https://stanford.edu/~boyd/papers/pdf/cvx_portfolio_talk.pdf)
for the normalized impact model. This implementation is single-period and
does not implement the paper's multi-period forecasting/control system.
The legacy `portfolio.optimizer` API is unchanged; this bounded research
allocator is integrated specifically with step 2's independently audited replay.
