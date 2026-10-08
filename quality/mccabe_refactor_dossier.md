# McCabe refactor dossier — generated 2026-10-08

Measured evidence, not estimates. Every complexity value here comes from
`uv run ruff check --select C901` (ruff 0.16.9) over `src/`, via
`scripts/check_mccabe_ratchet.py`. Regenerate with:

```bash
uv run python scripts/check_mccabe_ratchet.py --write-census quality/mccabe_violations_2026_10_08.json
```

Branch structure was extracted with `ast` from the live tree, not read by eye.

## Measured state

| Metric | Value |
|---|---|
| Functions measured (C901 reported) | 19,748 |
| Max complexity in tree | 182 (`src/fx1/cli_audit.py:cli_audit`) |
| Functions **> 74** (hard ceiling) | **3** |
| Functions **>= 10** (soft threshold, ratchet scope) | **1,089** |
| Functions **> 10** (strictly above soft) | 857 |
| Baseline rows before this task | 358 |
| Baseline rows after `--allow-with-census` | 1,094 (736 added, 358 carried) |
| Functions still failing `check` | 15 |

### The INFLIGHT `w1714` numbers are stale

`w1714` reports "ceiling violations 30 → 14" and names 14 offenders. As of
2026-10-08 the ceiling is clean: **0** functions exceed 74 in files that ruff
actually lints. The 3 that do are all in `src/fx1/` and are explicitly
`C901`-ignored in `pyproject.toml` (lines 177–179). The 14 named in `w1714`
have all been resolved.

The 736 figure is accurate and confirmed: those are functions at/above the
**soft** threshold of 10 that were never listed in the baseline.

## The actual reason `--write` refuses

Not ceiling violations. `--write` refuses because **10 existing pins would
have to be raised** — an upstream merge made those functions more complex than
the pin recorded. Verbatim:

```
refusing to raise baseline values:
  src/quant_fund/backtest/carry_engine.py:run_carry_backtest 52 -> 57
  src/quant_fund/backtest/engine.py:_run_backtest_event_loop 36 -> 39
  src/quant_fund/execution/simulated_broker.py:SimulatedBroker._attempt_fill 10 -> 13
  src/quant_fund/labels/engine.py:build_labels 12 -> 14
  src/quant_fund/leakage/ast_scan.py:_check_lh009 10 -> 11
  src/quant_fund/leakage/ast_scan.py:_check_lh011 15 -> 24
  src/quant_fund/leakage/ast_scan.py:_check_lh013 14 -> 23
  src/quant_fund/leakage/ast_scan.py:_receiver_price_like 11 -> 12
  src/quant_fund/paper/ledger.py:promotion_dry_run 20 -> 22
  src/quant_fund/portfolio/risk_gate.py:check_order 22 -> 24
```

**Provenance.** The baseline was written at `150f1b448` (2026-09-28). Verified
by checking out `ast_scan.py` at that commit and re-running ruff: `_check_lh011`
was genuinely 15 and `_check_lh013` genuinely 14 then. `ast_scan.py` was then
rewritten by merge `25279246a` "fix(leakage): fail-closed audit of the leakage
scanner" (2026-09-29), which is **not** a descendant of `150f1b448`
(`git merge-base --is-ancestor` → false). That merge added fail-closed branches
and pushed 15 → 24 and 14 → 23.

So the refusal is the interlock working exactly as designed. The remaining 9
raises are the same class of drift from other external lanes.

## Characterization tests required BEFORE any refactor

Every function below is on a money path or a fail-closed gate. Per `INFLIGHT`
`w1714`, `lh011`/`lh013` specifically need characterization first. Do not
refactor before these exist — a decomposition that changes branch *order* in a
fail-closed gate converts a silent pass into a silent fail or vice versa.

| Function | Test that must exist first | Why it is load-bearing |
|---|---|---|
| `SimulatedBroker._attempt_fill` | golden-order fill matrix: partial fills, queue position, maker/taker, zero-qty, self-trade | Fill price/qty feeds the ledger. A wrong branch order corrupts P&L silently. |
| `risk_gate.check_order` | table-driven: each rejection reason asserted individually, plus an allow-path control | Order rejection is a risk control. Every `if` is a distinct rejection reason that must stay independently reachable. |
| `promotion_dry_run` | dry-run promotion on synthetic tape: golden promotion receipt | Promotes the paper ledger. Fail-closed by design; a missed branch could promote on invalid evidence. |
| `ast_scan._check_lh011` / `_check_lh013` | **mandatory** — per-rule fixture corpus asserting the finding set exactly, on files that must-trip and must-not-trip | Leakage scanner. 18 `If` + 20 `BoolOp` each. A dropped conjunct silently permits a leak. |
| `build_labels` | label generation on a fixed synthetic tape, exact label array | Labels are training truth. Off-by-one from a reordering is invisible downstream. |
| `run_carry_backtest` | carry backtest golden summary on a fixed fixture, incl. roll dates | 499-line function. Only a numerical golden test catches a mis-split. |
| `_run_backtest_event_loop` | event-loop replay determinism + fill sequence golden | 10 `IfExp`; ternary reordering changes fill semantics. |

## Per-function decomposition proposals

### `src/quant_fund/backtest/carry_engine.py:126` — `run_carry_backtest`, cx 57, 499 lines

Branch structure: `If`=39, `For`=12, `BoolOp/Or`=9, `IfExp`=6, `And`=3, `Except`=2, `While`=1.

Three natural seams, in this order:

1. `_collect_carry_positions(bars, positions)` — hoist the roll-eligibility
   scan (the `For`×12 + most `If` in the bar loop). ~20 complexity out.
2. `_roll_position(position, roll_price, roll_date)` — pure per-position roll
   arithmetic; the `IfExp`×6 and `And`×3 live here and become unit-testable
   without a backtest harness.
3. `_carry_costs_and_pnl(events)` — the two `Except` handlers and net-P&L
   accumulation.

Target: <20 each. Highest-value split on the whole list; pure-function seam 2
is directly unit-testable.

### `src/quant_fund/backtest/engine.py:307` — `_run_backtest_event_loop`, cx 39, 354 lines

`If`=26, `For`=10, `IfExp`=10, `And`=5, `Except`=2, `Or`=3.

- `_dispatch_event(state, event)` — the event-type switch; the 10 `IfExp` are
  mostly event-type selection and belong here as an explicit mapping.
- `_advance_clock(state, event)` — time/calendar advance, extracted so ordering
  vs. dispatch is testable independently.
- `_settle_portfolio(state)` — the two `Except` handlers.

Watch: the 10 `IfExp` mean dispatch is currently inlined in the loop body.
Extracting it into a dict dispatch changes control flow shape — the golden
fill-sequence test must be written first.

### `src/quant_fund/portfolio/risk_gate.py:61` — `check_order`, cx 24, 72 lines

`If`=23, `Or`=2, `And`=1. Pure validation chain — the textbook case.

- `_pre_trade_checks(order, book)` → list of violation codes.
- `_size_checks(order, portfolio)` → violation codes.
- `_concentration_checks(order, portfolio)` → violation codes.

Each helper returns codes; `check_order` aggregates and preserves **rejection
order**. The ordering is the risk semantics: an order rejected for size *and*
concentration must report the same first reason it does today. The
table-driven test in the table above must assert first-reason identity, not just
set membership.

### `src/quant_fund/paper/ledger.py:348` — `promotion_dry_run`, cx 22, 120 lines

`If`=20, `And`=5, `Or`=5, `IfExp`=1, `For`=1.

- `_validate_promotion_preconditions(ledger, attestation)` → the 20 `If`
  as a precondition list.
- `_compute_promotion_digest(ledger)` → `And`/`Or`/hash conditions.

Fail-closed: every precondition must still be evaluated (short-circuit changes
are the risk). Assert the *set* of preconditions evaluated, not just the final
verdict.

### `src/quant_fund/execution/simulated_broker.py:412` — `SimulatedBroker._attempt_fill`, cx 13, 148 lines

`If`=11, `Return`=4, `IfExp`=3, `Or`=3, `And`=2, `Except`=1.

- `_match_orders(order, book)` — queue/price matching.
- `_decide_fill_quantity(matched, requested)` — the 3 `IfExp`.
- `_build_fill_record(matched, price, qty)` — the single `Return` + `Except`.

Lowest complexity of the money-path set but the highest blast radius. The
golden fill matrix must pass unchanged before and after.

### `src/quant_fund/labels/engine.py:17` — `build_labels`, cx 14, 134 lines

`If`=12, `For`=1, `IfExp`=1.

- `_label_from_window(window, horizon)` — per-bar labeling predicate.
- `_assemble_label_series(labels, index)` — alignment/forward-fill.

### `src/quant_fund/leakage/ast_scan.py:1049` — `_check_lh011`, cx 24, 92 lines

`If`=18, `IfExp`=5, `For`=3, `Or`=3, `While`=2, `And`=1.

- `_lh011_candidate_nodes(tree)` — node collection (the `For`/`While`).
- `_lh011_is_forbidden_receiver(node)` — the predicate chain; this is where the
  18 `If` and 5 `IfExp` belong. A regex/pattern list would collapse much of it.
- `_check_lh011(tree)` — orchestration, ends near cx 5.

**Mandatory characterization fixture first.** Each `if` is a distinct rule
variant; dropping a conjunct silently allows a leak. The `w1714` note that this
"needs characterization tests first" is correct and must be honored.

### `src/quant_fund/leakage/ast_scan.py:1144` — `_check_lh013`, cx 23, 133 lines

`If`=18, `And`=13, `Or`=7, `For`=2, `IfExp`=1, `Except`=1.

27 `BoolOp` operands across 18 `If` — this is a conjunction-heavy predicate.
- `_lh013_receiver_is_clean(node)` — the `And`×13 chain as one named predicate.
- `_lh013_context_nodes(tree)` — context collection.
- `_check_lh013(tree)` — orchestration.

Same mandatory-fixture requirement as `lh011`. Note both functions were
expanded by merge `25279246a`; the fixtures should cover the fail-closed
cases that merge introduced, since those branches have no prior coverage.

### Externally-caused (out of scope for this dossier)

| Function | cx | Note |
|---|---|---|
| `src/fx1/cli_audit.py:cli_audit` | 182 | `C901`-ignored in pyproject. Audit CLI, not money path. |
| `src/fx1/serve/api_audit.py:_probe_backend_probes` | 115 | `C901`-ignored. Audit probe harness. |
| `src/fx1/serve/parity_audit.py:parity_audit` | 82 | `C901`-ignored. Audit harness. |
| `src/quant_fund/paper/ledger.py:validate_ledger_schema` | 74 | Exactly at ceiling; the documented reference point. Not a violation. |

## Suggested order

1. Characterization tests for `_check_lh011` / `_check_lh013` (unblocks the
   two highest-value leakage splits).
2. `run_carry_backtest` — pure seam 2 gives the best complexity-per-risk ratio.
3. `check_order` — mechanical, and the table-driven test is cheap to write.
4. `promotion_dry_run`, `_attempt_fill`, `build_labels`.
5. `_run_backtest_event_loop` — last; its `IfExp`×10 makes it the easiest to
   subtly break.

After each refactor, lower the pin via `--write` (never by hand) and confirm the
complexity did not merely move into a new unlisted helper — `--allow-with-census`
must not become a way to re-hide debt in fresh functions.