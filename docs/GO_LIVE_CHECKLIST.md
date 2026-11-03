# Go-live checklist

The five conditions in `quant_fund.validation.live_readiness.live_readiness_report`.
`make prod-readiness` reports them as `live_condition_1` through
`live_condition_5`. Each stays `needs-owner` until the evidence below exists
outside this checkout. A module, a SYNTHETIC fixture, a paper session, or
this document does not satisfy a condition. `eligible` stays false,
`would_promote_live` stays false, and `live_pnl_claim` stays false.

Print the recorded block:

```bash
uv run python -c "from quant_fund.validation.live_readiness import format_readiness; print(format_readiness())"
```

Procurement detail lives in [EVIDENCE_PROCUREMENT.md](EVIDENCE_PROCUREMENT.md).
The condition anchors live in [INSTITUTIONAL_READINESS.md](INSTITUTIONAL_READINESS.md).

## 1. `licensed_point_in_time_data` — BLOCKED

Evidence: a licensed point-in-time equity extract, a written
universe-completeness attestation that includes delisted names, and
corporate-action adjustment lineage. Every row carries `event_time`,
`available_time`, `ingested_time`, `source_id`, and `revision_id`, with
`event_time <= available_time <= ingested_time` and
`available_time <= decision_time`. The hashed PIT receipt reproduces.

Who: Procurement and Compliance supply the license and the two letters.
Engineering points the entitlement at that feed.

Verify the extract with `quant_fund.data.qualifying.evaluate_feed`. The
verdict must have `status` `QUALIFYING`, `qualifying` true,
`synthetic_substitution` false, and `live_pnl_claim` false. The gate's
regression, which locks that rule and is not itself the license, is:

```bash
uv run pytest -q tests/unit/data/test_qualifying_feed.py
```

`qualifying_feed_closed` on the scorecard only checks that a missing
entitlement stays `UNAVAILABLE`. That pass is not condition 1.

## 2. `authenticated_broker_reconciliation` — BLOCKED

Evidence: an authenticated broker fills export in which every fill maps to
one order, a duplicate `fill_id` is idempotent, a missing fill and an orphan
fill are reported, and position and cash match the broker statement inside
the agreed tolerance.

Who: Compliance and Trading authorize the brokerage account and the
drop-copy. Engineering reconciles that export.

Verify the reference shape:

```bash
uv run pytest -q tests/unit/execution/test_order_recon.py tests/unit/paper/test_broker_adapter.py
```

Reconcile the broker export with `OrderFillLedger.reconcile` in
`quant_fund.execution.order_recon`. `dipcatcher trade session` is the
simulated book. The scorecard row `live_endpoint_refused` checks that a
live endpoint is refused. Neither command is condition 2.

## 3. `non_synthetic_forward_record` — NOT_YET_COLLECTED

Evidence: the `forward_2026H2` journal. The 2025 holdout is spent. The
pre-registration was RFC 3161 anchored on 2026-10-08. The first accepted
decision falls on a market date after the 2026-10-07 freeze. Warmup is bars
on or before 2026-09-18. `window.start` 2026-07-01 is the calendar span
label, not the first eligible session.

Who: Quant collects the eligible sessions after the freeze. Compliance
confirms the sealed pre-registration was not edited once outcomes exist.

Verify the journal:

```bash
uv run dipcatcher forward-shadow reconcile --run <journal> --expected-head <digest>
```

Exit 0 means the chain is intact and the head matches. A collected record
also has to be non-synthetic and inside the eligible window; the reconcile
command checks the chain, and the reviewer checks the dates.

```bash
uv run dipcatcher reality sweep
```

exits 4 with `REALITY_SWEEP_FETCH_UNAVAILABLE` when the fetch fails, and
does not substitute synthetic bars. That exit is the guard. It is not a
collected forward record.

## 4. `venue_cost_liquidity_borrow_financing` — BLOCKED

Evidence: five dated measurements from the target venue, each with a
source: effective cost, liquidity at the intended size, borrow availability
and fee, financing rate, and failure modes (rejects, halts, partial fills,
auctions, feed gaps, order-rate limits).

Who: Trading and Quant record the measurements. An independent reviewer
signs the packet.

Verify: compare the packet to section 3 of
[EVIDENCE_PROCUREMENT.md](EVIDENCE_PROCUREMENT.md). All five rows are dated
and sourced. `configs/backtest.yaml` and the following command are planning
inputs, not venue evidence:

```bash
uv run dipcatcher execution-sensitivity
```

`make prod-readiness` keeps `live_condition_4` at `needs-owner`.

## 5. `signed_promotion_and_live_authorization` — ABSENT

Evidence: a signed promotion receipt that the immutable verifier accepts,
and a separate written live authorization that names the principal. The
authorization is composed only after conditions 1–4 and the forward record
exist.

Who: Quant and an independent reviewer produce the receipt. Compliance and
the principal sign the authorization.

Verify the receipt machinery, and any receipt that is already on disk:

```bash
uv run dipcatcher verify-research
uv run dipcatcher verify-receipt <receipt>
make receipts-reverify
make evidence-audit
make checkpoint-chain
make verify-rotations
```

These commands check signatures and chains. They do not write the live
authorization, and they do not set `would_promote_live`.
