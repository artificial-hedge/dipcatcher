# Ungated epoch acknowledgments since `84d3d31b7`

Owner review list. These acknowledgments were **not reverted**.

## Why they went ungated

`84d3d31b7` (`fix(integrity): reseal lane receipts and repair quality epoch
fork`) is the first commit that grew `quality/epoch_allowed_removals.json`
past the five migrated 2026-10-13 entries. Direct pushes to `main` resolve
the admission/ack/chain base from `github.event.before` — the previous
push tip. When that push's evidence-audit run was cancelled or superseded
before those gates ran, the next push gated only `event.before..HEAD` and
never walked the cancelled range. Epoch-ack-gate therefore never saw the
new ledger entries against the last completed green evidence SHA.

Count at HEAD vs `84d3d31b7^`: **39** acknowledgments added, **5**
pre-existing (migrated 2026-10-13) unchanged. Ledger path:
`quality/epoch_allowed_removals.json`.

Pin signing is human-only (`gate_pins.sig` stale since 2026-09-30). This
list does not restamp or re-sign anything.

The evidence `base-ref` target now prefers the last completed green
evidence SHA (CI artifact `evidence-base-sha`) over a cancelled
`event.before`, so a later push is gated over the whole unverified range.

## The 39 entries

| # | corpus | member | transition | epoch |
|---|---|---|---|---|
| 1 | `receipts/*.json` | `capacity_eval_3622d0c059d42009.json` | `mutated` | `corpus_epoch_05bbdbfc2bab9f19.json` |
| 2 | `receipts/*.json` | `fleet_eval_5e907be710811a44.json` | `mutated` | `corpus_epoch_05bbdbfc2bab9f19.json` |
| 3 | `receipts/*.json` | `honest_verdict_b133e8b3992af893.json` | `mutated` | `corpus_epoch_05bbdbfc2bab9f19.json` |
| 4 | `receipts/*.json` | `monitor_run_13d20aef00774027.json` | `mutated` | `corpus_epoch_05bbdbfc2bab9f19.json` |
| 5 | `receipts/*.json` | `rankic_eval_8d740c0ddf6d5c54.json` | `mutated` | `corpus_epoch_05bbdbfc2bab9f19.json` |
| 6 | `receipts/*.json` | `vol_bench_f3bf2afc2f159ace.json` | `mutated` | `corpus_epoch_05bbdbfc2bab9f19.json` |
| 7 | `receipts/*.json` | `capacity_eval_3622d0c059d42009.json` | `mutated` | `corpus_epoch_6f6d75cd3751c48d.json` |
| 8 | `receipts/*.json` | `fleet_eval_5e907be710811a44.json` | `mutated` | `corpus_epoch_6f6d75cd3751c48d.json` |
| 9 | `receipts/*.json` | `honest_verdict_b133e8b3992af893.json` | `mutated` | `corpus_epoch_6f6d75cd3751c48d.json` |
| 10 | `receipts/*.json` | `monitor_run_13d20aef00774027.json` | `mutated` | `corpus_epoch_6f6d75cd3751c48d.json` |
| 11 | `receipts/*.json` | `rankic_eval_8d740c0ddf6d5c54.json` | `mutated` | `corpus_epoch_6f6d75cd3751c48d.json` |
| 12 | `receipts/*.json` | `vol_bench_f3bf2afc2f159ace.json` | `mutated` | `corpus_epoch_6f6d75cd3751c48d.json` |
| 13 | `quality/*.json` | `corpus_epoch_cd4b7ee4ad32d328.json` | `removed` | `corpus_epoch_792426f656af6eb2.json` |
| 14 | `tests/*` | `corpus_epoch_05724f90cfec84f6.json` | `removed` | `corpus_epoch_446ddadac2efb9d5.json` |
| 15 | `src/*` | `quant_fund/proof/promotion_receipt.py` | `removed` | `corpus_epoch_52900236a145cedd.json` |
| 16 | `src/*` | `quant_fund/schemas/receipt.py` | `removed` | `corpus_epoch_52900236a145cedd.json` |
| 17 | `scripts/*` | `scripts/README.md` | `removed` | `corpus_epoch_55b9336a9b3bc833.json` |
| 18 | `scripts/*` | `scripts/benchmark_crypto_external.py` | `removed` | `corpus_epoch_55b9336a9b3bc833.json` |
| 19 | `scripts/*` | `scripts/benchmark_garch.py` | `removed` | `corpus_epoch_55b9336a9b3bc833.json` |
| 20 | `scripts/*` | `scripts/benchmark_paper_loop.py` | `removed` | `corpus_epoch_55b9336a9b3bc833.json` |
| 21 | `scripts/*` | `scripts/keep_awake_for_agents.sh` | `removed` | `corpus_epoch_55b9336a9b3bc833.json` |
| 22 | `scripts/*` | `scripts/verify_binance_receipt.py` | `removed` | `corpus_epoch_55b9336a9b3bc833.json` |
| 23 | `scripts/*` | `scripts/verify_bugbot_findings.py` | `removed` | `corpus_epoch_55b9336a9b3bc833.json` |
| 24 | `scripts/*` | `scripts/verify_external_receipt.py` | `removed` | `corpus_epoch_55b9336a9b3bc833.json` |
| 25 | `data/metadata/*` | `resolved_config.json` | `removed` | `corpus_epoch_e73c101a7ab5d51b.json` |
| 26 | `docs/*` | `corpus_epoch_bf34205674030d20.json` | `removed` | `corpus_epoch_56ab5b59290569ce.json` |
| 27 | `docs/*` | `corpus_epoch_6b3d85be9a477735.json` | `removed` | `corpus_epoch_56ab5b59290569ce.json` |
| 28 | `receipts/*.json` | `capacity_eval_3622d0c059d42009.json` | `removed` | `corpus_epoch_1d1a6cfdadf37c0c.json` |
| 29 | `receipts/*.json` | `fleet_eval_5e907be710811a44.json` | `removed` | `corpus_epoch_1d1a6cfdadf37c0c.json` |
| 30 | `receipts/*.json` | `honest_verdict_b133e8b3992af893.json` | `removed` | `corpus_epoch_1d1a6cfdadf37c0c.json` |
| 31 | `receipts/*.json` | `monitor_run_13d20aef00774027.json` | `removed` | `corpus_epoch_1d1a6cfdadf37c0c.json` |
| 32 | `receipts/*.json` | `rankic_eval_8d740c0ddf6d5c54.json` | `removed` | `corpus_epoch_1d1a6cfdadf37c0c.json` |
| 33 | `receipts/*.json` | `vol_bench_f3bf2afc2f159ace.json` | `removed` | `corpus_epoch_1d1a6cfdadf37c0c.json` |
| 34 | `web/*` | `public/fixtures/receipts/capacity_eval_3622d0c059d42009.json` | `removed` | `corpus_epoch_12bce57f7c47989b.json` |
| 35 | `web/*` | `public/fixtures/receipts/fleet_eval_5e907be710811a44.json` | `removed` | `corpus_epoch_12bce57f7c47989b.json` |
| 36 | `web/*` | `public/fixtures/receipts/honest_verdict_b133e8b3992af893.json` | `removed` | `corpus_epoch_12bce57f7c47989b.json` |
| 37 | `web/*` | `public/fixtures/receipts/monitor_run_13d20aef00774027.json` | `removed` | `corpus_epoch_12bce57f7c47989b.json` |
| 38 | `web/*` | `public/fixtures/receipts/rankic_eval_8d740c0ddf6d5c54.json` | `removed` | `corpus_epoch_12bce57f7c47989b.json` |
| 39 | `web/*` | `public/fixtures/receipts/vol_bench_f3bf2afc2f159ace.json` | `removed` | `corpus_epoch_12bce57f7c47989b.json` |

Full `from_sha256` / `to_sha256` / `reason` fields remain in the ledger.
Owner action: review each row against the named epoch; do not treat this
list as an admission verdict.
