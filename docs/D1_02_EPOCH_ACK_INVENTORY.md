# D1-02 epoch acknowledgment inventory

Agents list only. Owner decides remediate vs grandfather. Do not shrink this ledger without an owner ruling.

**Source:** `quality/epoch_allowed_removals.json` (`epoch_acknowledgments.v1`).
**Measured at:** `b5f1f6d11` via `python3` over the committed ledger.
**Count:** 46 acks (13 mutated, 33 removed). Two `tests/*` `removed` acks were added in `43602dd9d` (`formal/_test_sm_probe.py`, `test_hf_rv_scaffolding.py`).

## How the work-order numbers map

- **13 self-excused including 12 in-place mutations:** commit `84d3d31b7` added 13 acks: 12 `mutated` reseals of six committed lane receipts (two epochs) plus 1 quality-fork `removed`. Those 12 are in-place mutations of committed receipts.
- **13 mutated acks in the live ledger:** 1 `corpus_real_drill.json` regeneration (from the 5-entry evidence-repair genesis) plus the 12 lane reseals.
- **39 ungated acks after the genesis 5:** that count is 44−5. Live ledger is now 46; genesis at `02fbf8612` had 5; 39 were added after genesis through the earlier inventory, then two `tests/*` removals in `43602dd9d`. `84d3d31b7` itself added 13, not 39.

ZR-9V has not landed a separate inventory. This is the integrator list.

## Genesis 5 (`02fbf8612`)

| # | transition | corpus | member | epoch |
|---|------------|--------|--------|-------|
| 1 | `removed` | `receipts/*.json` | `corpus_history_absence_1791a18109fce973.json` | `corpus_epoch_a4cb68008a317f90.json` |
| 2 | `mutated` | `receipts/*.json` | `corpus_real_drill.json` | `corpus_epoch_2bd794275af2b3d8.json` |
| 3 | `removed` | `quality/*.json` | `fuzz_drill_b966c0d125b2a999.json` | `corpus_epoch_d7ca8a37b299e9e8.json` |
| 4 | `removed` | `quality/*.json` | `fuzz_drill_cb1a0f56a4e67716.json` | `corpus_epoch_d7ca8a37b299e9e8.json` |
| 5 | `removed` | `web/*` | `public/fixtures/receipts/fast_replay_p42_conformance_20260927.json` | `corpus_epoch_326bc689d818d49e.json` |

## Added in `84d3d31b7` (13)

| transition | corpus | member | epoch |
|------------|--------|--------|-------|
| `mutated` | `receipts/*.json` | `capacity_eval_3622d0c059d42009.json` | `corpus_epoch_05bbdbfc2bab9f19.json` |
| `mutated` | `receipts/*.json` | `fleet_eval_5e907be710811a44.json` | `corpus_epoch_05bbdbfc2bab9f19.json` |
| `mutated` | `receipts/*.json` | `honest_verdict_b133e8b3992af893.json` | `corpus_epoch_05bbdbfc2bab9f19.json` |
| `mutated` | `receipts/*.json` | `monitor_run_13d20aef00774027.json` | `corpus_epoch_05bbdbfc2bab9f19.json` |
| `mutated` | `receipts/*.json` | `rankic_eval_8d740c0ddf6d5c54.json` | `corpus_epoch_05bbdbfc2bab9f19.json` |
| `mutated` | `receipts/*.json` | `vol_bench_f3bf2afc2f159ace.json` | `corpus_epoch_05bbdbfc2bab9f19.json` |
| `mutated` | `receipts/*.json` | `capacity_eval_3622d0c059d42009.json` | `corpus_epoch_6f6d75cd3751c48d.json` |
| `mutated` | `receipts/*.json` | `fleet_eval_5e907be710811a44.json` | `corpus_epoch_6f6d75cd3751c48d.json` |
| `mutated` | `receipts/*.json` | `honest_verdict_b133e8b3992af893.json` | `corpus_epoch_6f6d75cd3751c48d.json` |
| `mutated` | `receipts/*.json` | `monitor_run_13d20aef00774027.json` | `corpus_epoch_6f6d75cd3751c48d.json` |
| `mutated` | `receipts/*.json` | `rankic_eval_8d740c0ddf6d5c54.json` | `corpus_epoch_6f6d75cd3751c48d.json` |
| `mutated` | `receipts/*.json` | `vol_bench_f3bf2afc2f159ace.json` | `corpus_epoch_6f6d75cd3751c48d.json` |
| `removed` | `quality/*.json` | `corpus_epoch_cd4b7ee4ad32d328.json` | `corpus_epoch_792426f656af6eb2.json` |

## All live acks

| # | genesis? | transition | corpus | member | epoch | reason (truncated) |
|---|----------|------------|--------|--------|-------|--------------------|
| 1 | yes | `removed` | `receipts/*.json` | `corpus_history_absence_1791a18109fce973.json` | `corpus_epoch_a4cb68008a317f90.json` | Transient member, never committed to git: first stamped at corpus_epoch_7d6b1f5442b13ea1.json and removed at the next epoch, both stamped from b175c98f3b ('feat |
| 2 | yes | `mutated` | `receipts/*.json` | `corpus_real_drill.json` | `corpus_epoch_2bd794275af2b3d8.json` | Recorded in-place regeneration: corpus_inference.v1 output first committed in b07640432 (082f1623...) and regenerated in place by d291ddb5e ('replay manifests b |
| 3 | yes | `removed` | `quality/*.json` | `fuzz_drill_b966c0d125b2a999.json` | `corpus_epoch_d7ca8a37b299e9e8.json` | Removed at the epoch stamped from 950df4b3a2 ('chore(integrity): drop pre-fix fuzz receipts - canonical calibrated receipt follows') and later re-added with byt |
| 4 | yes | `removed` | `quality/*.json` | `fuzz_drill_cb1a0f56a4e67716.json` | `corpus_epoch_d7ca8a37b299e9e8.json` | Removed at the epoch stamped from 950df4b3a2 ('chore(integrity): drop pre-fix fuzz receipts - canonical calibrated receipt follows') and later re-added with byt |
| 5 | yes | `removed` | `web/*` | `public/fixtures/receipts/fast_replay_p42_conformance_20260927.json` | `corpus_epoch_326bc689d818d49e.json` | Web-explorer fixture copy of receipts/legacy-unsealed/fast_replay_p42_conformance_20260927.json (identical digest; a quarantined pre-seal receipt) removed from  |
| 6 | no | `mutated` | `receipts/*.json` | `capacity_eval_3622d0c059d42009.json` | `corpus_epoch_05bbdbfc2bab9f19.json` | Replay artifact path+digest refresh (faafaf040); seal updated in follow-up. |
| 7 | no | `mutated` | `receipts/*.json` | `fleet_eval_5e907be710811a44.json` | `corpus_epoch_05bbdbfc2bab9f19.json` | Replay artifact path+digest refresh (faafaf040); seal updated in follow-up. |
| 8 | no | `mutated` | `receipts/*.json` | `honest_verdict_b133e8b3992af893.json` | `corpus_epoch_05bbdbfc2bab9f19.json` | Replay artifact path+digest refresh (faafaf040); seal updated in follow-up. |
| 9 | no | `mutated` | `receipts/*.json` | `monitor_run_13d20aef00774027.json` | `corpus_epoch_05bbdbfc2bab9f19.json` | Replay artifact path+digest refresh (faafaf040); seal updated in follow-up. |
| 10 | no | `mutated` | `receipts/*.json` | `rankic_eval_8d740c0ddf6d5c54.json` | `corpus_epoch_05bbdbfc2bab9f19.json` | Replay artifact path+digest refresh (faafaf040); seal updated in follow-up. |
| 11 | no | `mutated` | `receipts/*.json` | `vol_bench_f3bf2afc2f159ace.json` | `corpus_epoch_05bbdbfc2bab9f19.json` | Replay artifact path+digest refresh (faafaf040); seal updated in follow-up. |
| 12 | no | `mutated` | `receipts/*.json` | `capacity_eval_3622d0c059d42009.json` | `corpus_epoch_6f6d75cd3751c48d.json` | Reseal receipt_sha256 after replay binding update; head bytes match lane replay pins. |
| 13 | no | `mutated` | `receipts/*.json` | `fleet_eval_5e907be710811a44.json` | `corpus_epoch_6f6d75cd3751c48d.json` | Reseal receipt_sha256 after replay binding update; head bytes match lane replay pins. |
| 14 | no | `mutated` | `receipts/*.json` | `honest_verdict_b133e8b3992af893.json` | `corpus_epoch_6f6d75cd3751c48d.json` | Reseal receipt_sha256 after replay binding update; head bytes match lane replay pins. |
| 15 | no | `mutated` | `receipts/*.json` | `monitor_run_13d20aef00774027.json` | `corpus_epoch_6f6d75cd3751c48d.json` | Reseal receipt_sha256 after replay binding update; head bytes match lane replay pins. |
| 16 | no | `mutated` | `receipts/*.json` | `rankic_eval_8d740c0ddf6d5c54.json` | `corpus_epoch_6f6d75cd3751c48d.json` | Reseal receipt_sha256 after replay binding update; head bytes match lane replay pins. |
| 17 | no | `mutated` | `receipts/*.json` | `vol_bench_f3bf2afc2f159ace.json` | `corpus_epoch_6f6d75cd3751c48d.json` | Reseal receipt_sha256 after replay binding update; head bytes match lane replay pins. |
| 18 | no | `removed` | `quality/*.json` | `corpus_epoch_cd4b7ee4ad32d328.json` | `corpus_epoch_792426f656af6eb2.json` | Drop forked quality epoch tip (parallel stamp from faafaf040); canonical chain follows e31cd642→cf04. |
| 19 | no | `removed` | `tests/*` | `corpus_epoch_05724f90cfec84f6.json` | `corpus_epoch_446ddadac2efb9d5.json` | Removed orphaned fork tip that branched from corpus_epoch_fd8b09ac86c69256.json alongside the canonical heads-pin chain through corpus_epoch_68f3489ea52c4bcf.js |
| 20 | no | `removed` | `src/*` | `quant_fund/proof/promotion_receipt.py` | `corpus_epoch_52900236a145cedd.json` | Historical src member removal already sealed in corpus_epoch_52900236a145cedd.json; acknowledged so evidence-audit --epoch-acks can pass without rewriting the a |
| 21 | no | `removed` | `src/*` | `quant_fund/schemas/receipt.py` | `corpus_epoch_52900236a145cedd.json` | Historical src member removal already sealed in corpus_epoch_52900236a145cedd.json; acknowledged so evidence-audit --epoch-acks can pass without rewriting the a |
| 22 | no | `removed` | `scripts/*` | `scripts/README.md` | `corpus_epoch_55b9336a9b3bc833.json` | Script path removed/relocated at corpus_epoch_55b9336a9b3bc833.json; acknowledged for evidence-audit epoch-acks. |
| 23 | no | `removed` | `scripts/*` | `scripts/benchmark_crypto_external.py` | `corpus_epoch_55b9336a9b3bc833.json` | Script path removed/relocated at corpus_epoch_55b9336a9b3bc833.json; acknowledged for evidence-audit epoch-acks. |
| 24 | no | `removed` | `scripts/*` | `scripts/benchmark_garch.py` | `corpus_epoch_55b9336a9b3bc833.json` | Script path removed/relocated at corpus_epoch_55b9336a9b3bc833.json; acknowledged for evidence-audit epoch-acks. |
| 25 | no | `removed` | `scripts/*` | `scripts/benchmark_paper_loop.py` | `corpus_epoch_55b9336a9b3bc833.json` | Script path removed/relocated at corpus_epoch_55b9336a9b3bc833.json; acknowledged for evidence-audit epoch-acks. |
| 26 | no | `removed` | `scripts/*` | `scripts/keep_awake_for_agents.sh` | `corpus_epoch_55b9336a9b3bc833.json` | Script path removed/relocated at corpus_epoch_55b9336a9b3bc833.json; acknowledged for evidence-audit epoch-acks. |
| 27 | no | `removed` | `scripts/*` | `scripts/verify_binance_receipt.py` | `corpus_epoch_55b9336a9b3bc833.json` | Script path removed/relocated at corpus_epoch_55b9336a9b3bc833.json; acknowledged for evidence-audit epoch-acks. |
| 28 | no | `removed` | `scripts/*` | `scripts/verify_bugbot_findings.py` | `corpus_epoch_55b9336a9b3bc833.json` | Script path removed/relocated at corpus_epoch_55b9336a9b3bc833.json; acknowledged for evidence-audit epoch-acks. |
| 29 | no | `removed` | `scripts/*` | `scripts/verify_external_receipt.py` | `corpus_epoch_55b9336a9b3bc833.json` | Script path removed/relocated at corpus_epoch_55b9336a9b3bc833.json; acknowledged for evidence-audit epoch-acks. |
| 30 | no | `removed` | `data/metadata/*` | `resolved_config.json` | `corpus_epoch_e73c101a7ab5d51b.json` | Gitignored derived config (data/metadata/**). It was stamped from a working tree that still had the file, so a clean CI checkout failed head_member_missing_live |
| 31 | no | `removed` | `docs/*` | `corpus_epoch_bf34205674030d20.json` | `corpus_epoch_56ab5b59290569ce.json` | Parallel docs stamps forked at corpus_epoch_5ec65ddd85b8b6f9.json. The pinned line matches the live tree. This side-line receipt is removed so the chain has one |
| 32 | no | `removed` | `docs/*` | `corpus_epoch_6b3d85be9a477735.json` | `corpus_epoch_56ab5b59290569ce.json` | Continuation of the unpinned docs fork. Removed with that side line so check_epoch_chain sees one head. The bytes remain in git history. |
| 33 | no | `removed` | `receipts/*.json` | `capacity_eval_3622d0c059d42009.json` | `corpus_epoch_1d1a6cfdadf37c0c.json` | Seal-prefix rename to capacity_eval_5cf0042e91c7953c.json at corpus_epoch_1d1a6cfdadf37c0c.json. 84d3d31b7 refreshed receipt_sha256 in place after artifact-bind |
| 34 | no | `removed` | `receipts/*.json` | `fleet_eval_5e907be710811a44.json` | `corpus_epoch_1d1a6cfdadf37c0c.json` | Seal-prefix rename to fleet_eval_ab613c423b36627d.json at corpus_epoch_1d1a6cfdadf37c0c.json. 84d3d31b7 refreshed receipt_sha256 in place after artifact-binding |
| 35 | no | `removed` | `receipts/*.json` | `honest_verdict_b133e8b3992af893.json` | `corpus_epoch_1d1a6cfdadf37c0c.json` | Seal-prefix rename to honest_verdict_9d631fa6aace437a.json at corpus_epoch_1d1a6cfdadf37c0c.json. 84d3d31b7 refreshed receipt_sha256 in place after artifact-bin |
| 36 | no | `removed` | `receipts/*.json` | `monitor_run_13d20aef00774027.json` | `corpus_epoch_1d1a6cfdadf37c0c.json` | Seal-prefix rename to monitor_run_e69a3aed9568e30d.json at corpus_epoch_1d1a6cfdadf37c0c.json. 84d3d31b7 refreshed receipt_sha256 in place after artifact-bindin |
| 37 | no | `removed` | `receipts/*.json` | `rankic_eval_8d740c0ddf6d5c54.json` | `corpus_epoch_1d1a6cfdadf37c0c.json` | Seal-prefix rename to rankic_eval_addd924f107eedd8.json at corpus_epoch_1d1a6cfdadf37c0c.json. 84d3d31b7 refreshed receipt_sha256 in place after artifact-bindin |
| 38 | no | `removed` | `receipts/*.json` | `vol_bench_f3bf2afc2f159ace.json` | `corpus_epoch_1d1a6cfdadf37c0c.json` | Seal-prefix rename to vol_bench_dc16af46c3f2975b.json at corpus_epoch_1d1a6cfdadf37c0c.json. 84d3d31b7 refreshed receipt_sha256 in place after artifact-binding  |
| 39 | no | `removed` | `web/*` | `public/fixtures/receipts/capacity_eval_3622d0c059d42009.json` | `corpus_epoch_12bce57f7c47989b.json` | Fixture copy of a lane receipt renamed onto its seal prefix. Removed at corpus_epoch_12bce57f7c47989b.json; the new fixture name is the seal-prefixed receipt. |
| 40 | no | `removed` | `web/*` | `public/fixtures/receipts/fleet_eval_5e907be710811a44.json` | `corpus_epoch_12bce57f7c47989b.json` | Fixture copy of a lane receipt renamed onto its seal prefix. Removed at corpus_epoch_12bce57f7c47989b.json; the new fixture name is the seal-prefixed receipt. |
| 41 | no | `removed` | `web/*` | `public/fixtures/receipts/honest_verdict_b133e8b3992af893.json` | `corpus_epoch_12bce57f7c47989b.json` | Fixture copy of a lane receipt renamed onto its seal prefix. Removed at corpus_epoch_12bce57f7c47989b.json; the new fixture name is the seal-prefixed receipt. |
| 42 | no | `removed` | `web/*` | `public/fixtures/receipts/monitor_run_13d20aef00774027.json` | `corpus_epoch_12bce57f7c47989b.json` | Fixture copy of a lane receipt renamed onto its seal prefix. Removed at corpus_epoch_12bce57f7c47989b.json; the new fixture name is the seal-prefixed receipt. |
| 43 | no | `removed` | `web/*` | `public/fixtures/receipts/rankic_eval_8d740c0ddf6d5c54.json` | `corpus_epoch_12bce57f7c47989b.json` | Fixture copy of a lane receipt renamed onto its seal prefix. Removed at corpus_epoch_12bce57f7c47989b.json; the new fixture name is the seal-prefixed receipt. |
| 44 | no | `removed` | `web/*` | `public/fixtures/receipts/vol_bench_f3bf2afc2f159ace.json` | `corpus_epoch_12bce57f7c47989b.json` | Fixture copy of a lane receipt renamed onto its seal prefix. Removed at corpus_epoch_12bce57f7c47989b.json; the new fixture name is the seal-prefixed receipt. |

