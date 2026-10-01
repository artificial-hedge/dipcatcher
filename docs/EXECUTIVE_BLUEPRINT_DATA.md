# Executive blueprint: data and entitlement audit

This is a bounded current-host inventory on 2026-10-01, not legal clearance or
a claim that every remote store/account was searched. It advances Q06 of the
[execution plan](EXECUTIVE_BLUEPRINT_PLAN.md); D04–D08 and A2–A4 remain open.
No data purchases, partner outreach, credential changes or feed subscriptions
were made. Source/catalog availability, data presence and permitted use are
separate questions.

| Inspected input | Current evidence | Rights / availability limit | Accepted use in this workstream |
|---|---|---|---|
| Default `data/metadata/data_manifest.json` | Manifest declares `source=synthetic`; default bars include a planted signal. | A generated fixture provides no market evidence. | Correctness testing only. |
| `data/file_us_wide` | Local Yahoo vendor-adjusted snapshot used by the saved GCN/DDPM runs; source/data hashes and causal-selection audits are in their receipts. | Commercial redistribution rights not established. First-published adjustment vintages, corporate-action completeness and observed availability are unverified; surviving pool and previously inspected tail are explicit. | Exploratory retrospective experiments with adverse results retained; no fresh OOS or promotion claim. |
| QuantCode-Bench import | Pinned upstream revision `f8bda951addb409a81aa316c00401dbde60774ae`, 400 tasks plus data requirements, hashes retained under `data/fx1/quantcode`. | Repository MIT license does not establish every external dataset/dependency right. Backtrader GPL and judge/runtime equivalence require separate review. | Task/protocol import and correctness tests. Zero actual model evaluations. |
| Parent-order synthetic tapes | Raw scenarios, seeds, source snapshots, model and policy traces saved under `data/metadata/blueprint_execution/seed7_fixed_v1`. | No real LOB/auction entitlement, queue calibration or counterfactual fill validation. | Synthetic DQN representation/ledger tests only. |
| Options/off-exchange/router/multimodal/meta/continual/anomaly tests | Typed clocks, source identities and explicit synthetic fixtures; annotation provenance propagates where applicable. | Independent real intent/event labels, permitted tape/news/image uses and historical coverage are not demonstrated by tests. | Correctness tests and bounded offline interfaces. |
| Remote Kimi-K3 base files | Read-only Windows inventory: 96 shards, about 1.56 TB logical size and config identity; inventory under `data/metadata/blueprint_remote_inventory_20261001.json`. | Weight checksums/integrity, applicable rights, compatible inference, trained fx-1 checkpoint and adequate cluster support are unverified. | Inventory only; no training or inference claim. |

## Current local datasource readiness

`fx1 sources probe` was run using the existing local environment. It only checks
script/credential availability; it does not fetch data or verify entitlement.
The configured root `/app/.agents/plugins` does not exist on this Mac and no
root override was configured. Of 18 catalog sources, **17 returned `no_script`
and one returned `mcp_required`**. The bounded search of the local agent plugin
directory found none of the selected Wind/Yahoo/IMF/CLS script names.

The result is saved as
`data/metadata/blueprint_source_probes_20261001.json`. This does not establish
unavailability on another host or through an independent connected service.
The existing datasource documentation's earlier live receipts remain historical;
they do not prove current execution in this process. No secret values were
inspected or recorded.

## Missing qualifying datasets and decisions

| Requirement | Evidence needed before empirical acceptance |
|---|---|
| D04 strategy/code model | Corpus and base-weight permitted-use register, contamination/split audit, actual candidate training/checkpoint and benchmark run. |
| D05 execution/router | Authorized full depth/order events, auction rules, venue order/ack/fill logs, event/receive clocks and finalized outcome labels. |
| D05 options/off-exchange/order flow | Authorized trade/quote conditions and revisions, timed identifiers, independently sourced annotation conventions, holdout events and realistic negative examples. |
| D06 narrative/multimodal/economic/alternative data | Identified text/visual/macro/alternative sources, release and ingest timestamps, vintages/revisions, entity links, missingness and redistribution/training rights. |
| D07 credit/federation | Authorized exposure/default histories and client permissions, isolation/threat model and leakage audit. Local client simulation proves no partnership or privacy. |
| D08 hardware | Authorized named device/backend and measured run. Classical annealing proves neither device access nor quantum advantage. |

For each candidate, record the underlying provider/version/content hashes,
coverage/universe, permitted research/training/serving/redistribution uses,
contract evidence, clock definitions and correction policy. An owner or provider
must supply actual entitlements where they are required. A registry entry, public
sample, successful fetch or content hash alone does not establish those rights.
Missing authorization remains explicit; synthetic or proxy replacements cannot
close the empirical requirement.
