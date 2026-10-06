# Executive blueprint: runnable pilots and measured evidence

This is the implementation record for the [28-feature execution plan](EXECUTIVE_BLUEPRINT_PLAN.md).
The [source audit](EXECUTIVE_BLUEPRINT_SOURCES.md) distinguishes published results,
source licenses and unresolved references. The full PDF goal remains incomplete.
The [data audit](EXECUTIVE_BLUEPRINT_DATA.md) records local source readiness,
inspected datasets and missing entitlement/vintage evidence.
These pilots do not establish a trained fx-1 strategy model, SOTA, data rights,
live readiness, or completion of every blueprint feature.

## Implemented scope

| Feature | Actual new behavior | Evidence and remaining scope |
|---|---|---|
| Strategy code | Caller-supplied inference backend, bounded AST interpreter, causal close-direction replay, proper scores, immutable feedback receipt and replay verifier. CLI and HTTP replay. | Hostile-code/PIT/resource/receipt tests. No general Python/Backtrader worker, independent semantic judge, trained checkpoint or real generation run in this workstream. |
| Narrative Say–Echo–Do | Actual return-aligned neighbor-loss token/projection learning, timed statement/echo/revision snapshots, separate later direction calibration, supplied voice/position proxy diagnostics and safe frozen JSON persistence. | 39 synthetic tests, independently calculated Brier/log/ECE and adversarial rehashed causal metadata checks. No pretrained transformer, authenticated corpus, independent labels, empirical benefit or full paper reproduction; missing Do and intent stay unknown. CLI/API remain open. |
| Smart router | Trained logistic fill model, Gaussian ridge toxicity model, strictly timed inputs/finalized labels and capacity-constrained linear child allocation. CLI and HTTP pilot. | Synthetic learning and independent allocation/score checks. No licensed venue logs, nonlinear size/impact, counterfactual execution validation or venue submission. |
| Asset GNN | Actual two-layer Gaussian GCN and same-capacity node-only training; frozen PIT correlation/sector graph, induction onto unseen IDs/subgraphs, saved experimental arrays. | 34 model tests and 10 experimental protocol tests. One adverse exploratory historical run below. Dynamic graphs, supply-chain data and prospective evidence remain open. |
| Joint diffusion | Actual joint horizon-by-asset epsilon denoiser, cosine DDPM schedule, train-only scaling/support, NumPy reverse sampling, training/model hashes and descriptive distribution diagnostics. | 45 model tests and 34 protocol/evidence tests. One adverse fixed historical experiment below. Conditional volume/LOB/event generation, TSTR and extreme-tail validation remain open. |
| Multimodal forecast | Learned token/numeric self-attention and cross-attention, optional identified visual vectors, explicit publication delays, train-only vocabulary/scaling, frozen Gaussian forecasts and ablations. | 22 synthetic tests. This small encoder is not pretrained FinGPT/BERT. Licensed text/image data, independent empirical comparisons and serving/persistence remain open. |
| Options flow | Typed option tape/PIT parser, BSM Greeks, trailing features, trained logistic/tree ensemble and separate chronological Platt calibration; Brier/log/ECE, local alerts. | 47 synthetic tests. Scores predict supplied annotation semantics. No independently verified informed intent, empirical tape, multi-leg support, stream or UI. |
| Off-exchange flow | Explicit UTP/CTS-version condition mapping, delayed attribution, as-of cancellation/correction handling, quote-proxy statistics, trained annotated phase classifier and later calibration. | 29 synthetic tests. Normalized fields are not a raw SIP decoder. ATS/non-ATS/unknown are distinguished; buyer intent stays unknown. Rights and independent labels remain external requirements. |
| QuantCode benchmark | Exact pinned 400-task and data-requirement importer, task/code/data/environment bindings and fail-closed injected execution/judge protocols. | 31 adapter tests and actual complete import. Zero model evaluations; official harness equivalence and isolated execution remain unverified. |
| Meta-learning allocation | Exact second-order Gaussian MAML with causal support-only scaling/adaptation, horizon/row/task separation, immutable models and constrained long-only research allocation. | 24 synthetic tests, including finite-difference meta-gradient verification. MAML/pooled/scratch comparisons retain all outcomes; equal adaptation budgets do not equalize pretraining compute. Empirical regime/forgetting evidence and serving remain open. |
| Continual learning | Actual online Gaussian gradients, delayed-only scaling, bounded Algorithm R reservoir, immutable prequential forecasts, past-loss drift response, frozen batch forgetting baseline and exact optimizer/RNG/pending-state rollback. | 54 synthetic tests. Cold forecasts fabricate no confidence. Empirical stream/retention evidence, operational integration and deployment governance remain open. |
| Federated learning | Actual local minibatch logistic SGD, train-only client normalization, parameter transport to common raw coordinates, sample-weighted FedAvg, atomic update validation, frozen round history and JSON replay; input-driven training/verification CLI, saved-evidence HTTP reader and dashboard comparison. | 44 model tests plus end-to-end training/replay/HTTP checks; the latest combined CLI/API/core/docs suite passed 64 tests. Local/centralized/federated controls retain differing compute budgets. Clients share one Python process; hashes, counts and parameters can leak information. No partner authorization, process isolation, secure aggregation, differential privacy or empirical privacy claim. |
| Trade anomaly | Actual isolation trees, frozen numerical traversal, separate delayed annotation calibration, later proper scores/alert diagnostics, safe JSON persistence and CLI score replay. | 30 synthetic model tests. Annotation evidence classes propagate through model/forecast/evaluation/restore. Insider misconduct and informed intent stay unknown; independent annotations, tape rights, empirical effectiveness and operational surveillance remain open. |
| Classical QUBO selection | Binary equal-notional mean/variance selection, safe squared-cardinality penalty, actual Metropolis/cooling updates, retained failed/infeasible outcomes, matched-constraint greedy/exhaustive controls and write-once replayed receipts; CLI. | 37 synthetic tests. Optimization objectives are not predictive proper scores; compute budgets differ. No MIQP comparison, validated economic forecast, hardware or quantum advantage. |

Parent-order RL/auction execution has passed 36 focused tests and a saved
synthetic DQN/TWAP/AC comparison, with full state/cost replay and model persistence.
Its receipt is `data/metadata/blueprint_execution/seed7_fixed_v1/receipt.json`,
SHA-256 `fcd2d9392ab9a3d3c2247687c45397f1e34cd214275155b50685fd79069d978e`.
The planted mechanism demonstrates synthetic learning only. Realistic auction/LOB
calibration, AS comparison, richer priority/impact and training/service APIs remain open.
CLI verification replays the saved policy actions and independently checks ledgers;
it does not retrain or validate real execution quality. Existing classical or related models remain named
baselines, not substitutes for the specified learned algorithms.

## Strategy CLI and HTTP

Use the installed environment; the lockfile remains authoritative. A
`StrategySpec` has `id`, `title`, `description` and scalar `parameters`.

```bash
fx1 strategy generate --spec spec.json --backend hosted_k3 --out generated.json
fx1 strategy replay --spec spec.json --generated generated.json --data tape.json --cost-bps 2 --out receipt.json
fx1 strategy verify receipt.json
fx1 strategy serve --port 8011
```

Generation requires the actual hosted credentials/backend. `MOONSHOT_API_KEY`
was absent from this workstream's local process; no hosted generation or trained
local inference was run. The local backend still does not provide a trained
strategy checkpoint. Synthetic injected backends are correctness fixtures.

The supported code is one function returning `probability_up` and `target_weight`
expressions. No host Python execution occurs. Imports, attributes, indexing,
assignments, loops and arbitrary functions are rejected. Histories contain only
prices available at each decision. Delayed current closes make a decision
unscoreable rather than changing the target horizon. Cost is target-weight
turnover times explicit bps, not a realistic fill/impact simulator.

`tape.json` contains `bars` (`event_time`, `available_time`, `close`), timezone-aware
`decision_times`, `data_source` and an explicit Boolean `synthetic`. Output paths
must be new. Receipts bind the entire specification, code, observation clocks,
replay, costs and evaluation, plus implementation/runtime identities. Verification
reproduces the replay; recorded external judge competence and generation identity
remain independently unverified.

The loopback HTTP pilot exposes `GET /v1/strategy/capabilities` and
`POST /v1/strategy/replay`. The latter takes the same spec/generated/bars/clocks/
source/synthetic/cost fields directly and returns a hash-bound receipt; it accepts
no filesystem paths. Generation and semantic grading are explicitly unavailable.
`FX1_API_KEY`, when configured, protects every route except `/health`, including
API docs. Otherwise remote clients are refused. Bodies are limited to 64 KiB,
bars to 512 and decisions to 128; domain work budgets also apply.

The product and harness APIs remain separate to preserve the repository's
one-way import boundary. A unified gateway is still an integration task.

## Learned offline router

```bash
dipcatcher blueprint route --logs orders.json --proposals proposals.json --train-asof 2020-01-01T12:00:00+00:00 --decision-time 2020-01-01T12:01:00+00:00 --quantity 100 --side buy --opportunity-cost-bps 10 --out route.json
dipcatcher api --port 8000
```

Training logs contain typed `OrderObservation` records: order/venue identities,
decision/features/outcome clocks, `FillFeatures`, observed fill and conditional
toxicity, source and synthetic labels. Proposals contain a venue, quote/features
clocks, price/capacity/fee/crossing cost and features. Fill features are
`price_distance_bps`, `spread_bps`, `queue_ahead`, `displayed_depth`,
`volatility_bps` and `seconds_to_deadline`.

Optional `--evaluation-logs` with `--evaluation-asof` scores strictly later
finalized holdout observations. The planner respects capacities and a supplied
limit; it can return unallocated quantity. Model hashes are checked against
actual parameters before prediction. The objective freezes per-unit fill/fee/
toxicity/opportunity cost; it cannot model quantity-dependent liquidity or impact.

The existing authenticated harness API exposes
`GET /v1/blueprint/capabilities` and `POST /v1/blueprint/route`. The POST takes
observations/proposals directly, training/decision clocks, quantity, side,
opportunity cost, optional limit and seed. Fits are ephemeral, with at most 512
logs and 32 proposals inside the existing 64 KiB body cap. Results bind inputs and
model/plan identities. This is neither a persisted trained checkpoint nor an
order submission API. Existing `QUANT_API_KEY`/loopback restrictions apply.

## Saved-artifact inspection and anomaly research

```bash
dipcatcher blueprint graph-evidence --run data/metadata/blueprint_graph/seed7_fixed_v1
dipcatcher blueprint execution-verify --receipt data/metadata/blueprint_execution/seed7_fixed_v1/receipt.json
dipcatcher blueprint anomaly --inputs timed-patterns.json --out new-anomaly-run
dipcatcher blueprint anomaly-verify --run new-anomaly-run
```

The graph command exposes the same bounded evidence reader used by the dashboard.
The execution verifier reproduces frozen synthetic policy actions and independent
ledgers. Both commands successfully read the preserved runs; neither repeats training.

Anomaly input is a JSON object with `training`, `fit_asof`, `calibration`,
`calibration_annotations`, `calibration_asof`, `evaluation`,
`evaluation_annotations` and `evaluation_asof`. Pattern rows have an ID, event/
available/decision clocks, source, explicit synthetic Boolean and six causal
features: log quantity, relative spread, signed price distance in bps, trailing
volume z-score, trade rate and cancel ratio. Annotations have pattern ID,
suspicious Boolean, availability, source, method and an explicit synthetic Boolean.
The caller must establish feature construction and annotation independence.

Training, calibration and evaluation IDs/times must be disjoint and chronological.
The output directory must be new; input data, safe frozen tree model and receipt
are retained. Verification checks their hashes and recomputes held-out probabilities
and scores. Training and source-availability verification remain separate. Synthetic
annotations make results synthetic even when the trade features are non-synthetic.
Probabilities concern supplied annotation semantics; they do not establish misconduct.

The fixed synthetic run `data/metadata/blueprint_anomaly/seed7_fixed_v1` used
128 training, 96 calibration and 80 later evaluation patterns, 16 trees,
64 samples/tree and seed 7. Its receipt SHA-256 is
`c706917b7387025b52ff52621d2c6dcb70f538c0309784698f39fb1936672c5b`.
Held-out Brier was 0.079999 against prevalence baseline 0.25; negative log loss
was 0.275031 against 0.693147. Seven false positives and 40 true positives were
retained at the predeclared 0.5 threshold. This planted injection fixture is
synthetic correctness, not empirical detection performance. The CLI verifier
reproduced its forecasts/scores and training/calibration input bindings.
The receipt retains a measured implementation source snapshot. Subsequent
JSON-restoration hardening changed the current source hash; verification reports
that difference while checking the preserved source snapshot and numerical replay.
It does not repeat training or independently authenticate the supplied labels.

```bash
dipcatcher blueprint qubo --inputs selection.json --out new-qubo-receipt.json
dipcatcher blueprint qubo-verify --receipt new-qubo-receipt.json
```

QUBO JSON has `forecast` and `covariance` evidence, `decision_time`, `cardinality`,
`risk_aversion` and `budget`. Both evidence records have aligned `assets`,
`horizon_seconds`, `asof`, `available_time`, `source_id`, `source_sha256` and an
explicit synthetic Boolean. Forecasts supply `expected_returns`; covariance
supplies finite symmetric positive-semidefinite `values`. Publication must precede
the selection decision. Weights are `budget/cardinality` on selected assets and
the remaining budget is cash; this is a static research allocation.

The default bit-flip method anneals the unconstrained penalized QUBO. The optional
cardinality-swap method is separately labeled constrained annealing. Finite
annealing can fail despite a theoretically sufficient penalty; failed outcomes
stay in the receipt. Exhaustive float64 comparisons run only within both asset
and combination limits and prove only exhaustion of that finite set.

The fixed eight-asset synthetic problem (generator seed 13, cardinality 3,
budget 0.8; solver seeds 11/23/47/83, 100 sweeps) used 3,200 proposals. Annealing
and greedy both reached the same objective as exhaustive enumeration of 56
feasible selections. This establishes no algorithm advantage or forecasting
quality. Receipt `data/metadata/blueprint_qubo/seed13_fixed_v1.receipt.json` has
SHA-256 `c671c4a4e13b24f247610685018150ebc0f1fc06a141373e38b9de331fe3fdf0`;
solver replay verification passed before type-validation cleanup. Its bound
source is preserved as `seed13_fixed_v1.source.py`. The unchanged protocol was
rerun with the current implementation as `seed13_fixed_v2.receipt.json`, SHA-256
`f36cdccdd70cf5a4279215ef3cc3622d2496fb9c2434fee6b504b2e33e526be2`;
current CLI solver replay passed with the same objective/gap/outcomes. Neither
immutable receipt was overwritten and the experiment was not tuned.

## Federated comparison and replay

```bash
dipcatcher blueprint federated --inputs timed-clients.json --out new-federated-run
dipcatcher blueprint federated-verify --run new-federated-run
```

Inputs declare the binary `task`, `clients` with `client_id`/`examples`, a later
`holdout`, `initial_asof`, `training_asof`, `aggregation_asof`, `evaluation_asof`,
`client_config`, `centralized_config` and `rounds`. Examples declare feature axes,
integer labels, all five event/feature/decision/target/publication clocks,
source evidence hashes, `train`/`holdout` split and explicit `synthetic` status.
The bounded CLI runs all clients in one process and pools rows only for the
declared centralized control. It saves input bytes, the measured model source,
the complete aggregation snapshot and immutable comparison receipt.

The fixed synthetic run `data/metadata/blueprint_federated/seed7_fixed_v1` used
64/96 client training examples, six rounds and 512 later examples generated with
a separate fixed seed. Receipt file SHA-256 is
`051844820687005bfc1fe54006803d26764f4259341aee6b4e462bbcfb860ce5`.

| Predictor | Brier | Negative log loss | Actual optimization scope |
|---|---:|---:|---|
| Global FedAvg | 0.175822 | 0.523701 | 300 cumulative client steps |
| Local alpha | 0.174462 | 0.520973 | 20 last-round steps; inherits previous global rounds |
| Local beta | 0.176997 | 0.526179 | 30 last-round steps; inherits previous global rounds |
| Centralized | 0.183354 | 0.548569 | 60 steps from the initial model |

The global predictor lost to local alpha. Unequal budgets, known planted labels
and a single synthetic trial establish no general federation advantage or
privacy property. All four score calculations were independently reproduced
with scalar sigmoid/Brier/log-loss formulas. Saved verification checks artifact
hashes, round/input bindings, weighted aggregation and every saved predictor's
held-out arithmetic; it does not repeat client gradients, authenticate sources,
prove economic independence or establish privacy.

`GET /v1/blueprint/federated/{run_id}` exposes this same bounded evidence reader.
The dashboard shows the scores, budgets and one-process/privacy limits. Desktop
and 390-pixel mobile browser checks passed; a missing run clears old scores and
reports refusal. The deliberate missing-run check produced its expected HTTP
404 console entry; normal loading had no application errors.

## Preserved historical experiments

Both experiments use local `data/file_us_wide` vendor-adjusted Yahoo snapshots.
Training ends in 2021, validation covers 2022–2023, and the previously inspected
test tail covers January 2024–September 2025. Membership/target clocks, complete
cases, embargo and training-only asset selection are explicit in the receipts.
Adjustment vintages, surviving-security pool, corporate-action completeness and
reconstructed availability prevent fresh-OOS or strong institutional claims.
One predeclared seed/budget was run per experiment; no tuning followed.

| Test proper score, lower is better | Learned model | Matched baselines | Finding |
|---|---:|---|---|
| GCN CRPS | 0.01037127 | Node-only 0.01024091; pooled 0.01057872; ridge 0.01059703 | Graph lost to same-capacity node-only on validation/test CRPS and Gaussian NLL. Graph value is unsupported. |
| Five-session joint energy score | DDPM 0.16288197 | Gaussian 0.09664299; marginal IID 0.09572460; block bootstrap 0.09614749 | DDPM lost to every baseline on both periods, despite lower training denoising loss. |

Graph: 24 assets, 1,451 training/495 validation/431 test dates; seed 7, hidden
size 16, 100 epochs. Receipt
`data/metadata/blueprint_graph/seed7_fixed_v1/receipt.json`, SHA-256
`516cd4da7d603c1225ddad05cc7437bfab3fda27353d936ff2f838b573f5d89b`.
Test GCN-minus-node CRPS moving-block interval is
`[0.00008057, 0.00018478]`; it is a diagnostic without multiplicity adjustment.
The measured receipt's legacy `log_score` field means negative log density;
the dashboard displays it as Gaussian NLL, lower is better.

Diffusion: first eight training-selected assets, nonoverlapping five-session
paths, 290 training/99 validation/86 test windows; 16 diffusion steps, hidden
size 32, 100 epochs, 128 scenarios per model, seed 7. Receipt
`data/metadata/blueprint_diffusion/seed7_fixed_v1/receipt.json`, SHA-256
`5395c0f6d300ad8b5b418bc82c4767e5e7c31b5810e32594f6ba14f187c03a60`.
Every generated ensemble stays **SYNTHETIC**, including Gaussian/bootstrap draws
from empirical training data. This unconditional path-distribution experiment
does not prove conditional forecasts, TSTR or tail-event coverage.

Exact measured source snapshots, parameters and predictions/ensembles are saved
with the runs. Score arithmetic was independently recomputed. Diffusion sampling
replayed bit for bit from saved NumPy parameters. These custom artifacts are not
automatically accepted `verify-research` promotion receipts.

## Interactive graph evidence pilot

With the harness API running, open `/v1/blueprint/dashboard`. Choose an asset or
edge threshold to inspect the actual training relationships; switch historical
periods to inspect the matched proper scores. It uses no external scripts.
The reader rejects corrupted receipts/artifacts, recomputes parameter identities
and score arithmetic, and exposes what it did not verify. It does not retrain
models, reconstruct features or independently establish historical publication
times. Time-varying topology, text/fundamental embeddings and forecasts for new
live data remain outside this pilot.

Desktop/mobile rendering, asset selection, period selection, data loading and
network/console health were checked in a real browser. The fixed adverse finding
and source limitations are visible. A missing/corrupt run fails closed.

## Q01–Q03 integrated CLI/API surface and receipt identities

This section coherently presents the three completed workstreams' command-line
interfaces, HTTP endpoints and immutable receipt identities so they can be
exercised together without hunting across scattered sections above. The broader
verification record follows in the next section.

### Receipt identity convention

All hashes below are the **internal canonical `receipt_sha256` field** of each
receipt JSON, not the SHA-256 of the file bytes on disk. This is the stable
identity: it is computed over a canonical serialization of the receipt payload
and survives reformatting or filesystem copies. File-bytes hashes change on any
re-serialization; internal hashes do not.

### Q01 / F01 — bounded strategy lane

| Surface | Command / endpoint | Notes |
|---|---|---|
| Generate (hosted) | `fx1 strategy generate --spec spec.json --backend hosted_k3 --out generated.json` | Requires `MOONSHOT_API_KEY`; no trained checkpoint was run in this workstream |
| Replay | `fx1 strategy replay --spec spec.json --generated generated.json --data tape.json --cost-bps 2 --out receipt.json` | Bounded AST interpreter only; no arbitrary Python execution |
| Verify | `fx1 strategy verify receipt.json` | Reproduces replay from saved receipt |
| Serve | `fx1 strategy serve --port 8011` | Loopback HTTP pilot; `FX1_API_KEY` protects all routes except `/health` |
| HTTP capabilities | `GET /v1/strategy/capabilities` | Returns supported code shape and backend list |
| HTTP replay | `POST /v1/strategy/replay` | Accepts spec/generated/bars/clocks/source/synthetic/cost directly; returns hash-bound receipt; bodies capped at 64 KiB, bars at 512, decisions at 128 |

No persisted strategy receipt exists on disk for this workstream: generation
requires hosted credentials that were absent from the local process. Synthetic
injected backends are correctness fixtures, not market evidence. The loopback
HTTP pilot exposes the same replay logic as the CLI but accepts no filesystem
paths. The product and harness APIs remain separate to preserve the repository's
one-way import boundary.

### Q02 / F26 — QuantCode benchmark importer

This workstream has **no CLI entry point**. It is a library module consumed by
tests and the fx-1 eval battery.

| Item | Value |
|---|---|
| Module | `src/fx1/eval/quantcode_bench.py` |
| Upstream repo | `https://github.com/LimexAILab/QuantCode-Bench` |
| Pinned revision | `f8bda951addb409a81aa316c00401dbde60774ae` |
| Task count | 400 |
| License | MIT |
| Task data SHA-256 | `b197e0271779f332c6808ea40167615e3b90061563544b8bdf3c48237a9f17d3` |
| Requirements SHA-256 | `7bc4039cfe971ec04de3618c652eca268c95ce07030c5f597c594209344f38b9` |
| Tests | 31 focused adapter tests + actual complete import verified |

The `BenchmarkPin` dataclass enforces HTTPS URL, full Git revision, exact task
count and both content hashes at construction time. Any deviation fails closed.
Official grading equivalence, OS sandboxing and model scoring remain unverified
(Q16/Q17).

### Q03 / F07 — asset GNN pilot

| Surface | Command / endpoint | Notes |
|---|---|---|
| Evidence reader | `dipcatcher blueprint graph-evidence --run <dir>` | Recomputes parameter identities and score arithmetic from saved arrays; rejects corrupted receipts/artifacts |
| Dashboard | `GET /v1/blueprint/dashboard` (with harness API running) | Asset/edge/period selection; proper-score inspection; fail-closed on missing/corrupt runs |
| Receipt path | `data/metadata/blueprint_graph/seed7_fixed_v1/receipt.json` | Frozen training-only absolute-return-correlation graph |
| Schema version | `blueprint_graph_empirical_exploratory_v1` (saved), `blueprint_graph_ui_v1` (UI export) |
| Receipt SHA-256 | `516cd4da7d603c1225ddad05cc7437bfab3fda27353d936ff2f838b573f5d89b` | Verified against PLAN doc citation |
| Models compared | GCN, node-only, pooled Gaussian, ridge Gaussian | Same-capacity ablation preserved |
| Scores | CRPS and gaussian_nll, both lower-is-better; proper scores only, no Sharpe/P&L |
| Promotion | `false` | Adverse exploratory finding retained honestly |
| Tests | 34 model + 10 experimental protocol | Empirical receipt preserves the adverse graph/no-graph result |

The evidence reader verifies artifact integrity, parameter identities and
Gaussian score arithmetic bit-for-bit. It does not retrain models, reconstruct
features or independently establish historical publication times. Dynamic
topology, supply-chain edges and prospective validation remain open (Q15/Q16).

### Cross-cutting constraints

All three pilots share these hard boundaries, which apply equally to every
blueprint feature:

- **Proper scores only.** Research outputs use CRPS, log score, Brier, ECE,
  pinball, PIT, Kupiec and HMM likelihood. Sharpe, Sortino, Calmar, PnL and
  NAV are forbidden as research-headline metric keys
  (`FORBIDDEN_RESEARCH_METRIC_KEYS` in `src/quant_fund/research/catalog/registry.py`).
- **Synthetic results labeled.** Every synthetic test result carries explicit
  labels; none is presented as market evidence.
- **No broker connectivity.** Live FIX/REST/CCXT and multi-broker deployment
  conflict with current repository instructions (D01). All order-path modules
  are reachable only from cli/paper/execution/simtest/pretrade/formal.
- **Receipts immutable.** Every claim should be reproducible from a receipt
  hash. Receipts live under `data/metadata/blueprint_*` and `receipts/`.
- **Honesty contract enforced.** `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS`
  mirrors the catalog's forbidden keys; drift is blocked by
  `tests/fx1/test_honesty_inheritance.py`.

## Verification and open acceptance

The fx-1 offline suite passed with **593 passed, 1 skipped** after strategy/API/
benchmark integration. Focused learned-model and experiment tests listed above
are synthetic correctness/protocol checks. The combined model/protocol/evidence suite passed 239 tests, integrated API/CLI
checks passed 96 tests, and strict MkDocs plus repository Ruff checks passed.
The later meta/continual/anomaly/API/docs focused suite passed 154 tests, and
the anomaly/QUBO/CLI suite passed 70 tests. Full harness mypy passed.
The broader offline lab run finished with **15,290 passed, 25 failed,
190 skipped and 1 xfailed** in 49 minutes 53 seconds. It used two workers,
excluded network/slow tests and disabled coverage; it was not the literal
`make test` gate. Failures include new-module integration checks and HEAD-existing
risk-test, receipt-schema/export/index, documentation and quality-manifest drift.
Repairs and their focused validation remain in progress; this run is not a pass.
A read-only Windows inventory found 96 Kimi-K3 base shards with ~1.56 TB logical
size; it did not establish checksums, rights, inference or fine-tuned fx1. The
source/config inspection is in `data/metadata/blueprint_remote_inventory_20261001.json`.
The full lint gate still rejects five independently added allowlist paths that
do not exist; their unrelated edits are preserved. Broader gates are tracked separately;
none of these counts substitutes for `make test`, all strict-allowlist checks,
full runtime integration or a trained-model benchmark.

The full goal remains open across F01–F28/A1–A6. Outstanding work includes a
trained strategy checkpoint and qualified benchmark executions, remaining learned
algorithms, capability persistence/serving/UI, actual rights/availability and
label audits, prospective empirical comparisons, throughput/recovery/feedback
evidence and external commercialization decisions. D01 live connectivity conflicts
with current instructions; D03 licensing and D09 patent clearance require owner
decisions. Synthetic pilots and document completion do not close these items.
