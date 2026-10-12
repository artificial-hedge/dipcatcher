# dipcatcher production product contract

**Date:** 2026-10-12. **Status:** the chat launcher, dataset attach, the
read-only job list, qualified family selection, partial selective runs,
session recovery, cancellation of owned subprocesses, step/time budgets,
and evidence-class grounding are implemented and covered by offline tests.
Scripted journey runs, a relevance run, a two-minute soak, a latency run
on this machine, and an offline wheel smoke exist. Production acceptance
gates P1–P7 below stay open. None of those runs is the evidence the gate
text requires. Live model smoke, a moderated user trial, a 24-hour soak,
and signed release artifacts are not implied. The requested 25% note in
the handoff is historical; it is not a readiness percentage. Partial
measurements are in
[the release evidence table](DIPCATCHER_RELEASE_EVIDENCE.md).

## Product definition

**dipcatcher is Artificial Hedge's research orchestration harness, built to
make artificialhedge.co's `fx1` and `fx1-lite` models useful through a
conversation in the terminal.** The primary experience is chat. The
`/superpower` command selects suitable research capabilities, executes them
through the harness, and explains the verified evidence in the same chat.

State of the art is the ambition. A comparative claim requires a dated,
reproducible evaluation against named baselines on the same tasks, data,
budgets, and scoring protocol. Catalog size and passing fixtures do not
establish that claim.

Use these names consistently:

| Name | Product responsibility |
|---|---|
| `dipcatcher` | User-facing CLI and orchestration product; `quant_fund` remains its independent research, data, evaluation, and verification engine. |
| `fx1`, `fx1-lite` | Artificial Hedge model identifiers; displayed and recorded exactly as selected. A configured identifier alone does not prove a released or available model. |
| `fx-1` / `src/fx1` | Existing model-line name and package containing model-facing integration, training, evaluation, and interactive code. Preserve existing API/version contracts. |
| `fxi`, `quant`, `fx1` executables | Existing compatibility and expert entry points. The product should not require users to learn all three. |

This release targets dependable research orchestration. Research software
readiness, each model's release readiness, and live-trading eligibility are
separate decisions. Preserve the constraints in
[Institutional readiness](INSTITUTIONAL_READINESS.md): proper-score research
headlines, explicit SYNTHETIC labeling, immutable receipts, and no live claims.

## What exists and what must change

Work on 2026-10-08 began at local `50f25b2ce` in a concurrently modified
working tree. The table includes the implementation in this handoff; it is
not hosted-model or full release certification.

| Surface | What exists now | What remains |
|---|---|---|
| Product entry | `dipcatcher_cli.app` opens chat on a terminal, supports `chat --model`, `exec` (including repeatable `--data`), `models check`, and `sessions`, and forwards lab commands. `quant` stays the lab entry. Redirected bare `dipcatcher` prints help and does not read stdin. An offline wheel smoke imported the three packages and ran those help paths. | Cross-platform UX acceptance, signed artifacts, rollback, and release packaging approval. P7 stays open. |
| Existing conversation | `/help` matches the dispatcher, including `/data`, `/runs`, `/models check`, `/resume`, and `/evidence`. Sessions reload with `/resume` and `chat --resume` and restore attachments. `/new` clears them. An interrupted job is recorded and is not restarted. Two automated 100-journey suites exist. The pty/in-proc suite's latest recorded run is 100/100, including its critical journeys, on a stub model. An earlier run of a different suite digest was 99/100 (`J082`, corrupt session file reported and not quarantined; that failure is fixed in the later run). | The five-user trial and retained terminal recordings. P1 stays open. The YAML suite was not re-executed for the evidence table. |
| Models | `fx1.interactive.profiles.MODELS` contains `fx1` and `fx1-lite`. `/models` and `dipcatcher models check` report presence. `--live` is one identity call per configured model and says the record is not a release claim. Offline check does not open a socket. | P2 stays open until a real authenticated smoke of both models records 401, unknown model, 429, timeout, and server failure against the real service. Stub and mock runs are a different check. |
| Research planning | The planner selects harness commands. The read-only `families` surface constrains `research --only`. Unsafe goals are refused before selection (exclusion code `unsafe`). `/data` and `exec --data` attach a CSV or Parquet under `<cwd>/data`; the sha256 is rechecked before a run. The relevance suite (55 cases) recorded supported-case pass rate 1.0, unsafe refusal rate 1.0, and invariant failures none. | The expert half of P3 (at least 90% on the preregistered rubric) is not rated. P3 stays open. |
| Plain-text routing | `intent.looks_like_orchestration` sends some sentences down the same planner. On the 64-phrase labeled set, precision is 1.0 and recall is 0.625. | Misses stay on the chat path until the operator types `/superpower`. P3 stays open. |
| Planner limits | Deterministic and refined plans enforce `max_harness`. Family selection enforces `max_families`. `RunBudget` caps steps, retries, and seconds and reports `spend_cap=None`. | Dollar spend cannot be capped without a price catalog. |
| Answer validation | Honesty validator plus `ground_summary`. Glued forms such as `sharpe3.2` are headlines. Verification phrasing is kept only when the run record lists that receipt path. Red-team tests exercise tamper, injection, and secret-leakage failures. | Lexical and path checks are not a semantic proof. Spelled-out numbers, a bare metric name, and a token glued onto a prefix stay outside the gate. `verify-research` is still required before a result is called verified. P4 stays open. |
| Jobs | In-process thread, cancel of an owned subprocess group, and a terminal stop that cancels the in-flight step and does not start a replacement. Sessions record `interrupted` and do not auto-retry. `/sessions` quarantines a corrupt file. `/runs` is read-only. | Remote Windows fleet cancellation still uses the existing manifest launcher. |
| Responsiveness | One latency record on this machine: cold prompt p95 0.244281 s over 30 starts, local ack p95 0.03819 s over 200 events, owned-sleeper stop 0.100916 s. `reference_hardware` is false and `p5_gate_closed` is false. | Reference hardware, crash/restart as the gate defines it, and model latency measured separately. P5 stays open. |
| Bounded operation | Step, time, retry, and token ceilings exist. Unknown price does not imply a spend cap. A 120-second stub soak recorded `ok` true, no orphan children, and `reference_24h` false. `make dipcatcher-evidence` asks for 60 seconds, which is a different run. | The 24-hour mixed-workload soak. P6 stays open. |
| Documentation | [FXI](FXI.md), [Concierge](DIP_CONCIERGE.md), the runbook's chat section, and [the release evidence table](DIPCATCHER_RELEASE_EVIDENCE.md) describe the tree and the measurements above. | Naming stays as in this contract. The docs epoch pin and the crown-jewel pin are red. Re-stamp needs the signing key. |

The repository-wide [production audit](ULTRA_PROD_READINESS.md) owns CI,
history reconciliation, artifact integrity, catalog cleanup, and release
infrastructure. Its dated findings must be remeasured, not copied as current
facts. This contract adds the user experience and orchestration acceptance
criteria; it does not replace that work.

## The chat experience

For this product, “like Codex CLI” means a terminal conversation with a
persistent composer, readable transcript, concise progress, in-place approvals,
visible model identity, and discoverable commands. It is a design direction,
not a claim of feature parity or a requirement to duplicate another product.

1. **Start:** after installation, `dipcatcher` opens chat. First-run endpoint
   setup occurs inside the experience with a masked secret input. Offline
   help and local inspection remain usable. Never assume a working endpoint
   from a company domain or present a fixture/backend fallback as `fx1`.
2. **Ask:** ordinary text handles research questions and follow-ups without
   requiring shell syntax, JSON, a family name, or a manual YAML edit. Ask a
   focused question only when missing data or scope materially changes the run.
3. **Delegate:** `/superpower <goal>` is the explicit shortcut for research
   orchestration. An equivalent natural-language request uses the same planner,
   policy checks, and execution path.
4. **Observe:** show the selected model, selected methods and why, progress,
   budget, evidence class, and cancellation state. Keep raw logs and full
   configuration available as optional details. A spinner is not progress.
5. **Control:** provide discoverable model selection, help, status, cancel,
   session resume, and evidence inspection. `/use`, `/model`, `/help`,
   `/status`, `/cancel`, `/resume`, `/evidence`, `/runs`, `/data`, and
   `/models check` exist. Keep multiline input, paste, history, keyboard
   navigation, narrow terminals, `NO_COLOR`, and plain output usable. A
   background job uses a cbreak line reader, so that line has no history
   and no arrow keys.
6. **Finish:** explain what was learned, what failed or remains uncertain,
   and link the verified receipt/artifacts. A partial run must say partial.
   Unconfigured models may return a clearly labeled deterministic status
   summary, never a fabricated model answer.

**Make “99% chat” measurable.** Before implementation, freeze a suite of 100
representative journeys across onboarding, model selection, data discovery,
research planning, execution, follow-ups, verification, cancellation, recovery,
and export. At least 99 must complete inside the conversation without an
external shell, hand-edited configuration, or required expert subcommand.
User-requested slash shortcuts, secret inputs, and scoped approvals count as
in-chat interaction. All critical honesty, credential, cancellation, and
recovery journeys must pass even if the aggregate reaches 99. Include a
moderated trial with at least five intended users; publish failures instead
of padding the suite with trivial prompts. This measures journeys, not an
arbitrary percentage of screen pixels or transcript words.

Illustrative target interaction (not a transcript or runnable evidence):

```text
$ dipcatcher
dipcatcher · fx1 · research
you › /superpower Check whether my volatility forecasts are calibrated.
dip › Which dataset and forecast column should I use?
you › Use the dataset already attached to this session and its variance forecast.
dip › I will check availability timing, select eligible volatility diagnostics,
      and score the forecast with QLIKE. Here are the data checks and run budget.
      [progress and any scoped approval appear here]
dip › [verified result, evidence label, limitations, receipt path and digest]
you › Explain the weakest period, then switch to fx1-lite for the summary.
```

## `/superpower` execution contract

The model proposes a plan; the harness determines what is eligible to execute
and what the resulting evidence supports. Do not expose the entire family
catalog as an undifferentiated tool menu.

| Stage | Required behavior |
|---|---|
| Understand | Resolve goal, data, horizon, output, and budget from the conversation. Missing market data is unavailable, never silently replaced by SYNTHETIC data. |
| Select | Choose only known, qualified, non-retired families with an implemented runner, satisfied data requirements, and permitted evidence class. Catalog membership or a descriptive provenance label alone is insufficient. Missing qualification metadata fails closed for execution. |
| Explain | Show selected families, rationale, prerequisites, proper scores, cost/time limits, and material exclusions. “No suitable family” is a valid result; explain the missing capability or data. |
| Validate | Validate family IDs and typed arguments, dependency order, output paths, authorization, and budgets after every model refinement. Deduplicate steps. Reject arbitrary shell text, unknown IDs, invented flags, and attempts to change the verifier. |
| Execute | Run through registered harness interfaces using argument arrays. Carry dataset/config/code/model identity and run IDs through every step. Stop dependent steps after a failed prerequisite. Bound retries and prevent duplicate effects. |
| Verify | Treat exit code zero as execution success only. Verify the actual research receipt and input identities before calling a finding verified. Historical receipts for retired families remain verifiable; retirement blocks new selection. |
| Report | Separate verified research, SYNTHETIC correctness results, heuristic web findings, and unverified model interpretation. Preserve provenance and uncertainty through follow-ups and exports. |

Each family needs a machine-readable contract: stable ID/version, purpose,
supported tasks, required input schema and availability timing, supported data
labels, runner/argument schema, dependencies, resource estimate and hard limits,
score definitions, qualification evidence, receipt schema/verifier, and
retirement status. Publish this through a registered read-only harness
interface; do not import research internals into the model/UI package.
Build on [Benchmark family lifecycle](BENCHMARK_FAMILY_LIFECYCLE.md) and
[Data contracts](DATA_CONTRACTS.md). Selection must constrain the benches
actually executed, not merely filter a report after running the full catalog.

Each plan/run needs a versioned record: goal, session/run/step IDs, selected
model and endpoint identity (no secret), family versions and selection reasons,
input/config hashes, code revision, budget, authorization decisions, state
transitions, output paths, receipt digests, and verifier verdicts. A mutable job
record is not itself an immutable research receipt.

Apply policy to model tool calls, retrieved memory, web content, and raw tool
output as untrusted data. A document or model suggestion cannot grant itself
permission. Keep normal authorized reads and analysis fluid; reuse approvals
within their recorded scope. Materially broader compute/spend, private-data
sharing, destructive writes, and consequential actions require a visible
scope decision. Noninteractive execution must have explicit scoped policy or
fail closed; never hang waiting for approval.

## Implementation boundaries

Use an outer CLI launcher outside `quant_fund` to route interactive
`dipcatcher` into the existing conversation code. Preserve explicit
`dipcatcher <existing-lab-command>` behavior and the `quant` automation entry
point. Keep `fxi` as a compatibility entry. Preserve `--help`, exit codes,
stdout/stderr semantics, and non-TTY execution; bare non-TTY invocation must
not start a blocking wizard. Prove this with installed-wheel tests.

The launcher must not recurse when the concierge invokes a lab subprocess.
Keep the dependency checks in `tests/unit/test_fx1_dependency_edge.py` and
`configs/arch_boundaries.toml`; no new `quant_fund -> fx1` import, dynamic-import
workaround, or model access to sealed research internals. Extend the public
harness registry for family metadata and execution where needed.

Endpoint configuration must distinguish the service URL from a completion
route and validate it consistently. Keep both model IDs explicit; do not
silently route to a different model/provider. Do not invent an Artificial
Hedge API hostname, release weight, context limit, price, or latency claim.
Validate those against the actual service contract during integration. Keep
credentials out of argv, logs, transcripts, exports, and unrelated child
processes; protect storage on each supported OS, including Windows ACLs.

Persist sessions and job transitions atomically with schema versions. On
restart, mark incomplete work interrupted, reconcile any external job before
retrying, and resume only safe/idempotent steps. Propagate cancellation and
deadlines into model calls and owned subprocesses. For Windows fleet jobs use
the existing manifest launcher/watchdog and durable paths from `AGENTS.md`.
Record a remote cancellation request until termination is confirmed.

Status may stream immediately. Research/model answer text must pass the
honesty contract before display; preserve buffering until a validated
incremental approach exists. Never trade the honesty gate for typing effects.

## Production acceptance gates

All rows are **OPEN**. Numeric limits below are proposed initial release
targets, not a service promise. The measurements in
[the release evidence table](DIPCATCHER_RELEASE_EVIDENCE.md) do not close
any row. Freeze the supported OS/Python/terminal matrix and reference
hardware before treating a number as the gate. Record sample sizes and
failures.

| Gate | Acceptance evidence |
|---|---|
| P1 — Chat front door | Installed `dipcatcher` opens chat; 99/100 frozen journeys complete in chat, every critical journey passes; five-user trial and terminal recordings retained. Legacy commands, help, pipes, and `fxi` compatibility pass. |
| P2 — Both models | Real authenticated smoke runs for `fx1` and `fx1-lite` record requested/served identity and deployment/version provenance. Missing/invalid auth, unknown model, 429, timeout, and server failure recover clearly. Mock tests do not close this gate. |
| P3 — Suitable research | At least 30 frozen goal/data cases spanning supported tasks, paraphrases, ambiguity, and missing/retired/ineligible families. Every planned step is eligible and within budget; all unsafe/unsupported cases refuse correctly. At least 90% of supported cases meet the preregistered expert relevance rubric. |
| P4 — Evidence and trust | Every claimed verified result resolves to a passing immutable receipt and exact inputs. Tampering, stale qualification metadata, prompt injection, forbidden headlines, synthetic mislabeling, secret leakage, and path escape tests all fail closed. Unverified prose never inherits verified status. |
| P5 — Responsive and recoverable | Reference-machine cold prompt p95 <=2 s over 30 starts; local input/status acknowledgement p95 <=100 ms over 200 events. Cancel acknowledgement <=1 s; owned local work stops within 5 s or explicitly reports cancellation pending/failure. Crash/restart and timeout tests preserve sessions and prevent duplicate steps. Model latency is measured separately. |
| P6 — Bounded operation | Hard time, step, retry, token and spend ceilings cover planning through synthesis. Unknown price cannot imply a guaranteed spend cap. A 24-hour mixed-workload soak leaves no orphan local children or unaccounted jobs; injected failures remain recoverable. Redacted diagnostics correlate session/run/step and model/family identity. |
| P7 — Repeatable release | Repository gates and required CI pass at the candidate revision; clean install/upgrade/uninstall, supported-platform smoke tests, dependency/security checks, signed artifacts, checksums, rollback drill, and support/runbooks are evidenced. Existing production-audit blockers are explicitly closed or the release remains blocked. |

The release evidence table is
[DIPCATCHER_RELEASE_EVIDENCE.md](DIPCATCHER_RELEASE_EVIDENCE.md). Columns are
gate ID, owner, candidate SHA, exact command/scenario, platform, artifact
path/hash, timestamp, result, and remaining blocker. Owner is unassigned
in that table. Docs or a merged implementation never substitute for passing
evidence.
Use the existing [Operations runbook](OPERATIONS_RUNBOOK.md) and
[API stability contract](FX1_API_STABILITY.md) for release and recovery details.
Model-release claims additionally require each model's own evaluation/card and
provenance under [FX1 training](FX1_TRAINING.md). Passing these harness gates
does not satisfy the five live-trading conditions.

The remaining delivery work is assigned in the
[Claude handoff](CLAUDE_DIPCATCHER_HANDOFF.md).
