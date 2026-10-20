# dipcatcher production product contract

**Date:** 2026-10-08, surfaces remeasured 2026-10-12. **Status:** release
requirements plus the implemented engineering: the chat launcher, in-chat
endpoint setup and identity probes, honesty checks on every model-authored
surface, bounded planner refinement with a research-family eligibility gate,
hard run budgets, cancellable subprocess execution, durable session journals
with interrupted-job reconciliation, and the Ctrl-C/history console layer.
Production acceptance (P1–P7) still requires the evidence listed below — code
and tests do not substitute for it; see the dated
[handoff](CLAUDE_DIPCATCHER_HANDOFF.md).

## Product definition

**dipcatcher is Artificial Hedge's research orchestration harness.** The
chat ids are `fx1` and `fx1-lite`. No fine-tuned weights are published. The
conversation in the terminal uses whatever endpoint the operator configured. The primary experience is chat. The
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

| Surface | Observed source (2026-10-12) | Gap to this contract |
|---|---|---|
| Product entry | `dipcatcher_cli.app` opens chat for bare terminal invocation, supports `chat --model` / `chat --resume`, `--version`, and forwards lab commands; `quant` retains its existing entry. | Cross-platform UX acceptance on real Windows/macOS terminals and release packaging approval. |
| Existing conversation | `concierge.run_console` with a `LineReader` (POSIX select / Windows msvcrt / blocking), Ctrl-C cancel-then-exit policy, secret-filtered persistent history, `/keys set` in chat, startup endpoint status with failure guidance. | Multiline editing/paste and resize handling; the frozen journey suite and moderated trial. |
| Models | `profiles.MODELS` holds `fx1` and `fx1-lite`; `/use` keeps the conversation and journals the switch; `/keys test` records requested vs served identity, latency and failure class; endpoint failures map to classes with next actions and bounded retries. | Real authenticated smoke runs against the service (P2) — mock tests do not close the gate. |
| Research planning | `superpower.plan_goal` selects from `HARNESS_REGISTRY` plus memory/web/synthesis; the read-only `research-families` schema v2 names 23 required and six audit-justified optional families for the planner, with counts for the remaining exploratory and retired families. A bounded, goal-bound hash lookup identifies bare names from the omitted sets. Named-family goals cannot automatically launch an unconstrained research scorecard; missing or unsupported names and failed lookups close the gate, and the model cannot reinstate a blocked capability. Plain-chat model tools cannot start `research`; an operator can request its full scorecard explicitly. | The explicit `research` command still executes its configured scorecard; per-family execution, typed dataset/config arguments, and data-availability checks are not implemented. |
| Planner limits | `max_harness` on both paths; `RunBudget`/`BudgetLedger` enforce wall, step, model-call, retry and web-source ceilings; the remaining wall budget becomes each subprocess deadline. | Token spend is capped only when configured; provider prices are unknown so no currency cap is claimed. |
| Answer validation | Ordinary chat, tool preambles and synthesis pass the honesty validator; every capability result carries an evidence class; partial runs say partial; web content is framed as untrusted data. | Semantic claim/receipt verification; audit other expert/export surfaces. |
| Jobs | `BackgroundJob` threads with cooperative cancellation; owned subprocesses receive SIGTERM → SIGKILL on cancel or deadline; the session journal records start/step/finish and marks work left open by a dead process as interrupted on the next start. | Automatic safe resume of idempotent steps; remote (fleet) cancellation acknowledgement. |
| Documentation | [FXI](FXI.md) and [Concierge](DIP_CONCIERGE.md) describe the implemented behaviour and remaining gaps. | Reconcile README/root instructions when the release candidate is fixed. |

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
