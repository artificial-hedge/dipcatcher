# fx-1 Training Plan — compute, ladder, and gates

This page is the plan. No trained checkpoint is in this repository, and no
training run has been launched from it.

## Capability status (read this first — source of truth)

This doc historically described the *intended* pipeline as if every stage were
wired. It is not. The table below maps each documented capability to its true
shipped state as of **2026-09-28**, with code locations. Where a section below
still describes intended behavior, it carries a greppable status callout so a
future lint can enforce it.

**Marker convention** (greppable; used inline throughout this doc and siblings):

- `**Status: SHIPPED**` — implemented and wired into the live path.
- `**Status: PARTIAL**` — code exists but is incompletely wired, not enforced,
  or reachable only outside the training pipeline (e.g. via a CLI command).
- `**Status: PLANNED**` — design intent only; **not yet wired**. Kept so the
  design is not lost, but it is NOT a claim of current behavior.

Verified gaps are tracked in
`docs/SOTA/09-finetuning-pipeline.md` → *Verified doc-drift backlog*.

| Capability | Status | Code location / what's missing |
|---|---|---|
| Corpus build with `receipt_sha256` provenance | SHIPPED | `data/corpus.py:22-31` (`SFTExample`), `:102` (`build_corpus`) |
| Quality gate: exact + near-dup dedup, shingle screen, frozen split | SHIPPED | `train/pipeline.py:87-112` (`run_quality_gate`) → `data/quality.py:42-94` (`dedup_and_filter`), `:112` (`frozen_split`) |
| Eval-contamination screen (shingle containment vs eval bank) | SHIPPED (basic) | `data/quality.py:64-68`; eval items default from `DEFAULT_BANK` at `train/pipeline.py:95-97` |
| 3-probe contamination audit (n-gram + Min-K% + rephrased gap) in the training path | PLANNED | `eval/contamination.py:132` (`run_contamination_audit`) is wired only to the CLI (`cli.py:229-263`), never called by `Pipeline` |
| Eval-before-train (base honesty gate blocks the run) | SHIPPED | `train/pipeline.py:114-124` (`run_eval_base`) |
| Immutable training receipt + verify (git, hashes, seed, `live_pnl_claim=false`) | SHIPPED | `train/receipts.py:51-131` |
| Statistical comparison (paired bootstrap CI + McNemar), candidate vs base | PARTIAL | computed in `train/pipeline.py:152-171` → `eval/compare.py:60`; **not enforced** (no fail-closed on non-improvement), **domain-only**, and no `promoted` flag exists |
| Promotion gate: refusal/honesty floor ≥ base, general non-regression, fail-closed | PLANNED | not implemented; `run_eval_candidate` returns and advances regardless of the comparison |
| Card stage (`run_card`) → model card + MRM dossier | PLANNED | `Stage.CARD` enum only (`train/pipeline.py:35`); `modelcard.py`/`mrm.py` are invoked via CLI (`cli.py:147,312`), never by `Pipeline` — there is **no `run_card`** |
| Cluster spec generation (ZeRO-3, precision `bf16|fp8|mxfp4`, seed) | SHIPPED | `train/cluster.py:16-64` |
| Experiment tracking (MLflow/JSONL) wired per stage | PLANNED | `train/tracking.py:17` (`Tracker`) is shipped and exported (`train/__init__.py:9`) but never instantiated or called anywhere in the pipeline |
| Curriculum ordering (contracts→interpretation→loops→refusal) | PARTIAL | `train/curriculum.py:39` (`build_curriculum`); reachable via CLI `fx1 curriculum` (`cli.py:195-205`), NOT invoked by `Pipeline` |
| DPO / preference stage | PARTIAL | `train/dpo.py:98` (`build_preference_pairs`) + CLI `fx1 dpo`; there is no `Stage.DPO` / `run_dpo` in the pipeline |
| DISTILL stage (K3-LoRA teacher → servable student) | PLANNED | `LadderStage.DISTILL` enum only (`train/config.py:21`); no distillation implementation |
| LoRA MoE target modules (expert `gate/up/down` + shared experts) | PLANNED | `LoRAConfig.target_modules` is attention-only (`train/config.py:28-30`) |
| Stage fail-closed validators: fp16-on-K3, PROXY receipt before K3 | PARTIAL | fp16 is structurally impossible in `ClusterSpec.precision` (`train/cluster.py:23`); but `TrainConfig` has no precision/stage-precision gate and no PROXY-receipt-before-K3 requirement (`_k3_run_disclosure` `train/config.py:57-71`) |
| `extra="forbid"` on train config models | PLANNED | absent on `TrainConfig`/`LoRAConfig`/`ClusterSpec`/`TrainingReceipt`; only `forecast/config.py` sets it |

## Hardware reality (K3 base)

| Fact | Consequence for fx-1 |
|---|---|
| K3 checkpoint ≈ 1.4TB MXFP4 (≈5.6TB dequantized BF16) | No local training; all runs on rental clusters |
| Serving needs ≈64 accelerators; smallest community 1-bit quant ≈610GB RAM+VRAM | The planned production shape is a distilled student sized for a single node |
| 104B active parameters per token | Even LoRA touches multi-node memory; FINAL_K3 runs need ≥2 nodes (enforced in `TrainConfig`) |
| Kimi Delta Attention + native MXFP4 QAT | Use only frameworks with confirmed K3 support (check vLLM/SGLang/Unsloth status at run time); keep QAT formats intact |
| Preserved-thinking-history training | Corpus must keep `reasoning_content` + `tool_calls` (enforced in `TrainConfig`) |

## The ladder

> **Status: PARTIAL.** The ladder stages are validated config enums
> (`train/config.py:15-21`) with cost-disclosure gates for FINAL_K3
> (`train/config.py:57-71`). DISTILL has **no implementation** — enum only.
> No gate requires a prior PROXY receipt before a FINAL_K3 launch (PLANNED;
> see the doc-drift backlog in `docs/SOTA/09-finetuning-pipeline.md`).

1. **PROXY** — iterate corpus, eval harness, and LoRA plumbing on a small
   open model or quantized K3 serving build. Cost: hundreds of USD per run.
2. **FINAL_K3** — LoRA/QLoRA on the full K3 base via a rental multi-node
   cluster. Cost estimate committed to this file *before* launch; a serious
   run is five to six figures USD. v0.x never full-fine-tunes.
   > **Status: PLANNED caveat — LoRA scope.** `LoRAConfig.target_modules`
   > defaults to attention-only `q/k/v/o_proj` (`train/config.py:28-30`); the
   > K3 MoE expert (`gate/up/down`) and shared-expert projections are **not**
   > adapted by any shipped config or code path. MoE-aware targeting is
   > planned, not shipped.
3. **DISTILL** — distill the K3-LoRA teacher into a smaller servable fx-1
   student so production inference fits a single node.
   > **Status: PLANNED — not yet wired.** No distillation stage, trainer, or
   > GKD/MOPD implementation exists.

## Pipeline contract (enforced in code)

> Items 1–3 below are enforced in shipped code. Items 4–5 are contract intent;
> see their status callouts — they are **not** enforced by the pipeline today.

1. Build corpus: `fx1.data.build_corpus` — every line carries
   `receipt_sha256`; ineligible receipts become negative examples.
2. Run eval harness on the **base model** and record results — training is
   blocked without an on-record eval summary whose honesty gate passed.
3. `build_training_manifest` validates corpus provenance + eval gate + cost
   disclosure and writes the immutable run manifest.
4. Train on cluster; log checkpoints as `fx-1.vX.Y` with a model card:
   base checkpoint hash, corpus receipt range, domain/general eval deltas,
   license tier.
   > **Status: PLANNED — not yet wired.** `Stage.CARD` exists as an enum member
   > (`train/pipeline.py:35`) but there is no `run_card`; `modelcard.py` /
   > `mrm.py` are reachable only through the `fx1 modelcard` / `fx1 mrm` CLI
   > commands (`cli.py:147`, `cli.py:312`). The pipeline emits no model card.
5. Planned ship gate: a candidate must beat the K3 base on domain tasks,
   keep general-task scores at or above the base, and pass every honesty
   task natively.
   > **Status: PARTIAL — computed, not enforced.** `run_eval_candidate`
   > (`train/pipeline.py:152-171`) does run the paired bootstrap CI + McNemar
   > comparison (`eval/compare.py:60`) and writes `comparison.json`, but it
   > never fails closed on the result, compares **domain tasks only** (general
   > non-regression is not checked), enforces no refusal-rate floor, and there
   > is no `promoted` flag anywhere in the codebase. The honesty gate is
   > enforced only on the **base** model (`train/pipeline.py:114-122`), not on
   > the candidate.

## Corpus status

Seed corpus built from `receipts/`: 5 receipts loaded, 5 eligible (positive),
0 ineligible. Since v8 the loader also recognizes the lab's research-run
manifest schema (`data/metadata/research/runs/*.json`: `claim: "research_only"`
+ `synthetic` flag): 88 manifests → 77 positive (all labeled SYNTHETIC —
simulated-data evidence, never market evidence) + 11 negative (no claim →
fail-closed refusal examples). That directory is host-local (gitignored) —
regenerate it with the lab research pipeline; the corpus build degrades
gracefully without it. The corpus grows automatically as Phases 1–3 of
the roadmap (real-data benchmark, tournament, Dip Quality Score bench,
leaderboard) produce new gate-passed receipts — the lab's research output *is*
fx-1's training data flywheel.

## Industry-grade pipeline (v3)

Training runs follow a six-stage gated pipeline (`fx1.train.pipeline`):
`data → quality → eval_base → train → eval_candidate → card`. A stage that
cannot produce its evidence stops the pipeline — no skip flags.

> **Status: PARTIAL.** Five of the six stages are implemented
> (`run_quality_gate`, `run_eval_base`, `run_training`, `run_eval_candidate`
> in `train/pipeline.py`); `card` has **no runner** — see item 4 above. The
> per-bullet truth:

- **Quality gate:** dedup, near-dup shingle screen, eval-contamination
  removal, frozen split manifest (`fx1.data.quality`).
  **Status: SHIPPED** for the basic shingle-containment screen
  (`data/quality.py:64-68`, eval items defaulted from `DEFAULT_BANK` at
  `train/pipeline.py:95-97`). The richer 3-probe audit
  (`eval/contamination.py:132`: n-gram + Min-K% + rephrased gap) is **CLI-only**
  (`fx1 contamination-audit`) and never runs inside the pipeline; the screen
  also does not fail closed when eval items are unavailable —
  **Status: PLANNED** for pipeline wiring.
- **Eval-before-train:** base-model results on the eval bank are mandatory
  and hashed into the run. **Status: SHIPPED** (`train/pipeline.py:114-124`;
  receipt binding at `train/receipts.py:70-100`).
- **Immutable training receipts** (`fx1.train.receipts`): git revision,
  dirty-worktree flag, config/corpus/split/eval hashes, seed, environment
  fingerprint; `verify_training_receipt` re-checks hashes against disk.
  **Status: SHIPPED** (`train/receipts.py:51-127`). Note: the candidate-eval
  and comparison hashes are **not** bound into the receipt (issued before
  training, `train/pipeline.py:126-137`) — eval-digest binding is PLANNED.
- **Statistical ship gate** (`fx1.eval.compare`): paired bootstrap CI on
  pass-rate deltas + McNemar; domain improvement must exclude zero.
  **Status: PARTIAL** — the statistics are computed and written
  (`train/pipeline.py:164-166`), but "must exclude zero" is **not enforced**:
  the pipeline advances regardless of the verdict, and there is no
  refusal-rate floor. Enforcement is PLANNED.
- **Cluster configs generated, not hand-edited** (`fx1.train.cluster`):
  validated multi-node ZeRO-3 specs for the 2.8T MoE; long-context runs
  require ≥8 nodes; MXFP4 QAT runs require ZeRO-3.
  **Status: SHIPPED** (`train/cluster.py:16-64`), but note `ClusterSpec` is
  not consumed by `Pipeline.run_training` — the injected trainer receives the
  `TrainConfig` only (`train/pipeline.py:41-51,138-142`).
- **Experiment tracking** (`fx1.train.tracking`): MLflow when available,
  JSONL always; tracking is observability — the receipt is the evidence.
  **Status: PARTIAL** — the `Tracker` class ships and is exported
  (`train/__init__.py:9`), but **nothing constructs or calls it** in the
  pipeline; per-stage logging is PLANNED.
- **Preference stage** (`fx1.train.dpo`): contract-derived DPO pairs make
  honesty native.
  **Status: PARTIAL** — pair construction ships (`train/dpo.py:98-120`,
  CLI `fx1 dpo`), but there is **no DPO stage** in `Stage` /
  `Pipeline` and no trainer consumes the pairs. Running DPO is PLANNED.
- **Curriculum ordering** (`fx1.train.curriculum`): contracts → interpretation
  → research loops → refusal.
  **Status: PARTIAL** — `build_curriculum` ships (`train/curriculum.py:39-70`,
  CLI `fx1 curriculum`) but the pipeline's quality gate never invokes it;
  corpus ordering in training runs is file order.
- **CI** (`.github/workflows/fx1.yml`): fx1 tests, lint, blocking honesty
  inheritance, and a corpus-contract smoke run on every fx1 change.
  **Status: SHIPPED.**
