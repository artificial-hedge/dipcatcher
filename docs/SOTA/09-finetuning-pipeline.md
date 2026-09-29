# 09 — Fine-tuning & training pipeline SOTA

**Lane:** FINE-TUNING & TRAINING PIPELINE SOTA
**Scope:** `src/fx1/train/`, `src/fx1/data/`, `src/fx1/eval/`, `docs/FX1_TRAINING.md`
**Base model:** Kimi K3 open weights (2.8T MoE, 104B active, native MXFP4)
**Status:** research notes + pipeline assessment + adoption plan. No code changed by this doc.

> **Honesty contract.** Everything here is research methodology for a
> `research_only=true`, `live_pnl_claim=false` model. No Sharpe/Sortino/Calmar/
> P&L/NAV headline is produced or implied by any pipeline described below;
> candidate promotion is gated on **proper scores** (eval-suite accuracy with
> paired-bootstrap CIs and McNemar, honesty-bait refusal rate, contamination
> probes). Synthetic corpora stay labeled `SYNTHETIC` and are correctness
> tests, never market evidence. Gaps marked **P0** are blocking; items marked
> *(plan)* are proposed work, not shipped behavior — this doc does not claim
> the pipeline already does them.

---

## 0. TL;DR

fx-1's training scaffolding is honest about cost and well-structured
(`Pipeline` stage gates, `TrainingReceipt`, `eval_compare` with bootstrap CI +
McNemar, `modelcard`/`mrm`), but the **trainer configuration surface is thin**
and several documented behaviors are **not wired into the pipeline**. The three
highest-leverage moves:

1. **Config hygiene + hyperparameter gates** — every config model gets
   `extra="forbid"`; `TrainConfig`/`LoRAConfig`/`DeepSpeedConfig` grow the
   SOTA knobs (scheduler, warmup, weight decay, grad clip, seed, precision,
   packing, loss masking, DoRA/rsLoRA, MoE target modules, replay/anchor) with
   stage-aware validators that *fail closed*.
2. **Run manifests tied to `receipts/`** — replace the ad-hoc
   `training_receipt.json` with a sealed `receipt.v2` envelope (dataset hash,
   code fingerprint, environment fingerprint, seed/determinism flags, bound
   eval digest, `live_pnl_claim=false`) verified by `dipcatcher verify-receipt`.
3. **Close the doc↔code drift** — `docs/FX1_TRAINING.md` advertises curriculum
   ordering, per-level dedup, contamination probes, packing, tracking, and a
   card stage that the pipeline never invokes. Either wire them or soften the
   doc; shipping the claim without the code is an honesty-contract exposure.

---

## 1. The K3 substrate dictates the recipe

Kimi K3 is not a generic dense checkpoint; the fine-tuning plan must respect
what the base model already is (Kimi K3 tech report, 2026):

- **2.8T-parameter MoE, 104B active**, 93 layers (69 KDA + 24 Gated MLA),
  **896 routed experts, 16 active/token** + 2 shared experts (Stable LatentMoE).
- **Native MXFP4 weights / MXFP8 activations** — quantization-aware from SFT
  onward, not post-hoc. FP16 training is off-table; BF16 mixed precision is the
  floor.
- **Quantile Balancing** (bias-only dispatch regularization) stabilizes routing
  at extreme sparsity; router aux-loss / z-loss still matter when fine-tuning
  experts to avoid dead-expert collapse.
- Post-training is **SFT → domain RL → Multi-Teacher On-Policy Distillation
  (MOPD)** with per-problem token-budget curriculum; the open weights are the
  *consolidated* artifact. Downstream SFT (fx-1) sits on top of an already
  RL+distilled policy, so **forgetting pressure is high** and on-policy /
  anchored methods are preferred over pure off-policy SFT.
- Data recipe: rule + classifier quality filtering, **exact + fuzzy
  deduplication**, per-domain sampling rates set by small-model ablation, and a
  **rephrasing** pass (style/perspective-diverse regeneration with fidelity
  verification) for knowledge/math.

Implication for fx-1: LoRA must target **MoE expert + attention** projections
(not just `q/k/v/o_proj`), precision must be **BF16/MXFP4-aware**, expert
routing must be **load-balanced**, and the SFT stage needs an explicit
**anti-forgetting** term (replay and/or KL anchor / on-policy distillation).

---

## 2. SOTA practice summaries (2024–2026)

### 2.1 Parameter-efficient fine-tuning: LoRA → QLoRA → DoRA → MoE-PEFT

| Method | Core idea | When to use | Notes |
|---|---|---|---|
| **LoRA** (Hu et al., 2021, arXiv:2106.09685) | Freeze \(W\), learn \(\Delta W = BA\), \(r \ll d\) | Default PEFT; adapter isolation preserves base | Scale \(\alpha/r\); apply to **all** linear layers, not just attention, for best quality (Hu et al. §7.1) |
| **QLoRA** (Dettmers et al., 2023, arXiv:2305.14314) | 4-bit NF4 frozen base + double quantization + paged optimizers + LoRA in BF16 | Single-GPU / memory-bound; ~33B on one 24GB GPU | NF4 is info-theoretically optimal for normal weights; **double quant** saves ~0.37 bit/param; paged optimizers ride CPU memory on spikes |
| **DoRA** (Liu et al., ICML 2024, arXiv:2402.09353) | Decompose \(W = m \cdot V/\lVert V\rVert\); LoRA on direction \(V\), separate magnitude \(m\) | Consistently beats LoRA at small rank, esp. visual/multitask; **closes ~half the gap to full FT at r=32–64** | Overhead is modest; strong default upgrade over LoRA when rank is budgeted |
| **rsLoRA** (Kalajdzievski, 2023, arXiv:2312.03732) | Scale \(\alpha/\sqrt{r}\) instead of \(\alpha/r\) | Any run with \(r \ge 32\) | Without it, high-rank LoRA **underperforms** low-rank (rank-stability); PEFT exposes `use_rslora` |
| **LoRA+** (Hayou et al., ICML 2024, arXiv:2402.12354) | \(\eta_B \gg \eta_A\) (ratio ~16) | Feature learning at large width | Cheap win; not yet standard in all trainers |
| **µP / µTransfer** (Yang et al., 2022, arXiv:2203.03466) | Width-scaled init/LR/regularizer so proxy HPs transfer zero-shot to target scale | **fx-1's PROXY→FINAL_K3 ladder is exactly this use case** | Adam µP prescription: per-layer LR \(\propto 1/\text{fan\_in}\), residual init \(\propto 1/\sqrt{L}\); tune small, transfer free |
| **MoE-Sieve** (2025–2026) | Route-guided expert selection: LoRA only on top-utilization experts (top ~25%) | MoE PEFT at scale; beats full-FT baseline at 530B with ~10% params | Avoids adapting dead experts; pairs with load-balance monitoring |
| **MixLoRA** (2025) | Fine-grained expert-level LoRA + routing-aware load balance for MoE | Concurrent multi-domain MoE adaptation | Relevant if fx-1 ever trains domain-specific adapters on K3's 896 experts |

**Practice rules of thumb (2024–2026 consensus):**

- **r=16–64, \(\alpha = 2r\)** (or \(\alpha=r\) + rsLoRA) for instruction SFT;
  r=32+DoRA is the sweet spot when quality matters and budget allows.
- **Target all linear projections** — attention (`q,k,v,o`) **and** MLP/expert
  (`gate,up,down`) — except the LM head. Attention-only LoRA (fx-1's current
  default) is the single most common quality leak.
- **LoRA dropout 0.05–0.1** for small corpora; 0 for large.
- **Merge only for inference**; keep adapters unmerged for composition,
  rollback, and WiSE-FT/soup experiments.
- **BF16 compute, FP32 master weights, FP32 optimizer states** — never FP16 for
  a MXFP4-native MoE.

### 2.2 Learning-rate schedules & warmup for LLM SFT

- **Linear warmup (1–5% of steps) then cosine decay to ~10% of peak** remains
  the default for SFT (T5, PaLM, Llama recipes). Warmup prevents the
  large-init/Adam-variance instability in early steps; skipping it on MoE risks
  router collapse.
- **WSO (Warmup-Stable-Only)** — linear warmup then constant — is increasingly
  used for **short SFT runs and continued pretraining** where cosine decay to
  near-zero wastes the tail; pair with a short final anneal. For fx-1's
  1–3 epoch LoRA SFT, **cosine with a 3% warmup** or **WSO + 10% anneal** are
  both defensible; the key is that the schedule is *explicit and recorded*,
  not the trainer default.
- **Peak LR for LoRA SFT is ~10× full-FT**: \(1\text{e-}4 \to 3\text{e-}4\) for
  r=16–64 (fx-1's `1e-4` default is reasonable, slightly conservative for
  r≥32). DPO/RL stages drop LR ~5–10× below SFT (5e-6–1e-5 full, 1e-5–5e-5 LoRA).
- **Token-horizon / batch-size scaling:** effective LR should track
  \(\sqrt{\text{batch}}\) (or linearly for small batch). Because K3 SFT runs on
  ~1.6M–4M tokens/GPU/day, **batch-size and LR must be co-recorded**; a config
  that fixes LR without fixing global batch is not reproducible.
- **LR re-warming for continued training:** when resuming or switching data
  mixture mid-run, re-warm LR over a few hundred steps rather than continuing
  the decayed tail (avoids the "decay-to-zero then new data" catastrophe).
- **Hyperparameter transfer:** tune schedule/LR on the PROXY stage and transfer
  via µP scaling rather than re-tuning at FINAL_K3 — this is the explicit
  purpose of fx-1's ladder and should be encoded as a gate (proxy receipt must
  exist and its HP block must match, modulo µP scaling, before K3 launch).

### 2.3 Data mixture & curriculum

- **Mixture is a first-class hyperparameter.** K3 sets per-domain sampling rates
  by small-model ablation; DoReMi (Xie et al., 2023, arXiv:2305.10429) learns
  domain weights with a small proxy via group DRO; **Data Mixing Laws** (Ye et
  al., 2024, arXiv:2403.16952) fit a scaling law over mixtures to predict the
  optimum. fx-1's `quality_weight`/`competence_level` fields are the right
  primitive but are **never consumed** by a sampler.
- **Curriculum ordering** (easy→hard, or capability-staged) gives modest but
  reliable gains for instruction SFT; K3's own post-training uses a **stage-wise
  budget curriculum** (anneal token-budget multiplier τ from max→high→low
  effort). fx-1's `curriculum.build_curriculum` (CONTRACTS→INTERPRETATION→
  RESEARCH_LOOP→REFUSAL) is well-designed and **should be wired into the
  quality gate**, with per-level dedup and a recorded level histogram.
- **Quality > quantity.** LIMA (Zhou et al., 2023, arXiv:2305.11206): 1,000
  curated examples rival 52k Alpaca. LIMO (2025, arXiv:2502.03373): 800
  examples targeting ~8 cognitive templates elicit strong reasoning. DEITA
  (Liu et al., 2024, arXiv:2312.15685): score **complexity × quality**, then
  greedily select for **embedding diversity** — beats larger sets at 6–10k.
  IFD (Li et al., 2023, arXiv:2308.12032): keep examples whose
  answer-given-instruction loss / answer-alone loss ratio is high (genuinely
  instruction-dependent). **Practical:** a few thousand high-IFD, diverse,
  deduplicated examples with reasoning traces preserved (K3's
  `preserve_reasoning_content`) will outperform a 10× larger noisy set.
- **Rephrasing / synthesis:** K3 rephrases knowledge+math with style-diverse
  prompting and **fidelity verification against source**. Any fx-1 synthetic
  corpus must carry a `SYNTHETIC` label and a provenance receipt
  (`corpus.py:record_provenance` already does this) and should be
  fidelity-checked, not trusted blindly.
- **Cooldown / annealing data:** late-training should upsample long-context and
  high-quality data (K3 §long-context); for SFT this maps to ending the
  curriculum on the hardest, most-aligned tier (REFUSAL) rather than a random
  tail.

### 2.4 Deduplication & decontamination

- **Exact dedup** (SHA-256 on normalized text) is table stakes — fx-1 has it.
- **Near-dedup at scale = MinHashLSH**, not pairwise Jaccard. fx-1's
  `dedup_and_filter` does an **O(n²) pairwise 8-token-shingle Jaccard**
  (`data/quality.py:66-79`) which is correct but won't survive a real corpus;
  LSH (Broder 1997; Indyk & Motwani 1998; `datasketch`) gives sub-quadratic
  recall at Jaccard ≥ ~0.7. **Per-domain dedup** (K3) beats one global pass.
- **Semantic dedup:** SemDeDup (Abbas et al., 2023, arXiv:2303.09540) removes
  embedding-space duplicates that survive lexical dedup; SEDD/FED variants exist
  for instruction data. Worth a *(plan)* tier once lexical dedup scales.
- **Decontamination is mandatory and must fail closed.** Standard practice
  (GPT-3, Llama, PaLM): **13-gram** (or 8–20 token shingle) containment against
  every eval/benchmark item; drop or flag any training doc that contains a
  benchmark span. fx-1 has the right function (`contamination_containment`,
  threshold 0.6) **but `Pipeline.run_quality_gate` never passes `eval_items`**
  (`pipeline.py:76-77`), so the probe is dead code in the live path. The richer
  3-probe audit in `eval/contamination.py` (n-gram + min-k% + rephrased gap) is
  also unwired. This is a **P0 correctness/honesty gap**: a model can be
  promoted on memorized eval items.
- **Rephrased-gap probe** (`contamination.py:rephrased_gap`) is the SOTA
  addition most labs skip — it separates memorization from capability by
  comparing original vs LLM-rephrased items. Keep it; wire it.

### 2.5 Early stopping & checkpoint selection

- **Validation perplexity is a weak selector for instruction SFT.** UGCS
  (Guo et al., 2025, arXiv:2503.13464): pick the checkpoint that is
  **uncertainty-aware** on *downstream* tasks — estimate the variance of the
  test statistic under model sampling and stop when improvement is within
  noise. RewardRank / reward-model-guided selection is the alignment analogue.
- **Always carry error bars.** Miller (2024, arXiv:2411.00640): report paired
  bootstrap CIs and **McNemar** for matched-item comparisons; a single-seed
  accuracy delta is not evidence. fx-1's `eval/compare.py` already does exactly
  this (bootstrap CI on the mean difference + McNemar with continuity
  correction, honest "cannot be attributed" language) — **it just isn't wired
  into a promotion gate.**
- **Checkpoint averaging / merging:** **Model Soups** (Wortsman et al., 2022,
  arXiv:2203.05482) average weights of same-arch fine-tunes; **WiSE-FT**
  (Wortsman et al., 2022, arXiv:2109.01903) interpolates zero-shot and
  fine-tuned weights for robustness; **DARE** (Yu et al., 2024,
  arXiv:2311.03099) drops-and-rescales deltas before merging. For LoRA, average
  or WiSE-FT-interpolate the **adapter** against the frozen base to recover
  general ability lost to SFT — a cheap, strong forgetting mitigation.
- **Early stopping:** patience on the *downstream* metric (not train loss),
  with a held-out honesty-bait refusal floor as a hard stop (refusal rate must
  not drop below base). fx-1 has `best_metric` on `TrainingReceipt` but nothing
  ever sets or reads it.

### 2.6 Catastrophic-forgetting mitigation

SFT on a narrow domain erodes the base policy's general + reasoning ability,
and K3's base is already an RL+MOPD-consolidated artifact, so the erosion is
into a *strong* prior. SOTA mitigations, roughly in order of fx-1 fit:

1. **Adapter isolation (LoRA/DoRA).** The frozen base *is* the anchor; merging
   is opt-in and reversible. fx-1 already benefits from this — keep adapters
   unmerged and retain the base path for A/B (`eval_compare`).
2. **Replay / rehearsal.** Mix in a held-out slice of general/base-distribution
   data (5–15%) during SFT. **Self-synthesized replay** (generate base-model
   continuations on general prompts, replay them) avoids needing the original
   pretraining data. *(plan)* — fx-1 has no replay knob.
3. **Anchored / trust-region updates.** Add a KL or anchor term keeping
   \(\pi_\theta\) near \(\pi_{\text{base}}\) (dynamic anchor weight; on-policy
   distillation / **GKD**, Agarwal et al., 2024, arXiv:2306.13649, minimizes
   reverse-KL on student-generated samples). K3's own MOPD is multi-teacher
   on-policy distillation — the natural DISTILL-stage method for fx-1.
4. **Preference stage after SFT.** DPO (Rafailov et al., 2023,
   arXiv:2306.05685) with **on-policy synthetic pairs** and small β (0.1)
   re-anchors refusals/honesty without a separate reward model; fx-1's
   `dpo.py` builds honesty-bait pairs already but **no pipeline stage runs DPO**.
5. **Weight interpolation (WiSE-FT / soup)** post-hoc to trade domain gain
   against base retention (see §2.5).
6. **LR discipline + fewer epochs.** Over-training is the main cause of
   forgetting; 1–2 epochs, cosine decay, early stop on the downstream metric.

### 2.7 Numerical stability: BF16 vs FP16

- **BF16 is the default for LLM training.** 8 exponent bits = FP32 dynamic
  range, so **no loss scaling** is needed; 7 mantissa bits trade precision for
  stability. FP16 (5 exponent bits) **requires dynamic loss scaling** to avoid
  gradient underflow and is prone to overflow in attention/softmax — the
  Nanotron guide and Megatron both default to BF16 on modern accelerators.
- **FP32 master weights + FP32 optimizer states** (Adam moments) are mandatory
  even under BF16 compute; the BF16 working copy is re-cast each step.
- **Stochastic rounding** recovers expected-value fidelity for low-precision
  accumulation; used by DeepSpeed and in MXFP4/FP8 recipes.
- **For K3 specifically:** the model is **MXFP4-weight / MXFP8-activation
  native**. The correct mixed-precision config is **BF16 master + MXFP4/FP8
  forward where the recipe supports it**, *not* FP16. fx-1's `DeepSpeedConfig`
  hardcodes `fp16.loss_scale=0`/`window=1000` when precision is fp16
  (`cluster.py:39-44`) — fine generically, but **FP16 should be forbidden for
  K3** by a stage gate, and an MXFP4/FP8 precision mode should exist.
- **MoE routing stability:** router **z-loss** (penalize large logits) and
  **aux/load-balance loss** (or K3's Quantile Balancing bias update) prevent
  dead experts and logit blow-up. Neither is exposed in fx-1's cluster config.
  When LoRA-adapting experts, monitor expert utilization and keep aux-loss on.
- **Grad clipping** (`max_grad_norm`, default 1.0) is universal; fx-1 doesn't
  pass it from config (relies on trainer default).

### 2.8 Training-run reproducibility

The reproducibility stack (PyTorch determinism guide; DeepSpeed; HF):

- **Seed everything:** `torch.manual_seed`, `transformers.set_seed`, Python
  `random`, NumPy, and **per-rank** seeding under distribution. fx-1 sets
  **no seed anywhere** — two identical configs are not bit-reproducible.
- **Deterministic kernels:** `torch.use_deterministic_algorithms(True)`,
  `cudnn.deterministic=True`, `cudnn.benchmark=False`, and the env var
  **`CUBLAS_WORKSPACE_CONFIG=:4096:8`** (required for cuBLAS determinism on
  CUDA ≥10.2). Note the throughput cost — gate it behind a `deterministic`
  flag rather than always-on.
- **Config hashing:** fx-1 already does this well —
  `params_sha256 = sha256(canonical_json_bytes(config.model_dump()))`
  (`receipts.py:60-61`). Keep it; extend the receipt to also hash the **corpus
  content** (not just `corpus_records` count) and the **code** and
  **environment**.
- **Run manifest / environment fingerprint:** the lab's `receipt.v2`
  (`src/quant_fund/research/receipt_v2.py`) is the house standard —
  interpreter, numeric stack, BLAS/LAPACK build, threadpools, a
  `fingerprint_sha256`, `code_fingerprint`, `dataset`/`params` digests,
  `live_pnl_claim=false`, `verdict`, canonical SHA-256 seal, verified by
  `dipcatcher verify-receipt`. fx-1's `TrainingReceipt` is a **separate, weaker
  schema** that does not seal into `receipts/` and is not verifiable by the
  lab's receipt command. Unify.
- **Git state:** record commit + dirty flag (the lab has
  `quant_fund.utils.reproducibility.git_revision`); fx-1's receipt omits it.

---

## 3. Assessment of the current fx-1 pipeline

Stage-by-stage, against `docs/FX1_TRAINING.md`'s claimed pipeline
`data → quality → eval_base → train → eval_candidate → card`.

| Stage | Code | What it does | SOTA delta |
|---|---|---|---|
| **data** | `data/corpus.py` | Builds SFT JSONL with `source_id/kind`, `competence_level`, `quality_weight`, provenance `receipt_sha256`; synthetic detection | Strong schema. **No sampler consumes `quality_weight`/`competence_level`**; `max_records` truncates unstratified; no mixture law / DoReMi |
| **quality** | `Pipeline.run_quality_gate` → `data/quality.dedup_and_filter` | Exact SHA dedup + O(n²) 8-token Jaccard ≥0.72 + contamination containment ≥0.6 (**only if `eval_items` passed**) | **Contamination probe is dead** (no `eval_items` passed, `pipeline.py:76`); pairwise Jaccard won't scale (no MinHashLSH); no per-domain dedup; no SemDeDup; **curriculum not invoked**; no quality/IFD/DEITA filter |
| **eval_base** | `run_eval_base` → `eval.evaluate` | Runs base model on task suite, writes `eval_base.json`, `baseline_pass=True` | Good gate. **Not compared to candidate** here; no CI/McNemar at this stage |
| **train** | `run_training` → `hf_train` | `hf_train(model, corpus, adapter_out, loss_masking=True, epochs, lr, batch_size)`; DeepSpeed via `cluster.py` | **Thin surface:** no scheduler/warmup/weight-decay/grad-clip/seed/precision/packing/grad-accum passed from config; LoRA targets attention only; no DoRA/rsLoRA/QLoRA; no MoE EP/z-loss/aux-loss; no replay/anchor; no checkpoint selection; no early stopping; `tracker` constructed but never used; DPO stage absent |
| **eval_candidate** | `run_eval_candidate` | Runs candidate adapter, reads `pass_rate`/`score` | **No promotion gate**: nothing compares candidate vs base with `eval_compare` CI/McNemar; no refusal-floor hard stop; `best_metric` never set |
| **card** | `Stage.CARD` enum only | — | **`run_card` does not exist.** `modelcard.py` + `mrm.py` are complete but the pipeline never calls them; `receipts.py` docstring even references `Pipeline.run_card()` (`receipts.py:5`) that isn't there |
| **receipt** | `train/receipts.py` | `TrainingReceipt` JSON in run dir; `params_sha256` config hash; verify checks params/base/corpus-count | **Not `receipt.v2`**, not sealed into `receipts/`, no dataset/code/env fingerprint, no seed/determinism, no eval-digest binding, no `live_pnl_claim`/`research_only` on the receipt, not verifiable by `dipcatcher verify-receipt` |

**What's genuinely good (keep):**

- `eval/compare.py` — bootstrap CI + paired McNemar + honest attribution
  language is exactly Miller (2024) best practice.
- `eval/contamination.py` — 3-probe audit incl. rephrased gap is ahead of most
  labs.
- `data/corpus.py` provenance (`record_provenance`, `receipt_sha256`) and
  synthetic handling.
- `modelcard.py` / `mrm.py` — M1–M5 invariant checks, refusal taxonomy,
  `live_pnl_claim=false`, sealed `receipt_sha256`.
- `config.py` hardware-disclosure validators (`_k3_run_disclosure`) — honest
  cost gating before launch.
- `cluster.py` ZeRO-3 + activation-checkpointing defaults for large models.

**Doc↔code drift (honesty exposure).** `docs/FX1_TRAINING.md` states the
pipeline does curriculum ordering, per-level dedup, eval-contamination probes,
sequence packing, gradient accumulation, checkpointing, MLflow tracking, and a
card stage. As of this assessment the pipeline does **none** of curriculum /
contamination-with-items / packing-config / tracking / card, and checkpointing
is just `save_path`. This is the kind of overclaim the honesty contract exists
to prevent. **P0:** either implement or correct the doc.

---

## 4. Gap list

Severity: **P0** blocking (correctness/honesty), **P1** major (quality/
reproducibility), **P2** enhancement.

### P0 — blocking

1. **Contamination probe never runs on real eval items.**
   `run_quality_gate` calls `dedup_and_filter(clean)` with no `eval_items`
   (`pipeline.py:76-77`), so `contamination_containment` is skipped. A model can
   be promoted on memorized benchmarks. *Fix:* pass the registered eval items;
   **fail closed** (raise) if no eval items are available; wire
   `eval/contamination.audit_contamination` into `run_eval_candidate`.
2. **No promotion gate.** `run_eval_candidate` records a pass_rate but never
   compares candidate vs base. *Fix:* run `eval_compare` (CI + McNemar), require
   a non-overlapping CI improvement **and** a refusal-rate floor ≥ base before
   `TrainingReceipt.promoted=True`; record the comparison digest.
3. **`Stage.CARD` is dead; no `run_card`.** Model card + MRM dossier are never
   produced by the pipeline, yet `receipts.py` and `FX1_TRAINING.md` reference
   the card stage. *Fix:* implement `run_card` calling
   `modelcard.validate` + `mrm.compile_dossier`; seal outputs.
4. **Training receipt is not `receipt.v2` and not in `receipts/`.** Not
   verifiable by the lab's `verify-receipt`; no dataset/code/env fingerprint; no
   `live_pnl_claim`/`research_only` flags on the receipt itself. *Fix:* §5.2.
5. **Doc overclaims pipeline features.** *Fix:* implement P0-1..4 or soften
   `FX1_TRAINING.md` to match shipped behavior.

### P1 — major

6. **LoRA targets attention only** (`config.py:28-30`: `q,k,v,o_proj`). On a
   896-expert MoE this leaves expert MLPs (`gate/up/down`), shared experts, and
   (carefully) the router gate unadapted — the dominant quality leak. *Fix:*
   MoE-aware `target_modules`; expose `use_dora`, `use_rslora`, `quantization`
   (none/nf4/fp8/mxfp4); α/r validator.
7. **No seed, no determinism.** Nothing sets a seed or
   `CUBLAS_WORKSPACE_CONFIG`/`use_deterministic_algorithms`. Runs are not
   reproducible. *Fix:* `seed` on `TrainConfig`; `deterministic` flag; record
   both in the receipt.
8. **Trainer hyperparameters not in config.** Scheduler, warmup, weight decay,
   grad clip, optimizer, precision, grad-accum, micro-batch, packing, sequence
   length are either trainer defaults or hardcoded in `cluster.py`
   (`micro=1, accum=16, global=256`). *Fix:* first-class fields + stage gates;
   pass through `hf_train`.
9. **FP16 allowed for K3.** `DeepSpeedConfig(precision="fp16")` is legal but
   wrong for an MXFP4-native MoE. *Fix:* stage gate forbids fp16 on
   FINAL_K3/DISTILL; add bf16/mxfp4/fp8 modes.
10. **No MoE routing stability knobs.** No z-loss / aux-loss / Quantile-Balancing
    config, no expert-parallel (EP) flags, no utilization monitoring. *Fix:*
    expose in `cluster.py`; assert aux/z-loss on when adapting experts.
11. **Curriculum + mixture unused.** `build_curriculum` and `quality_weight`
    exist but no sampler/ordering uses them. *Fix:* wire curriculum into the
    quality gate; add a weighted sampler; record the level histogram + mixture
    digest in the receipt.
12. **Dedup won't scale.** O(n²) pairwise Jaccard. *Fix:* MinHashLSH
    (`datasketch`), per-domain passes; keep exact-hash fast path.
13. **No anti-forgetting stage.** No replay, no KL anchor, no DPO stage, no
    WiSE-FT/soup. *Fix:* `replay_ratio` + `anchor_kl_beta` on `TrainConfig`;
    add a DPO stage using `dpo.py`; add adapter WiSE-FT interpolation option.
14. **Tracker never used.** `MLflowTracker` constructed in `Pipeline.__init__`
    but no `log_*` calls. *Fix:* log metrics/params/artifacts per stage.
15. **No early stopping / checkpoint selection.** `best_metric` unused; single
    final adapter only. *Fix:* `eval_steps`/`save_steps`/patience;
    UGCS-style downstream selection; persist the selected-checkpoint digest.

### P2 — enhancement

16. **No SemDeDup / embedding dedup** (after MinHashLSH lands).
17. **No IFD/DEITA quality scoring** for example selection.
18. **No µP/µTransfer hyperparameter transfer** from PROXY to FINAL_K3 (the
    ladder's stated purpose); encode as a gate that the proxy receipt exists and
    HPs match modulo scaling.
19. **No LoRA+/per-layer LR** option.
20. **No on-policy distillation (GKD/MOPD)** in the DISTILL stage (enum exists,
    no implementation).
21. **`max_records` truncation is unstratified** — bias toward file order.
22. **No `extra="forbid"`** on `TrainConfig`/`LoRAConfig`/`LadderStage`/
    `DeepSpeedConfig`/`TrainingReceipt` — typos in configs are silently ignored
    (ADR-0006 violation; `receipt_v2.py` models all set it).

---

## 5. Adoption plan

Phased; each phase ships with `tests/fx1` acceptance tests and a gate. No phase
weakens the honesty contract.

### 5.1 Config hygiene (P0-adjacent, P1, P2-22)

**Goal:** configs fail closed, carry the full SOTA surface, and are
stage-validated.

1. Add `model_config = ConfigDict(extra="forbid")` to **every** fx-1 config
   model (`TrainConfig`, `LoRAConfig`, `LadderStage` consumers,
   `DeepSpeedConfig`, `ZeROConfig`, `TrainingReceipt`, `DPOConfig`,
   `PreferencePair`). Mirror `receipt_v2.py`.
2. **`LoRAConfig`** gains:
   - `use_rslora: bool = False`, `use_dora: bool = False`
   - `quantization: Literal["none","nf4","fp8","mxfp4"] = "none"`
   - `target_modules` default expanded to K3 MoE set
     (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` + shared-expert
     projections; router gate opt-in and load-balanced)
   - validator: `alpha >= rank`; recommend `alpha == 2*rank` or rsLoRA;
     `rank <= 64` for FINAL_K3 unless justified.
3. **`TrainConfig`** gains (all recorded in the receipt):
   - `lr_scheduler_type: Literal["cosine","constant_with_warmup","wso"] = "cosine"`
   - `warmup_ratio: float = Field(0.03, ge=0.0, le=0.1)`
   - `weight_decay: float = Field(0.0, ge=0.0, le=0.1)`
   - `max_grad_norm: float = Field(1.0, gt=0)`
   - `optimizer: Literal["adamw_torch","paged_adamw_8bit","muon"] = "adamw_torch"`
   - `precision: Literal["bf16","fp16","mxfp4","fp8"] = "bf16"`
   - `seed: int = 42`, `deterministic: bool = False`
   - `micro_batch_size`, `gradient_accumulation_steps`, `global_batch_size`
     (consistency validator: `global == micro * accum * world_size`)
   - `max_seq_len`, `sequence_packing: bool = False` (with block-diagonal
     attention + position-reset assertion when True), `loss_masking: bool = True`
   - `eval_steps`, `save_steps`, `early_stopping_patience`,
     `metric_for_best_model`, `checkpoint_selection: Literal["last","best_downstream","ugcs"]`
   - `replay_ratio: float = Field(0.0, ge=0.0, le=0.3)`,
     `anchor_kl_beta: float = Field(0.0, ge=0.0)`
4. **Stage gates** (`@model_validator(mode="after")`):
   - FINAL_K3/DISTILL: `precision != "fp16"`; `estimated_nodes >= 2`; MoE
     target modules present; aux/z-loss enabled when experts adapted;
     `preserve_reasoning_content=True`.
   - PROXY: must have a prior PROXY receipt to launch FINAL_K3 (µP transfer
     gate, P2-18).
   - DISTILL: requires a teacher receipt + OPD/GKD config (P2-20).
5. **Acceptance tests:** `tests/fx1/test_train_config_gates.py` — extra fields
   rejected; fp16-on-K3 rejected; batch consistency enforced; α/r validator;
   MoE targets required; proxy-receipt gate.

### 5.2 Run manifests tied to `receipts/` (P0-4)

**Goal:** every training run seals a `receipt.v2` envelope the lab can verify.

1. Replace `TrainingReceipt` ad-hoc JSON with a **`receipt.v2` build** via
   `quant_fund.research.receipt_v2.build_receipt_v2` (or an fx-1 adapter that
   emits the same envelope), carrying:
   - `kind="fx1_training"`, `data_label`, `verdict ∈ {pass,fail,blocked}`
   - `dataset`: `{corpus_sha256 (content hash, not count), records, mixture_digest, level_histogram, contamination_digest}`
   - `params`: full `TrainConfig.model_dump()` + `params_sha256`
   - `code_files`: fingerprint of `src/fx1/train/*.py` + the exact `hf_train`/
     `cluster` writer files → `code_sha256`
   - `environment`: `environment_fingerprint()` (interpreter, numeric stack,
     BLAS, threadpools, GPU/accelerator list, `CUBLAS_WORKSPACE_CONFIG`,
     `torch.__version__`, CUDA version) → `fingerprint_sha256`
   - `seed`, `deterministic`, `git_revision` (commit + dirty)
   - `eval_binding`: `{eval_base_sha256, eval_candidate_sha256, compare_digest (CI+McNemar), refusal_rate_base, refusal_rate_candidate}`
   - **`live_pnl_claim=false`, `research_only=true`** (hard)
   - `promoted`, `stage`, `hardware` disclosure, `license_tier`
2. **Seal into `receipts/`** (`receipt_dir` already on `Pipeline`), signed with
   the canonical SHA-256 seal; verifiable by `dipcatcher verify-receipt`.
3. `verify_training_receipt` extended to check the v2 seal, dataset/code/env
   digests, eval binding, and the `live_pnl_claim=false` invariant — not just
   params/base/count.
4. **Acceptance tests:** `tests/fx1/test_training_receipt_v2.py` — envelope
   validates against `receipt_v2.schema.json`; `verify-receipt` passes;
   tampering any digest fails; `live_pnl_claim` cannot be true.

### 5.3 Hyperparameter gates + pipeline wiring (P0-1/2/3, P1)

**Goal:** the pipeline enforces SOTA defaults and gates promotion on evidence.

1. **Quality gate** (`run_quality_gate`):
   - Pass registered `eval_items`; **raise if none** (fail closed).
   - Invoke `curriculum.build_curriculum`; record level histogram + mixture
     digest.
   - Swap pairwise Jaccard → **MinHashLSH** (`datasketch`), per-domain; keep
     exact-hash fast path.
   - Run `eval/contamination.audit_contamination` (3-probe) on the cleaned set;
     block on rephrased-gap ≈ 0.
2. **Train** (`run_training`):
   - Set seed + determinism env when `deterministic`; pass scheduler/warmup/
     weight-decay/grad-clip/optimizer/precision/packing/grad-accum/loss-masking
     to `hf_train`.
   - MoE-aware LoRA target modules; aux/z-loss on when experts adapted.
   - Optional replay slice (`replay_ratio`) + KL anchor (`anchor_kl_beta`).
   - Wire `self.tracker.log_params/log_metric/log_artifact` per stage.
   - Checkpoint selection (`best_downstream`/`ugcs`); persist selected digest;
     set `best_metric`.
3. **DPO stage** *(new)*: `run_dpo` using `dpo.build_preference_pairs` +
   `DPOConfig(beta=0.1, lr ~5e-6)` after SFT; on-policy synthetic pairs;
   refusal-floor hard stop.
4. **Eval candidate** (`run_eval_candidate`):
   - Run `eval_compare(candidate, base, eval_items)`; require non-overlapping
     bootstrap CI **and** McNemar significance **and** refusal-rate ≥ base
     before `promoted=True`; record `compare_digest`.
5. **Card stage** *(new)* `run_card`:
   - Call `modelcard.validate(ModelCard(...))` with real eval + contamination +
     provenance evidence; `mrm.compile_dossier(card, activities)` with
     `evidence_status` mapped from receipt verification; seal
     `model_card.json` + `mrm_dossier.json`.
6. **DISTILL stage** *(P2-20, plan)*: GKD/MOPD on-policy distillation from the
   K3-LoRA teacher to a smaller servable student; reverse-KL on
   student-generated samples; teacher receipt required.
7. **Acceptance tests:** `tests/fx1/test_pipeline_gates.py` — quality gate fails
   closed without eval items; promotion blocked on overlapping CI; refusal floor
   enforced; card stage emits a valid sealed card; DPO stage runs and records a
   receipt; tracker called per stage.

### 5.4 Sequencing

| Phase | Contents | Depends on | Risk |
|---|---|---|---|
| **A** | Config hygiene (§5.1): `extra="forbid"`, new fields, stage gates | — | Low (additive + validators) |
| **B** | Receipt v2 (§5.2) | A (config in params) | Medium (schema parity with `receipt_v2`) |
| **C** | Quality-gate + contamination fail-closed (§5.3.1) | A | Medium (LSH dep, eval-item registry) |
| **D** | Train wiring: seed/scheduler/MoE/replay/tracker/checkpoints (§5.3.2) | A, C | Medium |
| **E** | Promotion gate + card stage (§5.3.4–5) | B, D | Low (uses existing `compare`/`modelcard`/`mrm`) |
| **F** | DPO stage (§5.3.3) | D | Low |
| **G** | DISTILL/GKD-MOPD, µP transfer gate, SemDeDup, IFD/DEITA (§2, P2) | E, F | High (research) |

Phases A–F are engineering with clear acceptance tests; G is research and should
land as `(proposed)` with SYNTHETIC correctness tests before any K3 claim.

---

## 6. Honesty & scope

- This document is **research notes + an adoption plan**. It changes no code and
  makes no market-performance claim.
- All candidate-promotion evidence is **proper scores** (eval-suite accuracy
  with paired-bootstrap CIs and McNemar, refusal/honesty-bait rate,
  contamination probes). **No Sharpe/Sortino/Calmar/P&L/NAV** appears as a
  promotion criterion or headline anywhere in the plan.
- Synthetic corpora and synthetic replay/distillation data remain labeled
  `SYNTHETIC` and are correctness tests, never market evidence.
- No live-trading capability is implied; `live_pnl_claim=false` and
  `research_only=true` are hard invariants on every training receipt produced by
  the adopted plan, mirroring `modelcard.py` and `receipt_v2.py`.
- Items marked *(plan)* / *(proposed)* are not shipped behavior; the doc↔code
  drift in §3 is itself flagged as a P0 honesty exposure to be corrected.

---

## 7. References

**Base model.** Kimi Team. *Kimi K3 Technical Report* (2026). 2.8T MoE, 104B
active, KDA + Gated MLA + AttnRes, Stable LatentMoE (896 experts/16 active),
Quantile Balancing, native MXFP4/MXFP8, SFT→RL→MOPD post-training, rephrasing +
exact/fuzzy dedup data recipe. (Local fetch: Kimi-K3 main + paper HTML.)

**PEFT.** Hu et al., *LoRA* (2021) arXiv:2106.09685. Dettmers et al., *QLoRA*
(2023) arXiv:2305.14314. Liu et al., *DoRA* (ICML 2024) arXiv:2402.09353.
Kalajdzievski, *rsLoRA / rank-stability* (2023) arXiv:2312.03732. Hayou et al.,
*LoRA+* (ICML 2024) arXiv:2402.12354. Yang et al., *µP / Tensor Programs V*
(2022) arXiv:2203.03466. *MoE-Sieve* (2025–2026), *MixLoRA* (2025).

**Schedules & optimization.** Vaswani et al., *Attention Is All You Need*
(2017) arXiv:1706.03762 (warmup). Loshchilov & Hutter, *AdamW / SGDR cosine*
(2017) arXiv:1608.03983, arXiv:1608.03983. Hägele et al., *Scaling Laws with
WSO* (2024) arXiv:2405.18392. Jordan et al., *Nanotron mixed precision* (2024)
arXiv:2406.07153. Shoeybi/Megatron + DeepSpeed ZeRO (Rajbhandari et al., 2020,
arXiv:1910.02054).

**Data mixture & curriculum.** K3 §4 data recipe. Xie et al., *DoReMi* (2023)
arXiv:2305.10429. Ye et al., *Data Mixing Laws* (2024) arXiv:2403.16952. Zhou
et al., *LIMA* (2023) arXiv:2305.11206. *LIMO* (2025) arXiv:2502.03373. Liu et
al., *DEITA* (2024) arXiv:2312.15685. Li et al., *IFD / Cherry LLM* (2023)
arXiv:2308.12032.

**Dedup & decontamination.** Broder (1997) shingling; Indyk & Motwani (1998)
LSH; Lee et al., *Deduplicating Training Data* (2021) arXiv:2107.06499. Abbas
et al., *SemDeDup* (2023) arXiv:2303.09540. Tirumala et al., *D4 / dedup for
LLMs* (2023). GPT-3 (Brown et al., 2020) + Llama/PaLM 13-gram decontamination
practice.

**Checkpoint selection & merging.** Guo et al., *UGCS* (2025) arXiv:2503.13464.
Miller, *Adding Error Bars to Evals* (2024) arXiv:2411.00640. Wortsman et al.,
*Model Soups* (2022) arXiv:2203.05482; *WiSE-FT* (2022) arXiv:2109.01903. Yu
et al., *DARE* (2024) arXiv:2311.03099.

**Forgetting & alignment.** Kirkpatrick et al., *EWC* (2017). Luo et al.,
*forgetting in SFT* (2023). Rafailov et al., *DPO* (2023) arXiv:2306.05685.
Agarwal et al., *GKD / on-policy distillation* (2024) arXiv:2306.13649. Lu &
Lab, *On-policy distillation* (2025, Thinking Machines). K3 §4.1.3 MOPD.

**Numerical stability.** Micikevicius et al., *Mixed Precision Training* (2018)
arXiv:1710.03740. NVIDIA BF16/TF32 docs. DeepSpeed FP16/BF16 + stochastic
rounding. K3 MXFP4/MXFP8 + Quantile Balancing. router z-loss (PaLM, Chow
et al.).

**Reproducibility.** PyTorch determinism guide (`CUBLAS_WORKSPACE_CONFIG`,
`use_deterministic_algorithms`). Dodge et al., *Random Seeds and LLMs* (2020)
arXiv:2002.00046. Picard, *torch-manual-seed-everything* (2021). Repo:
`src/quant_fund/research/receipt_v2.py`, `docs/RECEIPT_V2.md`,
`src/fx1/modelcard.py`, `src/fx1/eval/compare.py`, `src/fx1/eval/contamination.py`.

---

## Verified doc-drift backlog (confirmed 2026-09-28)

Re-verified by reading the current `src/fx1` tree (read-only pass; no code
changed). **The code has moved since §3/§4 above were written** — several P0/P1
items in this doc are now stale and are corrected here. Status vocabulary
matches the greppable convention now used in `docs/FX1_TRAINING.md`:
`**Status: SHIPPED**` / `PARTIAL` / `PLANNED`. Doc truth-restoration (status
tables + callouts in `FX1_TRAINING.md`, `FX1_ARCHITECTURE.md`, `FX1.md`,
`APPLY.md`) landed with this pass; implementation is the next wave.

### Corrections to this document's earlier claims (audit was stale)

| Earlier claim (§3/§4) | Current verified state |
|---|---|
| "`run_quality_gate` never passes `eval_items` (`pipeline.py:76-77`) — probe dead" | **Stale.** `run_quality_gate(eval_prompts=None)` defaults prompts from `DEFAULT_BANK` user messages (`src/fx1/train/pipeline.py:95-97`) and passes them as `dedup_and_filter(..., eval_prompts=prompts)` (`pipeline.py:98`; `src/fx1/data/quality.py:42-68`). The shingle-containment screen runs on every pipeline invocation. Residual gaps remain (below, G-1). |
| "No promotion gate — nothing compares candidate vs base with `eval_compare`" | **Stale/partial.** `run_eval_candidate` computes `compare_runs` (paired bootstrap CI + McNemar) and writes `comparison.json` (`pipeline.py:152-171`; `src/fx1/eval/compare.py:60-77`). What is still missing is *enforcement* (below, G-2). |
| "`DeepSpeedConfig` hardcodes fp16 fields (`cluster.py:39-44`); FP16 allowed for K3" | **Stale.** `cluster.py` was rewritten: `ClusterSpec.precision` is `pattern="^(bf16|fp8|mxfp4)$"` (`src/fx1/train/cluster.py:23`) — fp16 is structurally unreachable; no `DeepSpeedConfig` class exists anymore; `to_deepspeed_config()` emits only `bf16.enabled` (`cluster.py:43-56`). Residual gap: this is not a *stage* gate on `TrainConfig` (below, G-6). |
| "`tracker` constructed in `Pipeline.__init__` but never used" | **Stale shape, same substance.** `Pipeline.__init__` no longer constructs a tracker (`pipeline.py:61-76`); `Tracker` ships and is exported (`src/fx1/train/tracking.py:17`; `src/fx1/train/__init__.py:9`) but is instantiated **nowhere** in `src/fx1` (grep: only the class definition). Still unwired (below, G-4). |
| "`receipts.py` docstring references `Pipeline.run_card()`" | **Stale.** The docstring no longer mentions `run_card` (`src/fx1/train/receipts.py:1-8`). `run_card` still does not exist (below, G-3). |
| "`TrainingReceipt` … no seed, no `live_pnl_claim`" | **Stale.** The receipt carries `seed`, `git_revision`, `dirty_worktree`, `live_pnl_claim=False`, `research_only=True`, and `verify_training_receipt` fail-closes on both flags (`receipts.py:51-67,110-127`). It is still **not** a `receipt.v2` envelope and is not sealed into `receipts/` (below, G-5). |

### Confirmed gaps (current, with evidence)

- **G-1 — Contamination: 3-probe audit unwired in the training path; screen doesn't fail closed. Priority: P0.**
  Evidence: `run_contamination_audit` (`src/fx1/eval/contamination.py:132-158`)
  is referenced only by the CLI `fx1 contamination-audit` (`src/fx1/cli.py:240-263`);
  `Pipeline` never calls it (grep `src/fx1/train/` → no import of
  `fx1.eval.contamination`). The pipeline's basic screen
  (`src/fx1/data/quality.py:64-68`) silently no-ops when `eval_shingles` is
  empty (`if shingles and eval_shingles:`) rather than raising, and
  Min-K%/rephrased-gap probes never run in-training. Eval items are defaulted
  from `DEFAULT_BANK` (`pipeline.py:95-97`) so the screen does run today, but
  it is a single 8-token-shingle containment check at threshold 0.6, not the
  3-probe audit.
  Requires: call `audit_contamination` in `run_quality_gate` + `run_eval_candidate`;
  raise when the eval bank is empty; record a `contamination_digest` in the receipt.
- **G-2 — Comparison computed but not enforced as a promotion gate. Priority: P0.**
  Evidence: `run_eval_candidate` (`pipeline.py:152-171`) writes
  `comparison.json` and advances unconditionally; `significant_improvement`
  (`compare.py:77`) is never read by the pipeline; the comparison filters
  `kind == "domain"` only (`pipeline.py:162-163`) so general-task regression is
  unchecked; no refusal/honesty-rate floor; no `promoted` field exists anywhere
  in `src/fx1` (grep). Requires: fail-closed gate (`delta_ci_low > 0`, McNemar
  significance, general non-regression, candidate honesty pass, refusal floor
  ≥ base) before advancing past `EVAL_CANDIDATE`; persist the decision +
  `compare_digest` into the receipt.
- **G-3 — `Stage.CARD` has no `run_card`; model card/MRM never pipeline-produced. Priority: P0.**
  Evidence: `Stage.CARD = "card"` (`pipeline.py:35`) is the last enum member;
  `Pipeline` has methods `run_quality_gate/run_eval_base/run_training/run_eval_candidate`
  only (`pipeline.py:87-171`) — advancing *to* CARD is terminal and nothing
  produces card artifacts. `modelcard.py`/`mrm.py` are CLI-reachable only
  (`cli.py:147-160`, `cli.py:312-326`). Requires: `run_card` calling
  `ModelCard` validation + `mrm.compile_dossier` against real eval/comparison/
  contamination evidence, sealed to the run dir.
- **G-4 — Curriculum, dedup-stratification, and experiment tracking unwired. Priority: P1.**
  Evidence: `build_curriculum` (`src/fx1/train/curriculum.py:39-70`) is called
  only from `cli.py:201-203`; `Tracker` (`tracking.py:17-60`) is never
  constructed in `src/fx1`; `SFTExample` (`src/fx1/data/corpus.py:22-31`) has
  no `competence_level`/`quality_weight` fields (those were removed or never
  landed — the audit's mixture-sampler critique now has no field to consume);
  near-dup is a bounded-window pairwise Jaccard (last 500 examples, threshold
  0.9, `quality.py:70-74`), so sub-quadratic but still not MinHashLSH and not
  per-domain. Requires: invoke curriculum in the quality gate +
  record level histogram; construct/log via `Tracker` per stage; MinHashLSH
  for scale.
- **G-5 — Receipt is not `receipt.v2`, not sealed into `receipts/`, and omits eval-candidate/comparison binding. Priority: P1.**
  Evidence: `TrainingReceipt` (`receipts.py:51-67`) hashes config/corpus/split/
  eval_base only — `eval_candidate`/`comparison` hashes are absent because the
  receipt is issued *before* training (`pipeline.py:126-137`); output goes to
  the run dir (`training_receipt.json`), not `receipts/`; `verify_training_receipt`
  (`receipts.py:102-127`) is not registered with `dipcatcher verify-receipt`.
  Requires: §5.2 of this doc (receipt.v2 envelope, corpus-content hash,
  code/env fingerprints, eval binding, seal + verifier registration).
- **G-6 — LoRA attention-only; no stage-aware fail-closed config gates. Priority: P1.**
  Evidence: `LoRAConfig.target_modules = ["q_proj","k_proj","v_proj","o_proj"]`
  (`src/fx1/train/config.py:28-30`) — expert `gate/up/down` + shared-expert
  projections of K3's 896-expert MoE are never adapted by any shipped default.
  `TrainConfig._k3_run_disclosure` (`config.py:57-71`) checks base model,
  `estimated_nodes >= 2`, `preserve_reasoning_content` — it does **not** check
  precision (TrainConfig has no precision field at all), does **not** require
  MoE target modules, and does **not** require a prior PROXY receipt before
  FINAL_K3. No config model in `src/fx1/train/` sets `extra="forbid"` (grep:
  only `src/fx1/forecast/config.py:14,123` does). Mitigant: fp16-on-K3 is
  structurally blocked at the cluster layer (`cluster.py:23`), so the audit's
  "FP16 allowed" framing is now true only in the sense that `TrainConfig` and
  `ClusterSpec` are unconnected (`Pipeline.run_training` passes `TrainConfig`
  to the injected trainer, `pipeline.py:138-142`; nothing derives a
  `ClusterSpec` from it). Requires: §5.1 (MoE target set, α/r validator,
  `extra="forbid"`, stage gates incl. proxy-receipt precondition, and wiring
  `ClusterSpec` generation into the train stage).
- **G-7 — DPO and DISTILL are enum/module-only. Priority: P2.**
  Evidence: `build_preference_pairs` (`src/fx1/train/dpo.py:98-120`) + CLI
  `fx1 dpo-build` exist; `Stage` has no DPO member and no trainer consumes the
  pairs. `LadderStage.DISTILL` (`config.py:21`) has no implementation.
  Requires: §5.3.3 (`run_dpo` stage with refusal-floor hard stop) and §2.6
  GKD/MOPD design for DISTILL.
- **G-8 — Seed/determinism not enforced end-to-end. Priority: P2.**
  Evidence: `TrainConfig` has no `seed`/`deterministic` field; seeds are
  hardcoded at call sites (`pipeline.py:126` `seed=17`, `quality.py:117`
  `seed=17`, `cluster.py:29` `seed: int = 17`, `compare.py:34/61` `seed=7`);
  no `torch.manual_seed`/`set_seed`/`CUBLAS_WORKSPACE_CONFIG` anywhere in
  `src/fx1` (grep). Requires: `seed`+`deterministic` on `TrainConfig`, passed
  through the trainer seam and recorded in the receipt.

### Suggested sequencing for the next wave

P0 first: G-1 → G-2 → G-3 (they compose: the promotion gate consumes the
contamination digest and the card stage consumes the gate decision). Then
G-5/G-6 (config + receipt.v2, phases A–B of §5.4), then G-4, G-7, G-8.
Doc truth-restoration is already landed, so no shipped doc asserts these as
current behavior in the meantime.
