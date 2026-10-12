# fx-1 Architecture — the map an auditor reads first

## The inversion in one paragraph

fx-1 (`src/fx1`) is an in-tree sub-project: corpus construction, an eval
bank, training plumbing, and a training plan. No trained checkpoint is in
this repository. dipcatcher is the harness: the chat-first CLI that
orchestrates fx1 and fx1-lite ([the dipcatcher CLI](DIP_CONCIERGE.md)), over
the data engine, evaluation bench, and verification layer. The package is organized so a future training
run would inherit its integrity from the harness gates, then expose that
integrity through four specified moves (masked eval, attested inference,
corpus ledger, MRM dossier). This page describes that plan and the code
layout.

## Layer diagram

```
                         ┌─────────────────────────────┐
                         │  SERVING  (fx1.serve)        │
                         │  hosted K3 · local seam      │
                         │  release signing · TEE/zkML  │
                         │  cited_complete (honesty+    │
                         │  provenance at inference)    │
                         └──────────────┬───────────────┘
                                        │ model cards + ship gate
┌──────────────────────────────────────────────────────────────────┐
│ TRAINING  (fx1.train)                                            │
│  staged pipeline: data→quality→eval_base→train→eval_cand→card    │
│  LoRA ladder · DPO pairs · cluster specs · immutable receipts    │
│  curriculum · experiment tracking                                │
└──────────────┬───────────────────────────────────┬───────────────┘
               │ corpus                            │ scores
┌──────────────▼───────────────┐   ┌───────────────▼───────────────┐
│ DATA  (fx1.data)             │   │ EVAL  (fx1.eval)              │
│  receipts·notebooks·ledgers· │   │  task bank · red team ·       │
│  traces → SFT corpus         │   │  masked twins + memory gap ·  │
│  quality gates · frozen      │   │  time partitions ·            │
│  splits · HASH-CHAINED LEDGER│   │  contamination audit · stats  │
└──────────────┬───────────────┘   └───────────────┬───────────────┘
               │                                   │
        ┌──────▼───────────────────────────────────▼──────┐
        │ HARNESS  (fx1.harness → dipcatcher / quant_fund) │
        │  32 registered lab commands, 4 roles, fail-closed│
        │  benches (proper scores) · verify-research ·     │
        │  honesty gates (FORBIDDEN metrics) · PIT data    │
        └──────────────────────────────────────────────────┘
```

## Interactive layer

The chat is `fx1.interactive`, launched by `dipcatcher` (`dipcatcher_cli`)
or by `fxi`. Plain text is a turn with fx1 or fx1-lite. Slash commands
and `dipcatcher exec` are the other doors. The model may run a lab command
only when it is one of the 32 entries in `fx1.harness.HARNESS_REGISTRY`
(verification, data engine, evaluation, model training). `/diagnostics`
in this tree reported `harness: commands: 32`. The lab CLI that
`dipcatcher <name>` forwards to is larger; unqualified names are listed
by `dipcatcher capabilities list` and are not executed by the qualified
planner.

Operator detail, exit codes, and the honesty gate are in
[the dipcatcher CLI guide](DIP_CONCIERGE.md). That guide's model checks
used a loopback stub. They are not a live-model or release claim.

## Trust flow (what an auditor should trace)

No step below has produced a checkpoint in this repository. The sequence is
the plan the package encodes.

1. **Corpus example** → carries `receipt_sha256` of a harness artifact that
   passed `verify-research`; recorded in the hash-chained ledger.
2. **Training run** → blocked until quality gates, contamination scan, frozen
   split, and a passing base eval exist; emits an immutable receipt.
   `build_training_manifest` writes that receipt. It does not launch training.
3. **Checkpoint** → none is in the tree. The ship gate would require a model
   card whose eval deltas beat the base statistically (bootstrap CI +
   McNemar) and whose honesty gate passed natively, plus a signed release
   and optional TEE/zkML attestation.
4. **Served answer** → the serve path is specified to validate honesty and
   attach provenance. Refusal behavior is specified as DPO pairs and
   red-team cases. Local generation is unimplemented.
5. **Regulatory dossier** → `fx1 mrm` compiles the available evidence into the
   five-activity structure.

## Invariants (all enforced in tests)

| Invariant | Enforcement |
|---|---|
| No forbidden headline metrics as research results | `fx1.honesty` + lab catalog gates + DPO + red team |
| No live-performance claims | receipt eligibility + ledger scan + card validator |
| No unsigned serving (when keyed) | `LocalFx1Backend` signature check |
| No training without eval + receipt | pipeline stage gates |
| No corpus without provenance | corpus builder + quality gates + ledger |
| No eval leakage | contamination audit + masked twins + time partitions |

## Where the boundaries honestly are

- GPU trainer binding is an injectable seam (needs a cluster).
- TEE crypto verification delegates to platform SDKs at deployment.
- zkML tier is operator-manifest scaffolding pending prover maturity.
- Post-cutoff eval grows as new harness artifacts accumulate.
