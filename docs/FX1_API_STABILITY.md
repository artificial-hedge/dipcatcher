# fx-1 API Stability & Versioning Policy

An auditor asks "what can I rely on?" This is the answer.

## Stable (semver-guaranteed within a major version)

| Surface | Guarantee |
|---|---|
| `fx1.harness.Harness.run/list_commands` | registered commands only fail-closed; registry grows, never silently changes semantics |
| `fx1.data.build_corpus/build_full_corpus` | output schema (`SFTExample`) stable; new fields are additive |
| `fx1.eval.run_suite/EvalTask` | task schema stable; bank contents may grow |
| `fx1.honesty.validate_fx1_output` | fail-closed contract; rules may tighten, never loosen |
| `fx1.train.receipts.issue_receipt/verify_training_receipt` | receipt schema versioned; old receipts remain verifiable |
| `fx1.serve.signing.sign_release/verify_release` | manifest + detached-signature format stable |
| `fx1.mrm.compile_dossier` | five-activity structure fixed; sections additive |
| CLI commands (`fx1 *`) | flags additive; existing flags never change meaning |

## Unstable (may change with minor versions)

- Internal helper functions (underscore-prefixed or undocumented).
- Prompt text in `fx1/prompts/` (behavioral contract, tested, but wording evolves).
- Eval bank task *contents* (bank grows; scores are always reported per-version).

## Versioning

`fx1.__version__` follows semver. A future checkpoint would be versioned
independently as `fx-1.vX.Y` with a model card. No trained checkpoint is in
this repository. Every receipt, ledger entry, and dossier embeds the
producing package version implicitly through its hashes.

**`fx1.__version__` is the version of the harness API surface described in
this document. It is not a model release number.** As of 2026-10-08
`fx1.__version__` is `0.4.0` and `git tag -l 'v*' | wc -l` returns **0** — no
release tag has ever been cut, so no release of this API has been published.
(`git tag | wc -l` returns `4`, but all four are
`attic/receipt-provenance/*` provenance anchors from 2026-09-27/28, not
releases.) A model version would be `fx-1.vX.Y` on a model card, per the
paragraph above, and no such version exists.

## Deprecation

No silent removals: any breaking change ships behind a new function/flag with
the old one emitting a `DeprecationWarning` for at least one minor version.
