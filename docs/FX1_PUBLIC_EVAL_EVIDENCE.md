# Public fx-1 evaluation evidence contract

**Status: no public model is approved.** The serving trust registry is empty,
and both `fx1` and `fx1-lite` remain unavailable even if a release signature
and model card are otherwise valid. The case checker in
`fx1.eval.evidence_contract` verifies artifact consistency offline. Its result
does not authorize public serving or prove that the recorded answers came
from a model.

## Owner inputs required before an evaluation run

| Input | Required independent proof |
|---|---|
| Held-out bank | An evaluator-side custodian holds the exact `fx1_heldout_bank.v1` JSON bytes outside training and model-development access. A prior reviewed trust-registry revision pins their SHA-256 digest, bank custody, task counts and scoring protocol. The committed `DEFAULT_BANK` is ineligible as a held-out bank. |
| Bank timestamp | An RFC 3161 token for the exact bank manifest, checked with `verify_prereg_timestamp`, a separately approved TSA CA fingerprint, TSA leaf and pinned OpenSSL executable. Its signed `genTime` must precede the run-start token. A timestamp proves byte existence, not secrecy. |
| Evaluator identity | A public Ed25519 key enrolled before the run, controlled by an independent evaluator and distinct from the release signing key. Record the custodian's approval and access controls outside the model team's release bundle. |
| Base and candidate | Actual checkpoint SHA-256 digests and tokenizer/inference configuration bytes. The candidate digest must equal the signed release inventory's `weights.safetensors` digest; the base digest must equal `ModelCard.base_checkpoint_sha256`. |
| Start challenge | Exact `fx1_eval_run_start.v1` bytes with a fresh 256-bit nonce, bank digest, model alias, both weight digests, configuration/tokenizer digests and evaluator key. An independent RFC 3161 token must anchor this challenge before the first model call. Preserve independently reviewable first-call timing and evaluator custody records; the evaluator's self-reported start time is insufficient. |

The bank must contain at least 10 honesty, 30 domain and 30 general tasks, with
unique ordered names, actual user prompts and nonvacuous scoring criteria.
These are structural floors, not proof that the bank is difficult, novel, or
representative. The independent custodian must approve case quality and keep
the prompts withheld until evaluation is complete.

## Evaluator output and release binding

The evaluator records one exact ordered base/candidate raw response pair per
bank task in `fx1_eval_cases.v1`, labels the data honestly, and signs the
exact JSON bytes with the enrolled evaluator key. The bundle binds the run
start digest, bank digest, alias, both weight digests, configuration digest,
and tokenizer digest. The release inventory must include the bundle, detached
signature, run-start artifact, bank artifact and their immutable hashes.
`ModelCard.eval_delta` must exactly equal the recomputed case rates; the
domain paired bootstrap confidence interval must exclude zero in favor of
the candidate, general tasks must not regress, and the honesty gate must pass
across **all** task kinds. A `SYNTHETIC` bundle is a correctness fixture and
can never qualify a public alias. A `MEASURED_MODEL_EVAL` label alone also
does not establish model provenance.

The offline `audit_signed_case_consistency` checker already enforces bounded
JSON, no duplicate fields, the detached Ed25519 signature, distinct evaluator
and release keys, exact bank/start and weight/config/tokenizer pins, ordered
case coverage, re-scoring with `score_task` and `compare_runs`, paired
significance, and exact card equality. It does **not** validate the TSA
tokens, bank custody, first-call order, or actual model inference. It returns
a consistency audit with no public eligibility verdict.

## Work required before the public gate can open

1. Code-review and pin the independent custodian, bank digest, evaluator key,
   TSA CA and OpenSSL hashes in a trusted registry before the run. A key or
   bank digest supplied in the candidate's release bundle is not a trust root.
2. Verify both RFC 3161 tokens with the pinned trust roots, compare their
   signed `genTime` values, and validate the independent first-call record.
   The timestamp helper does not establish custody or model execution.
3. Verify the Ed25519 release inventory and read-only mount before scoring.
   Reject fixture cards, `SYNTHETIC` or unlabeled output, incomplete or
   untrusted bank coverage, and any difference from the card or actual
   candidate weights. Pin a complete verdict to that exact inventory.
4. Make model listing and direct backend resolution use the same cached
   verdict. Preserve the current hard refusal until these checks, the owner
   artifacts, and independent review are complete.

This contract concerns model capability for a research-serving product. It
does not claim market performance or authorize live trading.
