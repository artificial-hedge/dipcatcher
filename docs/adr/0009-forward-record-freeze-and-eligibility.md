# ADR-0009: The forward-record freeze is externally anchored, and `window.start` is a label, not an eligibility boundary

## Status

Accepted (2026-10-08). Documents a completed remediation and corrects a
documentation defect that misstated the forward-window eligibility rule.

## Context

Institutional condition 3 requires a non-synthetic forward/shadow record. The
pre-registration for that record is `receipts/forward_record_preregistration_v1.json`,
declared 2026-10-07, carrying `status = NOT_YET_COLLECTED`.

Two defects were found while auditing that record for production readiness.

**Defect 1 — the freeze was self-attested.** The receipt carried two seals:
an embedded `_seal.content_sha256` over the canonical payload, and a sidecar
`*.seal.json` binding the file's SHA-256. Both verified clean. But both are
*local* tamper-evidence, as `src/quant_fund/data/prereg_seal.py` itself states:

> Both are *local* tamper-evidence: they detect accidental or silent edits as
> long as the seal is not itself rewritten by whoever controls all local files.
> Independent timestamping/anchoring of the seal is an operational step; the
> code does not assert it occurred.

`quality/timestamps/anchors.json` covered three integrity pins
(`quality/checkpoint.json`, `quality/crown_jewels.json`,
`quality/epoch_heads.json`) but **not** the pre-registration. The single
artifact gating condition 3 therefore had its freeze date asserted by the
repository itself — exactly the party with an interest in the date.

**Defect 2 — the documented eligibility rule contradicted the sealed rule.**
`docs/REALITY_PREREGISTRATION.md` stated, in the same table:

- *Forward window*: `forward_2026H2`, **start 2026-07-01 (first eligible session on/after)**
- *Splits*: forward = first accepted decision **strictly after the externally recorded freeze**

Those cannot both hold. The receipt resolves it: `splits.forward` reads *"first
accepted decision must occur on a LATER market date than the recorded freeze"*,
and the freeze is **2026-10-07**. So the first eligible forward session is the
first market date after 2026-10-07 — not 2026-07-01.

`window.start = 2026-07-01` is the calendar **span label** of `forward_2026H2`
(second half of 2026) and nothing more. Read as a collection start it would
wrongly imply July–October 2026 closes are eligible forward evidence.

## Decision

1. **Anchor the seal externally.** Run
   `uv run dipcatcher anchor-timestamp --file receipts/forward_record_preregistration_v1.json.seal.json`.
   Only the file's SHA-256 is transmitted to the timestamp authority; never its
   contents. The resulting token is committed at
   `quality/timestamps/receipts__forward_record_preregistration_v1.json.seal.json.tsr`
   with an `anchors.json` entry.

2. **`window.start` is a label.** Treat it as the span of the named window and
   never as an eligibility boundary. The eligibility rule is `splits.forward`,
   resolved against the recorded freeze. Correct every document that restated
   2026-07-01 as "first eligible session on/after".

3. **Do not edit the sealed receipt.** The receipt is immutable evidence. A
   documentation correction is the correct remedy for a labeling defect; mutating
   the receipt to "fix" its label would be the exact failure mode the receipt
   system exists to prevent. No v2 was created, because the receipt's operative
   rule was always sound.

4. **Collection remains open.** `status` stays `NOT_YET_COLLECTED`. Anchoring the
   freeze does not collect anything.

## Consequences

- The freeze instant is now asserted by an external authority. A history rewriter
  cannot backdate it.
- `verify_timestamps('.')` reports `ok=true`, `errors=[]`, and for this target
  `chain_verified` + `fresh=true`. The three integrity pins report
  `fresh=false`, which is the documented "pins moved on" state — the anchors
  remain valid proof of the earlier pinned state and are re-anchored by
  `make anchor-pins` at quiet points.
- Threading the eligibility rule through `docs/REALITY_PREREGISTRATION.md`,
  `docs/EVIDENCE_PROCUREMENT.md`, `docs/INSTITUTIONAL_READINESS.md`,
  `docs/FORWARD_SHADOW_POWER.md`, and `docs/REPO_IMPROVEMENT_PLAN.md` removes a
  live contradiction in the condition-3 evidence path.
- Condition 3 stays **NOT YET COLLECTED**. This ADR advances the *integrity* of the
  forward record, not its existence.

## Related

- `docs/ULTRA_PROD_READINESS.md` §2.2 and §10 (implementation log)
- `docs/REALITY_PREREGISTRATION.md` — the authoritative pre-registration
- ADR-0001 — research receipts are immutable, self-sealing evidence
- [RFC 3161 Trusted Timestamping](https://www.rfc-editor.org/rfc/rfc3161)