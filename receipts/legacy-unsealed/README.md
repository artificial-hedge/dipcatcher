# legacy-unsealed/

Artifacts committed here predate the `receipt_sha256` envelope convention
(`receipt.v2` and the sealed `*.v1` writers). They are retained for
provenance — content is unchanged, git history intact — but **they cannot be
re-verified**: no top-level seal exists to check, and the schema-specific
honesty verifiers (`verify-research`) target the research-notebook schema,
not these bench-result formats.

Quarantine is itself pinned, not assumed, and it is enforced by the
production gates, not only by tests: `quality/legacy_quarantine.json`
(`legacy_quarantine.v2`) records each file's sha256, its exact
`verify-receipt` error set, a category and a reason. `suite-health --strict`
(the CI `evidence-audit` gate), `verify-all` and `admit-batch` all read it.
They verify this directory instead of skipping it: an unregistered file here,
drifted bytes, a drifted error set, a registry entry whose file is gone, or a
file that now verifies cleanly all fail strict. `suite-health` never counts a
quarantined artifact as verified and never harvests it into pooled evidence.
The lattice and `corpus_inference` still read top-level quarantined receipts,
a known pre-existing gap.

The registry is listed as a crown jewel, but jewel enforcement is not yet
effective: `make evidence-audit` stops at inherited chain drift before
`crown-jewels --check`, and the signed pins need `GATE_SIGNING_KEY` (see
`docs/EVIDENCE_INTEGRITY.md`, "Residual boundaries"). Until the key holder
re-stamps and re-signs, review of this file is the control.

Four of the seven additionally fail `forbidden_metric_keys`: they headline
Sharpe-style metrics from before the proper-scores honesty contract.
`basis_reversion_screen`, `dip_bench_crypto` and `fast_replay_p42_conformance`
fail only the seal check.

The same registry also quarantines retained receipts *in place* at their
original paths when moving them would break an anchor or other binding. The
preregistration documents and their RFC 3161-anchored `prereg_seal.v1`
sidecars sit at `receipts/` root for that reason.

If any of these experiments are re-run under a sealed writer, the new receipt
belongs in `receipts/` proper. Never re-seal, edit, or move a quarantined
artifact to make a gate pass.
