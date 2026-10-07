# legacy-unsealed/

Artifacts committed here predate the `receipt_sha256` envelope convention
(`receipt.v2` and the sealed `*.v1` writers). They are retained for
provenance — content is unchanged, git history intact — but **they cannot be
re-verified**: no top-level seal exists to check, and the schema-specific
honesty verifiers (`verify-research`) target the research-notebook schema,
not these bench-result formats.

Because of that, this directory is deliberately **outside** the
`receipts/*.json` glob used by `receipts-reverify` and `verify-all`: the
sealed set must be end-to-end re-verifiable, fail-closed. If any of these
experiments are re-run under a sealed writer, the new receipt belongs in
`receipts/` proper.

Quarantine is itself pinned, not assumed: `quality/legacy_quarantine.json`
records each file's sha256 plus its exact `verify-receipt` error set, and
`tests/unit/research/test_legacy_quarantine.py` fails if either drifts — a
retired artifact cannot be silently edited, and a file that ever verifies
cleanly does not belong here. Four of the seven additionally fail
`forbidden_metric_keys` (they headline Sharpe-style metrics from before the
proper-scores honesty contract); `basis_reversion_screen`, `dip_bench_crypto`,
and `fast_replay_p42_conformance` fail only the seal check.

Two later arrivals (`basis_carry_dd7705fc0f2f1c25`,
`crossvenue_basis_3f4ff76f517655a7`) are sealed and contract-clean but
declare non-synthetic `dataset_sha256` bindings (`kraken`, `kraken+okx`)
whose tapes were never pinned by a committed `tape_manifest.v1` — the
tape-binding ratchet fails them `tape_manifest_unknown`, fail-closed by
design. They return to `receipts/` once the underlying tapes are pinned
(`dipcatcher tape-pin … --eval-stream`) and their digests attestable.
