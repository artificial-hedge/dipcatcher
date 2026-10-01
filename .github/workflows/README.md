# Workflow files are epoch-chained

Every `*.yml` here is a stamped member of the corpus-epoch integrity chain
(`quality/epoch_heads.json` pins the head). After adding/editing/removing a
workflow, re-stamp: `make stamp-epochs` (or `dipcatcher corpus-epoch
--corpus-dir .github/workflows --glob '*.yml' --out-dir
.github/workflows --heads-pin quality/epoch_heads.json`).
`make evidence-audit` fails on `head_member_digest_drift` /
`head_member_missing_live` until the stamp is committed alongside the
workflow change. The `corpus_epoch_*.json` file in this directory is the
chain head — do not delete it.
