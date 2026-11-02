# Workflow files are epoch-chained

`cursor-ci.yml` launches a Cursor cloud agent for heavy checks on pull
requests and pushes to `main`. The other heavy workflows skip those events
unless the `USE_CURSOR_CI` repository variable is exactly `false`. Setup,
including the `CURSOR_API_KEY` secret, is `docs/CURSOR_CI.md`.



Every `*.yml` here is a stamped member of the corpus-epoch integrity chain
(`quality/epoch_heads.json` pins the head). After adding/editing/removing a
workflow, re-stamp: `make stamp-epochs` (or `dipcatcher corpus-epoch
--corpus-dir .github/workflows --glob '*.yml' --out-dir
.github/workflows --heads-pin quality/epoch_heads.json`).
`make evidence-audit` fails on `head_member_digest_drift` /
`head_member_missing_live` until the stamp is committed alongside the
workflow change. The `corpus_epoch_*.json` file in this directory is the
chain head — do not delete it.
