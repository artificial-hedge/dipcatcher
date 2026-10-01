# Trusted timestamp anchors

Committed RFC 3161 tokens (`*.tsr`) issued by [FreeTSA](https://freetsa.org)
over the sha256 of integrity-critical files. `anchors.json` maps each token
to `{target, sha256}` — the digest the token commits to, frozen at stamp
time.

- `verify-repo` reports the `timestamp_anchors` gate: `anchored`, per-target
  `fresh` (token matches the file's *current* bytes), and `chain_verified`
  (openssl `ts -verify` against `freetsa_cacert.pem`/`freetsa_tsa.crt`,
  both committed so verification is offline and reproducible).
- Anchors go **stale by design**: `stamp-epochs` advances the pins past the
  last anchor. A stale anchor remains valid proof that the earlier pin
  state existed by the token's time. Re-anchor at quiet points — after a
  merge lands, at release — never mid-churn:

```bash
dipcatcher anchor-timestamp --file quality/epoch_heads.json
dipcatcher anchor-timestamp --file quality/crown_jewels.json
```

The request sends only the 32-byte digest — file contents never leave the
machine. The imprint check fails closed: a token that doesn't commit to the
requested digest is never registered.

Trust note: this anchors *existence by time T*. It does not certify the
content was honest — that is the seals', chains', and review's job.
