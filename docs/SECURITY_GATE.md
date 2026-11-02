# Security gate

`make -f deploy/security.mk security-gate` runs `scripts/security_gate.py` and writes `.tmp_security/summary.json`. That directory is gitignored. The scorecard row is in `docs/ULTRA_PROD_READINESS.md` (row 19). The checker is:

```bash
uv run python scripts/check_security_gate_summary.py
```

A passing summary has schema `dipcatcher.security_gate/1`, `ok` true, `dirty` false, `new_findings` empty, and `head` equal to `git rev-parse HEAD`. `tools` has an `ok` record for bandit, gitleaks, hadolint, osv-scanner, pip-audit, service, trivy, and zap. Tool versions and digests are pinned in `deploy/security/tools.lock.json`. Accepted ZAP and hadolint ids, and the unfixed Trivy CVEs, live in `deploy/security/accepted-findings.json` and `deploy/security/trivy-allow.json` and expire on 2027-02-02.

`cursor-ci.yml` runs the same make target with `contents: read` on `instabox-large`. The other security workflows print that make line and keep `runs-on: instabox-large`.

## Lab `GET /ready` returns 503

`quant_fund.api.app.readiness` calls `quant_fund.pipeline.doctor.doctor`. The probe is ready only when the data manifest is `ok`, the research receipt is `ok`, core imports are `ok`, the raw/bronze/silver/gold/metadata directories are `ok`, and live mode is not `REJECTED`. On an empty lake the manifest and the receipt are `missing`, so the response is 503 `{"status": "not_ready", ...}`. The body drops `version` and `python_package`. That 503 is the fail-closed signal. It is what ZAP plugin 100000 reported for lab `GET /ready` with authentication on.

`deploy/security/zap-rules.tsv` marks plugin 100000 `OUTOFSCOPE` when the alert URL is `/ready`. The expression is matched from the start of the URL, the same way `zap_common.is_in_scope` does. A 500 on any other URL still alerts. `scripts/security_gate.py` copies the file into the ZAP work directory and passes `-c zap-rules.tsv` to `zap-baseline.py`. The rule is covered by `tests/unit/security/test_ready_not_ready.py`.

## Scanner rate limit

A client that arrives through the published Docker port is not a loopback peer, so the default bucket of 10 requests/second applies. Active SQL rule 40018 was reporting that 429, which hides the real response. A rescan sets both of these for the test process only:

```text
QUANT_API_RATE_PER_SEC=off
RESEARCH_API_RATE_PER_SEC=off
```

`deploy/compose.hardened.yml` does not pass those variables into the container. A rescan uses an uncommitted compose override. `0`, `off`, `false`, `no`, or an empty value disables the limit. Production keeps the default of 10 requests/second. `.env.example` is hash-pinned and does not list them. `docs/THREAT_MODEL.md` is the operator description.

The rescan for this slice sent 40 keyed requests to each service after the override. Every one was HTTP 200. ZAP's API scan then ran with active rules outside the SQL family set to IGNORE. Rules 40018, 40019, 40020, 40021, 40022, and 40027 passed on the lab and the research API, in the unauthenticated scan and in the keyed scan. The raw report's only HTTP 503 is lab `GET /ready`. The pinned ZAP 2.17.0 catalog has no active rule 40024. No handler change came out of that rescan.

## Authorization

`QUANT_API_KEY` and `RESEARCH_API_KEY` are the admin keys. `QUANT_API_READ_KEY` and `RESEARCH_API_READ_KEY` are the read keys. Safe methods are GET, HEAD, and OPTIONS. The matrix in `tests/unit/security/test_authz_matrix.py` fails when a new route is missing. `.env.example` does not gain the read-key names.

## Owner steps this gate does not perform

- Bump the crown `uv.lock` virtualenv pin past 21.14.12 for GHSA-cx6f-vjhg-g3cc.
- Change the crown `Dockerfile` numeric `USER` and the JSON `HEALTHCHECK`.
- Restamp epochs and re-sign `gate_pins.sig`.
