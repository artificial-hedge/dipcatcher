# Application security gate.
#
# The root Makefile is a crown jewel, so this file is the target that runs:
#
#   make -f deploy/security.mk security-gate
#
# The command downloads each pinned tool, checks the sha256 in
# deploy/security/tools.lock.json, and writes .tmp_security/summary.json.
# The exit status is non-zero when any finding is outside
# deploy/security/accepted-findings.json or the trivy allowlist. Unfixable
# HIGH and CRITICAL image CVEs need an expiry date in
# deploy/security/trivy-allow.json. A fixable HIGH or CRITICAL CVE fails
# even when the id is listed.

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

.PHONY: security-gate

security-gate: ## Bandit, pip-audit, osv-scanner, gitleaks, hadolint, trivy, ZAP
	uv run python $(ROOT)/scripts/security_gate.py --root $(ROOT)
