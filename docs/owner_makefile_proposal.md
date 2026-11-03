# Proposed Makefile diff for the owner

`Makefile` is a crown jewel. This branch does not edit it. `Makefile` also
does not include `deploy/*.mk`, so the throwaway ceremony has no make
target. Run it with:

```bash
uv run python scripts/owner_rehearsal.py
```

`OWNER_STEPS.md` step 7 calls `dipcatcher quorum-sign` directly. The
checkpoint that follows is time-stamped with `dipcatcher anchor-timestamp`,
not with `make anchor-checkpoint`. On this tree `make sign-pins` runs
`dipcatcher sign-pins`, and a committed quorum registry rejects that
lone-key v1 signature. `make anchor-checkpoint` runs
`dipcatcher checkpoint --anchor`, which writes a second checkpoint. The
spine then fails because that predecessor is unwitnessed.

Apply the diff below, then restamp and re-sign, before pointing
`OWNER_STEPS.md` back at the two make targets. The diff does not add an
`owner-rehearsal` target.

```diff
--- a/Makefile
+++ b/Makefile
@@ -367,8 +367,12 @@ stamp-epochs: ## Re-stamp all corpus-epoch chains + head pin after touching any
 	  uv run dipcatcher corpus-epoch --corpus-dir "$$spec" --glob '*' --out-dir "$$spec" --heads-pin quality/epoch_heads.json || exit 1; \
 	done
 
-sign-pins: ## Ed25519-sign the integrity pins (needs GATE_SIGNING_KEY or --key-file); run LAST, after stamp-epochs
-	uv run dipcatcher sign-pins
+sign-pins: ## Sign the pin manifest after stamp-epochs. A quorum registry needs quorum-sign; v1 fails closed.
+	@if [ -f quality/gate_quorum.json ]; then \
+		GATE_QUORUM_KEYS="$${GATE_QUORUM_KEYS:-$${GATE_SIGNING_KEY:-}}" uv run dipcatcher quorum-sign; \
+	else \
+		uv run dipcatcher sign-pins; \
+	fi
 
 anchor-pins: ## RFC 3161 timestamp-anchor both pin files via FreeTSA (network); run after sign-pins, at quiet points only — every stamp-epochs stales the anchors
 	uv run dipcatcher anchor-timestamp --file quality/epoch_heads.json
@@ -377,8 +381,8 @@
 checkpoint: ## Sign the pin state into quality/checkpoint.json (needs GATE_SIGNING_KEY); run LAST — it binds the current signature
 	uv run dipcatcher checkpoint
 
-anchor-checkpoint: ## RFC 3161-anchor the checkpoint (network); one token time-binds the whole pin state
-	uv run dipcatcher checkpoint --anchor
+anchor-checkpoint: ## RFC 3161-anchor the existing checkpoint (network); does not mint a second one
+	uv run dipcatcher anchor-timestamp --file quality/checkpoint.json
 
 witness-checkpoint: ## Witness the checkpoint into the public Rekor transparency log (network; needs WITNESS_SIGNING_KEY); commits a self-verifying proof under quality/witness/
 	uv run dipcatcher witness-checkpoint
```
