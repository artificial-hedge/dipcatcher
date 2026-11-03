#!/usr/bin/env bash
# Owner ceremony for the prod-readiness rows a key holder can clear.
# Commands only. Supply key material through the environment or files.
# Do not paste private keys into this file, the shell history, or the commit.
#
# Makefile does not include deploy/*.mk, and Makefile is a crown jewel, so
# this tree has no owner-rehearsal target. Prove the ceremony with:
#   uv run python scripts/owner_rehearsal.py
# Step 7 below signs the pins directly. The stock sign and checkpoint-anchor
# targets do not match this ceremony. The edits that would make them match
# are in docs/owner_makefile_proposal.md. Apply that diff before switching
# this script back to those targets.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Required environment (never commit the values):
#   GATE_SIGNING_KEY       outgoing Ed25519 seed, 64 hex characters.
#                          It must match the committed gate public key.
#   GATE_SIGNING_KEY_NEW   incoming Ed25519 seed, 64 hex characters.
#   FX1_PUBLISHED_CARD     model card whose version or base_model names fx-1.
#   FX1_PUBLISHED_WEIGHTS  weight file beside that card
#                          (.safetensors, .bin, .gguf, .pt, or .ckpt).
#   WITNESS_SIGNING_KEY    ECDSA P-256 PEM text or path. It must match the
#                          committed witness public key.
# Optional, when the quorum is not the 1-of-1 gate key this tree ships:
#   QUORUM_NEXT_REGISTRY     gate_quorum.v1 file. Its members must include
#                            the incoming gate key, and the key passed to
#                            sign-pins and checkpoint must meet the threshold.
#   QUORUM_MEMBER_KEY_FILE   outgoing member material, one '<seed>:<pub>' line.

# fx1_weights
# This repo does not generate published fx-1 weights. artifacts/fx1_tiny_lm
# and Kronos do not count. known_limits must not contain
# "not an fx-1 release candidate".
install -d models/fx1
cp "$FX1_PUBLISHED_CARD" models/fx1/modelcard.json
cp "$FX1_PUBLISHED_WEIGHTS" models/fx1/weights.safetensors

# key_rotation. GATE_SIGNING_KEY is still the outgoing seed.
# The rotation record is a quality/*.json epoch member, so it lands before stamp.
make rotate-key

# The incoming public key is a crown jewel. Install it before the jewel pin.
# Later signing must leave these exact bytes in place.
old_pub="$(uv run --no-sync python scripts/owner_rehearsal.py --public-hex GATE_SIGNING_KEY)"
new_pub="$(uv run --no-sync python scripts/owner_rehearsal.py --public-hex GATE_SIGNING_KEY_NEW)"
printf '%s\n' "$new_pub" > quality/gate_signing.pub

work="$(mktemp -d)"
chmod 700 "$work"
if [ -z "${QUORUM_MEMBER_KEY_FILE:-}" ]; then
  QUORUM_MEMBER_KEY_FILE="$work/outgoing.key"
  printf '%s:%s\n' "$GATE_SIGNING_KEY" "$old_pub" > "$QUORUM_MEMBER_KEY_FILE"
  chmod 600 "$QUORUM_MEMBER_KEY_FILE"
fi
if [ -z "${QUORUM_NEXT_REGISTRY:-}" ]; then
  QUORUM_NEXT_REGISTRY="$work/next_quorum.json"
  GATE_SIGNING_PUBKEY_NEW="$new_pub" \
    uv run --no-sync python scripts/owner_rehearsal.py \
    --quorum-registry "$QUORUM_NEXT_REGISTRY"
fi

# sign-pins, quorum-sign, and checkpoint read <seed>:<pub>, not a bare seed.
export GATE_SIGNING_KEY="${GATE_SIGNING_KEY_NEW}:${new_pub}"

# quorum_rotations. Rewrites quality/gate_quorum.json, an epoch member.
uv run --no-sync dipcatcher quorum-rotate \
  --registry "$QUORUM_NEXT_REGISTRY" \
  --key-file "$QUORUM_MEMBER_KEY_FILE" \
  --reason "scheduled quorum rotation"

# crown_jewels. After the new public key is on disk, and before stamp.
uv run --no-sync dipcatcher crown-jewels --write

# epoch_chains. After the jewel pin, the rotation record, and the quorum registry.
# Receipts are append-only. receipts/fx1_retrieval_audit.json was edited after
# its stamp; stamping that digest records member_mutated and the chain fails.
# Put every drifted receipts member back to the digest the head epoch already
# names. The bytes come from this checkout's git history. A clean chain prints
# nothing. This does not sign, and it does not invent a receipt.
uv run --no-sync python scripts/owner_rehearsal.py --restore-stamped-receipts
make stamp-epochs

# 7. pin_signatures. A committed quorum registry rejects a lone-key v1
# signature, and make sign-pins on this tree runs that v1 command.
# Call quorum-sign directly until docs/owner_makefile_proposal.md is applied.
# The pair above must be a member of the registry just installed.
GATE_QUORUM_KEYS="${GATE_QUORUM_KEYS:-$GATE_SIGNING_KEY}" \
  uv run --no-sync dipcatcher quorum-sign

# timestamp_anchors for the two pin files. Later steps do not rewrite them.
make anchor-pins

# One checkpoint. make anchor-checkpoint on this tree runs
# `dipcatcher checkpoint --anchor`, which writes a second checkpoint.
# That predecessor is unwitnessed and the spine fails. Time-stamp the
# checkpoint just written until docs/owner_makefile_proposal.md is applied.
make checkpoint
uv run --no-sync dipcatcher anchor-timestamp --file quality/checkpoint.json

# The previous head is already in the witness set. This head has to join it.
make witness-checkpoint

# release_tag. The scorecard counts local v* tags. Pushing the tag runs
# release.yml. OWNER_REHEARSAL=1 keeps the tag local.
git add -u
git add quality/rotation_*.json quality/quorum_rotations models/fx1
git commit -m "chore: restamp and re-sign the integrity pins"
git tag -a v0.4.0 -m "fx-1 harness 0.4.0"
if [ "${OWNER_REHEARSAL:-}" = "1" ]; then
  echo "owner rehearsal: local tag v0.4.0; not pushing"
else
  git push origin v0.4.0
fi

# Rows this script does not clear.
# pr_test_suite reads an external cursor-ci commit status, or a signed
# cursor_ci_result.v1 file, for the exact HEAD sha. It does not run the
# suite and it does not write that attestation. See docs/CURSOR_CI.md.
# windows_fleet, on the Windows host after the Defender exclusion:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fleet_spawn.ps1 -Manifest jobs.json
# live_condition_1 through live_condition_5 stay needs-owner until the
# evidence in docs/GO_LIVE_CHECKLIST.md exists. live_readiness_report
# stays blocked until that evidence is observed outside this checkout.
