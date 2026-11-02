#!/usr/bin/env bash
# Owner ceremony for the prod-readiness rows a key holder can clear.
# Commands only. Supply key material through the environment or files.
# Do not paste private keys into this file, the shell history, or the commit.
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
make stamp-epochs

# pin_signatures. A committed quorum registry rejects a lone-key v1 signature.
# make sign-pins dispatches to quorum-sign. The pair above must be a member
# of the registry just installed.
make sign-pins

# timestamp_anchors for the two pin files. Later steps do not rewrite them.
make anchor-pins

# One checkpoint. anchor-checkpoint only time-stamps that file. A second
# checkpoint would name an unwitnessed predecessor and the spine would fail.
make checkpoint
make anchor-checkpoint

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

# Rows the scorecard leaves at needs-owner. These commands do not flip them.
# pr_test_suite: make test
# windows_fleet, on the Windows host after the Defender exclusion:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fleet_spawn.ps1 -Manifest jobs.json
# live_condition_1 through live_condition_5 have no in-repo command that
# marks them satisfied. live_readiness_report stays blocked until the
# external evidence exists.
