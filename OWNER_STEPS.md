#!/usr/bin/env bash
# Owner ceremony for the prod-readiness rows a key holder can clear.
# Commands only. Supply key material through the environment or files.
# Do not paste private keys into this file, the shell history, or the commit.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# fx1_weights
# This repo does not generate published fx-1 weights. The scorecard passes
# when models/fx1/modelcard.json has version or base_model naming fx-1 and a
# weight file sits beside the card. artifacts/fx1_tiny_lm and Kronos do not count.
# known_limits must not contain "not an fx-1 release candidate".
install -d models/fx1
cp "$FX1_PUBLISHED_CARD" models/fx1/modelcard.json
cp "$FX1_PUBLISHED_WEIGHTS" models/fx1/weights.safetensors

# crown_jewels
uv run dipcatcher crown-jewels --write

# key_rotation
# Export GATE_SIGNING_KEY (current quality/gate_signing.pub) and
# GATE_SIGNING_KEY_NEW (incoming seed) out of band before this command.
make rotate-key
export GATE_SIGNING_KEY="$GATE_SIGNING_KEY_NEW"

# quorum_rotations
# QUORUM_NEXT_REGISTRY is a gate_quorum.v1 file. QUORUM_MEMBER_KEY_FILE is
# an outgoing member key. Repeat --key-file until the current threshold is met.
uv run dipcatcher quorum-rotate \
  --registry "$QUORUM_NEXT_REGISTRY" \
  --key-file "$QUORUM_MEMBER_KEY_FILE" \
  --reason "scheduled quorum rotation"

# epoch_chains
# After the jewel pin, the rotation record, and the quorum registry are on disk.
make stamp-epochs

# pin_signatures
# Refuses to be useful without GATE_SIGNING_KEY: a missing key mints an
# ephemeral signer. The variable above is the new key.
make sign-pins

# timestamp_anchors
make anchor-pins

# checkpoint
make checkpoint

# timestamp_anchors for quality/checkpoint.json
make anchor-checkpoint

# release_tag
# Commit the ceremony, then cut a local vX.Y.Z tag. The scorecard counts
# local tags from `git tag -l 'v*'`. Pushing the tag is what runs release.yml.
git add -u
git add quality/rotation_*.json quality/quorum_rotations models/fx1
git commit -m "chore: restamp and re-sign the integrity pins"
git tag -a v0.4.0 -m "fx-1 harness 0.4.0"
git push origin v0.4.0

# Rows the scorecard leaves at needs-owner. These commands do not flip them.
# pr_test_suite: make test
# windows_fleet, on the Windows host after the Defender exclusion:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fleet_spawn.ps1 -Manifest jobs.json
# live_condition_1 through live_condition_5 have no in-repo command that
# marks them satisfied. live_readiness_report stays blocked until the
# external evidence exists.
