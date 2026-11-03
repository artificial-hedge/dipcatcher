# Owner re-sign checklist

Key-holder procedure for the pins, the lock, the census, and the release tag.
Run the steps in this order. Steps 1–8 change bytes. Step 9 rewrites
`quality/crown_jewels.json` over those bytes. Steps 10–13 stamp, sign, anchor,
and checkpoint. Step 14 anchors an audit log outside the repo. Step 15 cuts
the release tag.

One `make stamp-epochs` at step 10 covers every earlier edit. A stamp in the
middle of steps 1–8 leaves the later bytes outside the new chain, and every
later stamp stales the RFC 3161 anchors from step 12.

Digests below are SHA-256 of the raw file bytes, the same function
`quant_fund.research.crown_jewels.crown_jewel_digests` uses. They were
measured on `a416f16648a0931e9e3161a0d46ae04ea183b91b` except where a step
says the command prints the value. This file, the TiRex smoke edit, the fx1
package-import fix, and the atlas refresh land after that commit. They change
`docs/*`, `tests/*`, and `src/*`, and step 10 includes them. The atlas
refresh drops the generated package edge `fx1` → `fx1.secret_guard`
(cross-package import edges 425 → 424) because `src/fx1/__init__.py` loads
`quant_fund/utils/secret_guard.py` by path. The module edge
`fx1.secret_guard` → `quant_fund.utils.secret_guard` stays. Regenerate the
atlas only when a later import moves.

`GATE_SIGNING_KEY` and `WITNESS_SIGNING_KEY` stay in the environment or a
mode-`0600` file. They are never written into the tree, this file, or a log.

## 1. Copy the security environment block into `.env.example`

`.env.example` is a crown jewel. The pin still records
`3601a96a5af0539c0f7e9358d984be5c9e01cc4d8ac81da6f6e31d6c7b2f0cc7`. The
working file is already
`6dd1359ba4cbd3d7bda595ca33f0307aadb80ff2c6b9f16c9728bcf860c9398b`.

Append the fenced body under "Block for `.env.example`" in
`docs/SECURITY_ENV.md` (the lines inside the `text` fence, from the
`# ── Lab and research API security` heading through `# DIPCATCHER_RATE_STORE=`).
Leave one blank line between the current last line (`# FX1_NEW_PROBE_CONFIG=`)
and that heading. The production values in that block are the ones to commit:
loopback `100` / `1000`, remote `10` / `30`, audit rotation `1048576`, audit
reads `on`, rate backend `memory`. Key material stays commented and empty.

```bash
sha256sum .env.example
```

Expected after that append:
`972de84e5ab89ecf64a4dce4bb648e569a361b9722c1277a368615c9365a1d2c`.

## 2. Update the audit-coverage census

`quality/audit_coverage.json` and `quality/audit_coverage_fx1.json` are
epoch members (`quality/*.json`). Edit them before step 10. Status vocabulary
is `audited`, `partial`, `pending`, `waived`.

### `quality/audit_coverage.json`

Current file: `6c44392e1a0d4a8cc2020d6359043b7d80310dbb7f20672c0e93b44b4d23e462`.

Set these `n_modules` fields and leave every status as it is:

| Directory | Status | Pinned | Actual |
|---|---|---|---|
| `config` | audited | 7 | 8 |
| `utils` | audited | 10 | 12 |
| `api` | audited | 7 | 10 |

The actual sets, non-waived, are:

- `config`: `__init__.py`, `impact.py`, `inherit_audit.py`, `loader.py`, `models.py`, `schema_fingerprint.py`, `validator_fuzz.py`, `yaml_safe.py`
- `utils`: `__init__.py`, `atomicio.py`, `cli_errors.py` (module override `pending`, still counted), `hashing.py`, `logging.py`, `numeric.py`, `outbound.py`, `receipt.py`, `reproducibility.py`, `secret_guard.py`, `seeds.py`, `series.py`
- `api`: `__init__.py`, `admin_audit.py`, `app.py`, `error_shape.py`, `http_controls.py`, `middleware_audit.py`, `param_fuzz.py`, `rate_store.py`, `research_api.py`, `surface_pin.py`

Files without a module override inherit the directory status. After the three
number edits and no other change, the file sha256 is
`00571d349658b30be6c3c0123c9a7197bd3a9ffea399bbbe9b0fea3316dc07fb`.
`test_audited_floor_only_grows` then counts 2499 audited modules (floor 508).

### `quality/audit_coverage_fx1.json`

Current file: `d92e130fc4b76f1902b150df50926c94d6bca368debf0b7e0c99a2d1662a02af`.

`serve.n_modules` is 59 and the directory status is `partial`. Set it to 61.
The two files that make the count 61, and that inherit `partial`, are
`serve/client_wire.py` and `serve/parity_audit_bench.py`.

`src/fx1/cli_audit_bench.py` and `src/fx1/secret_guard.py` have no slash, so
they need their own module rows. `test_every_module_has_a_status` lists both,
and `test_audited_floor_only_grows` raises `KeyError` on `cli_audit_bench.py`
until the row exists. Add:

```json
"cli_audit_bench.py": {
  "status": "pending"
},
"secret_guard.py": {
  "status": "pending"
}
```

Place `cli_audit_bench.py` immediately after the `cli_audit.py` object, and
`secret_guard.py` immediately after the `sbom_audit.py` object, with the same
two-space indent as the other module keys. After the `serve` count and those
two objects, and no other change, the file sha256 is
`d113320d8233043cd2db51d7d9e6d0f431876909b42a715f28856aad94afa0df`.
The audited count is then 123 (floor 65).

```bash
uv run pytest -q -p no:xdist tests/unit/test_audit_coverage.py
```

Expected: the file's four tests pass.

## 3. Lock `virtualenv` 21.14.13 and keep `pyasn1` 0.6.5

`uv.lock` and `pyproject.toml` are crown jewels.

`uv.lock` already pins `pyasn1` 0.6.5 (sdist
`sha256:6b11616a6bcf3a40e635f082bb232fa57b5e0dd7f517bd7ddd69ec3794508a47`,
wheel `sha256:4017b0fce125096b8573b39c989fdb1a4b19cf761e683398184550fea0e66587`).
That release is the patched version for `GHSA-fm9c-5rcf-2q5p`,
`GHSA-mm57-4v3p-rchx`, `GHSA-2qm8-x737-gmw5`, and `GHSA-9m75-vc3g-4x39`.
The gate's pip-audit snapshot records `pyasn1` 0.6.5 and `pyasn1-modules`
0.4.2 with empty vulnerability lists. Keep 0.6.5 when the lock is rewritten.
There is no newer pyasn1 release to pin.

`virtualenv` is locked at 21.14.12 (sdist
`sha256:bbcfb78e36c22ad70c6917c4aecb5e63477d3c138e9f88c946f320e31859c76a`).
`GHSA-cx6f-vjhg-g3cc` (`CVE-2026-115586`) affects `>=21.14.9,<21.14.13` and
is fixed in 21.14.13. `pyproject.toml` still says `virtualenv>=21.14.9`.
The accepted ids `osv:GHSA-cx6f-vjhg-g3cc:virtualenv` and
`pip-audit:GHSA-cx6f-vjhg-g3cc:virtualenv` expire 2027-02-02.

Current `pyproject.toml` sha256 is
`68483824a560eaf795641b9b12b87f1a5db6b6b7067b44e2afbb08be2ec45e4a`
(pin still records `6c4f2a92d811a9be0885326205c1b94f30ef46ab002242234fcb271d0b04e454`).
Change the constraint line to:

```toml
constraint-dependencies = ["urllib3>=2.8.0", "virtualenv>=21.14.13"]
```

That one-line edit on the current file yields
`24adf9529e5e4e358566c31b067c834b968bae735d321e22bb42c1733cc1c0e3`.

Then rewrite the lock and sync it:

```bash
uv lock
uv sync --frozen --all-groups --all-extras
uv run python -c "import importlib.metadata as m; print(m.version('virtualenv')); print(m.version('pyasn1'))"
```

Expected versions: `21.14.13` then `0.6.5`. Confirm the lock still contains
`name = "pyasn1"` / `version = "0.6.5"` and now contains `name = "virtualenv"`
/ `version = "21.14.13"`. `sha256sum uv.lock` after `uv lock` is the digest
step 9 records. The pre-bump lock digest is
`ed8397b8850b5cbf19c3ad535d0ecae7dab8de0cc23a6351f63fd403dc0c86d5`
(pin still records `f4b4f747b433f61730ec4d4cef03ce219a7264d9a8f9e030a8d286f5d76e8627`).

## 4. Dockerfile numeric `USER` and JSON `HEALTHCHECK`

`Dockerfile` matches its crown pin today:
`f367dfc1d9634fccb6c24f264c78b436304a30da148715c3e83a31463799f005`.
`deploy/security/accepted-findings.json` accepts `hadolint:DL3066:Dockerfile:28`
and `hadolint:DL3025:Dockerfile:31` until 2027-02-02 and points at the
supply-chain owner diff. Apply that diff.

Replace both `FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea`
lines with:

```dockerfile
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
```

Keep the `AS builder` and `AS runtime` stage names. Replace the runtime
user and healthcheck:

```dockerfile
COPY --chown=dipcatcher:dipcatcher configs ./configs
RUN python -c "import sys; from pathlib import Path; dest = Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'configs'; dest.symlink_to('/app/configs')"
USER 10001
EXPOSE 8000
# Bind loopback by default. For 0.0.0.0, set QUANT_API_KEY and override CMD host.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://{os.environ.get('HOST', '127.0.0.1')}:{os.environ.get('PORT', '8000')}/health\", timeout=3)"]
CMD ["dipcatcher", "api", "--host", "127.0.0.1", "--port", "8000"]
```

`sha256sum Dockerfile` is then
`eb17ec9e4d3476edf073c2fde4c7cd944f8c10a0b05ba28e558b7d5977b8c85b`.

`deploy/lab.Dockerfile` repeats `USER 10001` and the JSON `HEALTHCHECK`, and
its comment says to delete the apt-get pin block once the base digest above
lands. Delete from the line `# Package pins for the current crown base`
through the configs-symlink `RUN`, including `USER 0`, and leave `USER 10001`
and the JSON `HEALTHCHECK`. `sha256sum deploy/lab.Dockerfile` is then
`6ffce9521541866421decdfcd68162d39d91ae495fa88c27e0b05c476736d739`.
`deploy/*` is an epoch corpus (current head
`corpus_epoch_42003ac55034faa4.json`, sha256
`5657868445c5ce4b49e92cea56f076ed383cbafd5fa9dead27b65aff14fb8e7b`).

## 5. Drop the accepted findings those two fixes close

After steps 3 and 4, delete these four objects from
`deploy/security/accepted-findings.json` and leave the ZAP object:

- `hadolint:DL3025:Dockerfile:31`
- `hadolint:DL3066:Dockerfile:28`
- `osv:GHSA-cx6f-vjhg-g3cc:virtualenv`
- `pip-audit:GHSA-cx6f-vjhg-g3cc:virtualenv`

The remaining file is:

```json
{
  "schema": "dipcatcher.security_accepted/1",
  "findings": [
    {
      "expiry": "2027-02-02",
      "id": "zap:10049:Non-Storable Content",
      "owner": "Cache-Control no-store is intentional."
    }
  ]
}
```

sha256 `7c4c779c0e9097e23c7dbc83089e1aa96f340dae74f953feb0b09dd104ffebc6`
(current file `0a25d4e8c536c0e03598e6248618923389c9e04a7f0cf39f34f56dd8ea56aaf4`).
Deleting the ids before the scanner is clean makes `security-gate` report
them as new findings. `trivy-allow.json` stays; it lists separate unfixed
image CVEs.

## 6. Canon qualification summary

`quality/canon_qualification_summary.json` sha256
`b5f7bd8e6288f50e272a90992a8d5b3659edfdf62cd61f0762c4377216cc65db`
(232812 bytes). `ruleset_version` is 1. `ruleset_hash` is
`7274f95d98db230d75b78a6752c909f21799428adb0e8c696375c4105f1cf3f4`.
`models_tree_sha256` is
`0176166492f704a80c1882bc99b426961e935344f9393f97b7e9f33537cd7578`
and matches the live `src/quant_fund/models/*.py` population (10388 files,
`__init__.py` and `canon_qualification.py` excluded). The recorded full dump
is `.dsh-24x7/canon_qualification_audit.json`, 24521759 bytes, sha256
`ad4fb2eb14f8554d5698dde71abd7d254eadbd017a1fc85d3aa3478c543d4934`.
That dump file is absent on this checkout, so `--check` is stale until it is
regenerated. An off-repo regeneration of the same tree
(`--dump` outside the repo) reproduced the dump byte-for-byte
(`ad4fb2eb14f8554d5698dde71abd7d254eadbd017a1fc85d3aa3478c543d4934`,
24521759 bytes) and the same counts, ruleset, family lists, and
`models_tree_sha256`. The only summary delta was the dump path argument.

```bash
uv run python scripts/canon_qualify.py --check
uv run python scripts/canon_qualify.py
uv run python scripts/canon_qualify.py --check
```

The first `--check` prints `STALE:` while the dump is missing. The default
regenerate writes `.dsh-24x7/canon_qualification_audit.json` and rewrites the
summary. On this tree the summary stays
`b5f7bd8e6288f50e272a90992a8d5b3659edfdf62cd61f0762c4377216cc65db` and the
second `--check` prints:

```text
OK: .dsh-24x7/canon_qualification_audit.json and quality/canon_qualification_summary.json match a fresh audit run
```

If the summary bytes change, `sha256sum quality/canon_qualification_summary.json`
is the digest step 10 stamps, and
`uv run python scripts/canon_qualify.py --emit-retired-families src/quant_fund/research/catalog/retired_families.py`
must be re-run in the same landing. The TiRex smoke edit is under `tests/`
and does not move `models_tree_sha256`. Extending the same eager fallback
into `src/quant_fund/models/tirex2.py` (where Windows already sets
`torch._dynamo.config.disable` when `cl` is absent) does move that digest
and belongs in this same regeneration.

## 7. Reseal `receipts/kill_audit.json` and refresh the evidence index

The committed receipt is file sha256
`bd65e84d2037d452282b5fba052b6be102fef2e50995a14cb2a381405d8d2565`,
embedded seal
`493cd55611dd273c687b0e94c6b3b6e11ed233ee9a3381cd498b7c38eb280e7c`,
`git_revision` `cb3c2e76d4ebc1d484d3fd59a2044cbd47820190`. The claim is
`n_checks` 57, `n_violations` 0, `verdict` `ok`. `docs/evidence/index.md`
records that row with seal column `not_checked`.

A fresh `kill_audit_bench()` on `a416f16648a0931e9e3161a0d46ae04ea183b91b`
still reports 57 / 0 / `ok`. Its seal is
`06f06b6df048345930571fa354170277cfb0f6044a0e4ca1139dbb409255d54e`,
and `json.dumps(payload, indent=2) + "\n"` is file sha256
`1b49526a60832f51c3affccf39d89d43e4dd43309a05b010aa9445852997dcb8`.
Any commit after that parent, including this checklist, changes
`git_revision` and therefore both digests. Reseal on the commit you are
about to stamp:

```bash
uv run python - << 'PY'
import json
from pathlib import Path
from quant_fund.monitoring.kill_audit import kill_audit_bench
payload = kill_audit_bench()
Path("receipts/kill_audit.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(payload["git_revision"])
print(payload["receipt_sha256"])
claim = payload["claim"]
print(claim["n_checks"], claim["n_violations"], claim["verdict"])
PY
sha256sum receipts/kill_audit.json
uv run dipcatcher verify-receipt receipts/kill_audit.json
make evidence
```

Expected claim line: `57 0 ok`. The printed `receipt_sha256` is the embedded
seal. `sha256sum` is the file digest. The evidence index row for
`receipts/kill_audit.json` must show that file digest, that seal, and the
printed `git_revision`. `receipts/*.json` current head is
`corpus_epoch_ff585b9d7c8badaa.json` (sha256
`3f1533c16f51153d03996f7253c79dc68816d8df89d214c8b37ca8ee17b27500`).
`docs/*` current head is `corpus_epoch_889962b885f2ee22.json` (sha256
`36fc98aa85bfb7b266bc484b47065673f4ec8273ebdebdafdcac45c08941c22e`).
`make evidence` rewrites only `docs/evidence/index.md`.

## 8. fx-1 weights

No fine-tuned fx-1 weights are published. The tracked file
`artifacts/fx1_tiny_lm/weights.safetensors` is a fixture (127000 bytes,
sha256 `cae879e7d8d80ed3a897e6ad550b909976d6ec098f3d45b4424f5be2a093ca07`).
`artifacts/fx1_tiny_lm/modelcard.json` labels it an internal research fixture,
`research_only: true`, and says it is not an fx-1 release candidate.
Confirm the digest and leave the file in place:

```bash
sha256sum artifacts/fx1_tiny_lm/weights.safetensors
```

Expected: `cae879e7d8d80ed3a897e6ad550b909976d6ec098f3d45b4424f5be2a093ca07`.
Publishing a checkpoint is a separate decision and a new artifact, followed
by another pass through steps 9–13. `fx1.__version__` `0.4.0` is the harness
API.

## 9. Rewrite the crown pin

Do this after steps 1–8 and before the stamp. It overwrites
`quality/crown_jewels.json`.

```bash
uv run dipcatcher crown-jewels --write
uv run dipcatcher crown-jewels --check
```

Expected write line, after the `DATA_LABEL=SYNTHETIC` line:
`crown-jewels pin=quality/crown_jewels.json jewels=76`.
Expected check: `DATA_LABEL=SYNTHETIC` then `crown-jewels intact`.

These eleven files already differ from the pin. `--write` records the
working-tree digest. Steps 1, 3, and 4 replace four of them; the other seven
stay at the actual column if you leave their bytes alone.

| File | Pinned | Actual before steps 1–4 |
|---|---|---|
| `.env.example` | `3601a96a5af0539c0f7e9358d984be5c9e01cc4d8ac81da6f6e31d6c7b2f0cc7` | `6dd1359ba4cbd3d7bda595ca33f0307aadb80ff2c6b9f16c9728bcf860c9398b` (step 1 → `972de84e5ab89ecf64a4dce4bb648e569a361b9722c1277a368615c9365a1d2c`) |
| `.gitleaks.toml` | `c7d9a797adcecacdaa364fc245d798d84b3df69bbc97a6720852deb113eadf64` | `c731d4b8483fd4d41aec45114a98143eb0bb3c882797b9d496d26fa9540bf14f` |
| `README.md` | `05eb7f9fd83b4383610b17a113ac63f99eee030bc9e8fe1dfc2fc1aead04b480` | `145e0d7139913781dae6b5be9862cd7175834e921bad31422b54bed36ae430ba` |
| `mkdocs.yml` | `e4f2931d9a3151f4bd8e73a7c2569855aa7d6565b56cc06e00040449402275f0` | `48c76104c8df8e868d5abd4e14d98bc453bb5b184098a9cf54d9caee48a56613` |
| `pyproject.toml` | `6c4f2a92d811a9be0885326205c1b94f30ef46ab002242234fcb271d0b04e454` | `68483824a560eaf795641b9b12b87f1a5db6b6b7067b44e2afbb08be2ec45e4a` (step 3 → `24adf9529e5e4e358566c31b067c834b968bae735d321e22bb42c1733cc1c0e3`) |
| `src/quant_fund/research/auditor_bundle.py` | `c525bb4be737efefc1dd1a80cd6dac755af555864614c77f8ce9f96c639abcf9` | `aecc4415c0ebc8ca09f5165bd72d44ea142fc0b8e01b4d1abf4219f7922b48e3` |
| `src/quant_fund/research/integrity_witness.py` | `85586dd844821086be097e8be69effd16e6480e649de4ab0d83381e8d2646e08` | `91da1897dd5a15ca7beeb3bfb4fc4452ddf38d9756ce4ad086f970348c186dbb` |
| `src/quant_fund/research/ots_anchor.py` | `7e9e2c10a47c3e71664e1e352034036336f9b5b31f49ab7d7f0be1cd8d432ac6` | `866e998372a22e4e69825fbff4a2aadd1383b964f689faa95a342bf57d3e55ff` |
| `src/quant_fund/research/timestamp_anchor.py` | `2a504facaea79184b87af5b9ce0dbd816060ad14ebd0991080bb65fee9e45905` | `216b5c3fc478c6bd8c83e3c1b6b635c243aa4ac1bf16ee130c99dc6674a2140a` |
| `src/quant_fund/research/witness_scan.py` | `a664fd4b49c94539d725379a2fe58b0d08fcbcfd9284f331123ec70b6394ebea` | `1fb61fcba3a5e37b9cd3a463f353e732ad6ce0a8fcbe9b1f8beeeb77fe978e7f` |
| `uv.lock` | `f4b4f747b433f61730ec4d4cef03ce219a7264d9a8f9e030a8d286f5d76e8627` | `ed8397b8850b5cbf19c3ad535d0ecae7dab8de0cc23a6351f63fd403dc0c86d5` (step 3 replaces this) |
| `Dockerfile` | `f367dfc1d9634fccb6c24f264c78b436304a30da148715c3e83a31463799f005` | same as the pin (step 4 → `eb17ec9e4d3476edf073c2fde4c7cd944f8c10a0b05ba28e558b7d5977b8c85b`) |

## 10. Stamp epochs

```bash
make stamp-epochs
```

This re-stamps every corpus, including ones this work already changes.
`n_epochs` below is the `quality/epoch_heads.json` field for that pattern.
`tests/*` (TiRex smoke and the fx1 import-contract test; current head
`corpus_epoch_5ee56d61178506d3.json`, sha256
`c053bf392ef161f78ca904e3cec331eedb7ad13d9b69be58cbf562ece2467d7f`,
`n_epochs` 36), `docs/*` (this checklist, the atlas refresh, and the evidence
index from step 7; current head `corpus_epoch_889962b885f2ee22.json`, sha256
`36fc98aa85bfb7b266bc484b47065673f4ec8273ebdebdafdcac45c08941c22e`,
`n_epochs` 13), `quality/*.json`
(census, crown pin, canon summary; current head
`corpus_epoch_c7e2dfd9fa8fbe73.json`, sha256
`ef9215841605d5ee548b583c2648357ab48b2b0b1ed992bd38bb477f76b02bb5`,
`n_epochs` 263), `receipts/*.json`, `deploy/*`, and `src/*` (current head
`corpus_epoch_60c46c6a0dcfc751.json`, sha256
`688f5bea98ee82d6b31b0af12e62975b95bacd5e54062c6f55cb0283740c36dd`,
`n_epochs` 44). Each `dipcatcher corpus-epoch` command exits 0. The new
receipt names are the hashes of the new trees and are printed by the command.
`src/*` moves because `src/fx1/__init__.py` loads secret redaction by file
path. It also moves if step 6 rewrites `retired_families.py` or a models file.

## 11. Sign the pins

```bash
make sign-pins
```

Requires `GATE_SIGNING_KEY` (`<priv_hex>:<pub_hex>` or a path to a mode-`0600`
file with those two lines). `quality/gate_signing.pub` is one line of hex:
`6de7a13e9079e74e28a77848ba79731ee55f1a66aecb641cb619b0eb2f7aa674`.
`key_id` is the first 16 hex characters of SHA-256 over those decoded bytes.
`sha256sum quality/gate_signing.pub` hashes the text file. The expected line is:

```text
signature=gate_pins.sig key_id=bc3bcff676056adf
```

The signature hex covers the pin bytes from steps 9 and 10 and is produced
by that command. `gate_pins.sig` lives at the repo root, outside the epoch
corpora.

## 12. Anchor the pins

```bash
make anchor-pins
```

RFC 3161 via FreeTSA, network, quiet point. It anchors
`quality/epoch_heads.json` and `quality/crown_jewels.json`. Exit 0.
Another `make stamp-epochs` after this stales both anchors.

## 13. Checkpoint

```bash
make checkpoint
```

Run last among the signers. It binds the current `gate_pins.sig` into
`quality/checkpoint.json` and needs the same `GATE_SIGNING_KEY`. Optional
follow-ups, each with its own key or network: `make anchor-checkpoint`,
`make witness-checkpoint` (`WITNESS_SIGNING_KEY`), then `make verify-witness`,
`make checkpoint-chain`, and `make verify-rotations`.

After 11–13:

```bash
make epoch-consistency EPOCH_BASE=<ref-before-these-edits>
make evidence-audit
```

`evidence-audit` re-verifies receipts, epoch chains, crown jewels, the
witness, the repo, the checkpoint spine, rotations, the tamper drill, and
`fuzz-drill --seed 7`.

## 14. Sign an audit-log anchor

This does not modify the git tree. `scripts/anchor_audit_head.py` appends one
unsigned JSON line (`head`, `time`) to the anchors file and does not sign it.
An empty chain's `head` is the SHA-256 of empty bytes:
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
A non-empty chain's `head` is the SHA-256 of the last raw audit line.

```bash
python scripts/anchor_audit_head.py PATH --anchors ANCHORS
python scripts/verify_audit_chain.py PATH
python scripts/prod_backup.py --audit-log PATH --anchors ANCHORS --dest DIR
```

`verify_audit_chain.py` prints `audit chain ok` and exits 0 when the chain
holds. Signing the `head` digest uses the same key material as
`gate_pins.sig`. The repo has no command that writes that signature; keep it
with the anchors file (mode `0600`), beside the prod-backup copy.

## 15. Cut the `v*` tag

`git tag -l 'v*'` is empty. The four existing tags are
`attic/receipt-provenance/*`. `install.sh` defaults `FXI_VERSION` to `0.4.0`
and fetches
`https://github.com/artificial-hedge/dipcatcher/releases/download/v0.4.0/fx_1-0.4.0-py3-none-any.whl`.
`release.yml` runs on tags matching `v*.*.*`, builds the sdist and wheel,
writes a CycloneDX 1.5 SBOM, signs the distributions with Sigstore, and
uploads workflow artifacts. It does not publish to PyPI.

Tag the signed SHA only after `CI` and `evidence-audit` are green on that
SHA:

```bash
git tag v0.4.0 <signed-sha>
git push origin v0.4.0
```

The release job's SBOM step prints `CycloneDX 1.5 components=<N>`. The
verify step prints `verifying certificate identity https://github.com/<owner>/<repo>/.github/workflows/release.yml@refs/tags/v0.4.0`.
The wheel sha256 is the `SHA256SUMS` line the job writes under `dist/` and
is not knowable before that build. Confirm the tagged tree still shows
`sha256sum artifacts/fx1_tiny_lm/weights.safetensors` equal to
`cae879e7d8d80ed3a897e6ad550b909976d6ec098f3d45b4424f5be2a093ca07`.
