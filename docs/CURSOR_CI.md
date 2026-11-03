# Cursor cloud-agent CI

Pull requests into `main` and pushes to `main` launch a Cursor cloud agent
that runs the heavy checks. The GitHub Actions job in
`.github/workflows/cursor-ci.yml` only launches that agent and polls it. The
runner stays `instabox-large`. The agent makes no code changes.

The Actions job passes only when the agent finishes and the commit status
context `cursor-ci` on the head SHA is `success` with description
`CURSOR_CI_RESULT: pass`. On a pull request the same result must appear in
exactly one issue comment containing `<!-- cursor-ci -->`. Anything else
fails the job: missing secret, HTTP error, timeout (90 minutes), agent
status other than `FINISHED`, or a report that is not that pair of tokens.

## Secret Advaith must set

Repository: `artificial-hedge/dipcatcher`.

1. Create a Cursor API key from the Cursor dashboard (API Keys). The v0
   Cloud Agents API uses it as the HTTP Basic username with an empty password.
2. In GitHub, open **Settings → Secrets and variables → Actions → Secrets**.
3. Add a repository secret named exactly `CURSOR_API_KEY`. Paste the key as
   the value.

Do not commit the key. Do not put it in a workflow, a log, a prompt, or a
pull request. The workflow reads it only as `${{ secrets.CURSOR_API_KEY }}`.

The Cursor account or service account that owns the key must be allowed to
start cloud agents on `artificial-hedge/dipcatcher`. The agent's GitHub
connection must be allowed to create the commit status context `cursor-ci`
and to comment on pull requests. Fork pull requests do not receive Actions
secrets, so this job fails closed on them.

## Switching the legacy workflows back on

Legacy heavy jobs still exist. On `pull_request` and `push` they are skipped
unless the repository **variable** `USE_CURSOR_CI` is exactly `false`.

**Settings → Secrets and variables → Actions → Variables**:

| `USE_CURSOR_CI` | cursor-ci job | legacy heavy jobs on pull_request and push |
| --- | --- | --- |
| unset, or any value other than `false` | runs | skipped |
| `false` | skipped | run |

`schedule` and `workflow_dispatch` always run the legacy workflows, including
when the variable is unset. That is the manual path for the macOS/Windows
matrix and for a full Actions replay.

These workflows stay on the Actions runner either way, because they publish
a GitHub result a cloud agent cannot replace:

- `codeql.yml` (code scanning)
- `dependency-review.yml`
- `scorecard.yml`
- `release.yml` (Sigstore and SLSA attestations)
- `docs.yml` job `deploy` (GitHub Pages)
- `ci.yml` job `package` on **push** (the attest step needs the Actions
  OIDC token; pull requests skip that job while cursor CI is on)

`property-nightly.yml` was already `schedule` and `workflow_dispatch` only.

## Branch protection

Required check: the Actions job `cursor-ci` in workflow `cursor-ci`. The
commit status context is also `cursor-ci`; the Actions job is the one that
fails closed when that status is missing.

While cursor CI is on, legacy jobs report as skipped. Point required checks
at `cursor-ci` rather than at `CI / lint`, `CI / test`, and the other heavy
job names. Set `USE_CURSOR_CI=false` before requiring those legacy names
again.

One Linux agent runs the lab suite once: `pytest -n 2` over the full offline
suite (`-m "not network"`), not the eight-way shard matrix and not
macOS/Windows. Those legs stay on `workflow_dispatch` of `matrix.yml`.

## Epoch pins

Editing a workflow invalidates the `.github/workflows` corpus epoch
(`quality/epoch_heads.json`, quorum-signed via `gate_pins.sig`). New files
under `scripts/`, `tests/`, and `docs/` invalidate those corpora too.
Re-stamp only with `make stamp-epochs`, then `make sign-pins`, then
`make anchor-pins`. Do not hand-edit `corpus_epoch_*.json` or `gate_pins.sig`.
Until that stamp lands, `make evidence-audit` and epoch consistency fail on
member drift. That failure is the guard working. Do not bypass it.

## API

Launcher: `scripts/ci/cursor_ci.py`. Prompt: `.github/cursor-ci/prompt.md`.

`POST https://api.cursor.com/v0/agents` with `source.ref` set to the head
SHA and `target.autoCreatePr` false. Poll `GET /v0/agents/{id}`. v0 is the
legacy API; it still documents this request shape. Basic auth, key as the
username. The prompt tells the agent not to commit or push. Cursor may still
keep an internal agent branch; that branch is not the CI result.

## Scorecard row `pr_test_suite`

`make prod-readiness` does not run the full pytest suite. It does not post a
commit status, generate a key, or sign a result. The row stays `needs-owner`
until an external attestation names the exact `git rev-parse HEAD` sha.

Either of these is enough:

1. A GitHub combined commit status whose context is `cursor-ci`, whose state
   is `success`, and whose description is `CURSOR_CI_RESULT: pass`. The
   scorecard reads `GET /repos/{owner}/{repo}/commits/{sha}/status` when
   `GITHUB_TOKEN` or `GH_TOKEN` is set, and otherwise asks `gh api` for that
   GET. A missing remote, a missing status, or a transport error leaves the
   row at `needs-owner`.
2. A signed JSON file. Set `CURSOR_CI_RESULT_FILE` to the file and
   `CURSOR_CI_RESULT_SIG` to the detached signature (the default is the
   result path plus `.sig`). Set `CURSOR_CI_ATTESTATION_PUBKEY` to the
   64-character hex Ed25519 public key, or `CURSOR_CI_ATTESTATION_PUBKEY_FILE`
   to a file that contains only that hex. The signature covers the exact
   file bytes. The object is exactly these four fields:

```json
{"context":"cursor-ci","result":"pass","schema":"cursor_ci_result.v1","sha":"<40 lowercase hex>"}
```

`result` is `pass` or `fail`. A verified `fail`, a bad signature, a
malformed object, or a pass that disagrees with a fail is a scorecard
`fail`. A file whose `sha` is a different commit is ignored.

The scorecard ignores a symlink, a path inside its work directory, and any
tracked path. A commit cannot carry its own pass. The manual CI agent writes
the file outside the commit and signs it with a key this repository does
not hold. Do not generate that key inside the scorecard.
