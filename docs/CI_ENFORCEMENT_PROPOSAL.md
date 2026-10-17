# Proposed `main` CI ruleset — 2026-10-16

This is a **reviewable proposal, not an applied repository setting**. The
workflow changes must land on the default branch and the current red evidence
gates must be resolved before a successful merge-queue trial can be recorded.

## Measured state

- `gh api repos/artificial-hedge/dipcatcher/branches/main/protection/required_status_checks` returns `404 Branch not protected`.
- The only active repository rulesets are `24646693` (creation restriction on branches other than `main`, `isolated`, `others`) and `24648163` (deletion restriction on those three branches). Neither requires a status check or merge queue.
- `gh api repos/artificial-hedge/dipcatcher` reports `permissions.admin: true` for the active account. `gh auth status` reports classic token scopes `repo` and `workflow`; no settings-write call has been attempted. The existing GitHub Actions check runs identify the expected source as app ID `15368` (`github-actions`).
- `ci.yml`, `fx1.yml`, and `proofcore.yml` now request `merge_group.checks_requested` for `main`. Each has one unique aggregate check, respectively `ci-required`, `fx1-required`, and `proofcore-required`. Each aggregate depends on **every** job in its workflow and fails if any dependency is failed, cancelled, or skipped. `tests/unit/test_workflow_security.py` prevents future jobs from being omitted from these dependencies. The full 45-job platform matrix is nightly/manual and is a separate release-baseline check; do not select its jobs as required PR checks.

GitHub [requires the `merge_group` trigger](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue#triggering-merge-group-checks-with-github-actions) for queue checks, and [ruleset status checks are job contexts](https://docs.github.com/en/rest/repos/rules#parameters-for-create-a-repository-ruleset). A workflow name alone is not the status context.

## Exact settings proposed for review

Create **one additional repository branch ruleset** targeting exactly
`refs/heads/main`; retain the two existing rulesets. The following is the
proposed REST payload, not a command to execute:

```json
{
  "name": "main CI and merge queue",
  "target": "branch",
  "enforcement": "active",
  "bypass_actors": [],
  "conditions": {
    "ref_name": {"include": ["refs/heads/main"], "exclude": []}
  },
  "rules": [
    {
      "type": "required_status_checks",
      "parameters": {
        "do_not_enforce_on_create": false,
        "strict_required_status_checks_policy": false,
        "required_status_checks": [
          {"context": "ci-required", "integration_id": 15368},
          {"context": "fx1-required", "integration_id": 15368},
          {"context": "proofcore-required", "integration_id": 15368}
        ]
      }
    },
    {
      "type": "merge_queue",
      "parameters": {
        "check_response_timeout_minutes": 180,
        "grouping_strategy": "ALLGREEN",
        "max_entries_to_build": 1,
        "max_entries_to_merge": 2,
        "merge_method": "MERGE",
        "min_entries_to_merge": 1,
        "min_entries_to_merge_wait_minutes": 0
      }
    }
  ]
}
```

`max_entries_to_build: 1` bounds the current ~37-job fast-gate fan-out to
one queue candidate at a time against the observed 20-job concurrency cap.
The 180-minute response window covers the 90-minute fx1 coverage budget plus
queue delay; it is an upper bound, not an expected runtime. `ALLGREEN` requires
each queued entry's merged commit to pass. `MERGE` preserves the repository's
current merge-commit style. Loose status checks avoid forcing PR authors to
rebase solely because `main` moved; the queue checks the actual synthetic merge
commit against the current base. The queue can merge at most two successful PRs
in one push.

## Activation evidence still needed

1. Land the three workflows on the default branch. Observe all three new
   aggregate context names from GitHub Actions on a test PR, and confirm their
   source is app ID `15368`. A workflow file only in this worktree cannot emit
   queue checks.
2. Resolve the current proofcore replay and signed-pin failures; do not
   configure a green requirement by excluding those jobs. The aggregate checks
   intentionally remain red until all their dependencies pass.
3. Review the change to `AGENTS.md`'s current direct-to-`main` convention:
   required checks and a merge queue make ordinary unchecked direct pushes
   incompatible with this production gate. Then apply the reviewed ruleset
   without a blanket bypass and exercise a real `merge_group` run on `main`.
4. Check the resulting ruleset via the read API, verify all three required
   contexts on the merge-group SHA, and record a green full-platform manual
   matrix run on the **same candidate SHA** before declaring a release baseline.

GitHub's [ruleset parameters](https://docs.github.com/en/rest/repos/rules#parameters-for-create-a-repository-ruleset) define the proposed field names and merge-queue values. No repository setting was written during this audit.

## Release preflight and remaining product gates

`.github/workflows/release.yml` runs on a `v*.*.*` tag. Its read-only
`preflight` job now requires `GITHUB_REF` to be an exact SemVer tag matching
Hatch's literal `fx1.__version__` source, the tag commit to be in `main` history, and the
**latest** exact-SHA `main` push runs of `ci.yml`, `fx1.yml`, and `proofcore.yml` to be
complete and successful, with each unique aggregate job successful. It also
requires the latest exact-SHA scheduled or manual `matrix.yml` run to be
complete and successful with all 45 platform jobs present and successful.
The signing/build job depends on this preflight. The [workflow-runs
API](https://docs.github.com/en/rest/actions/workflow-runs#list-workflow-runs-for-a-workflow)
accepts `head_sha` and `event` filters; the preflight prints each accepted run
URL for an audit trail. A tag created before these checks finish fails and may
be rerun after they pass, without moving the tag. This new preflight has not
yet run on GitHub and is not a green release verdict.

The wheel smoke exercises installed command entry points and bundled files.
It does not prove publicly served fx1/fx1-lite model behavior or live-trading
readiness. Add separate serving and live-trading release gates once their
deployable assets and safe live-path acceptance criteria exist; a successful
`Release` workflow alone is not a production go-live verdict for the three
product surfaces.
