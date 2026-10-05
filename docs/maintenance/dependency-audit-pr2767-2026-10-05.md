# PR #2767 dependency advisory supplement — 2026-10-05

The fresh PyPI advisory lookup returned **no known advisory matches** for
`anthropic==1.11.0` and `docstring-parser==0.18.0`, with no skipped packages.
It completed at **17:55:12 UTC on 2026-10-05** and is bound to merged commit
[`3fc9c007a111bd10c1f407b93d93b69ab64dca9b`](https://github.com/artificial-hedge/dipcatcher/commit/3fc9c007a111bd10c1f407b93d93b69ab64dca9b).

This supplements the [original dependency scan](dependency-audit-2026-10-05.md).
The original scan of 228 Python registry pins remains tied to commit
`f7f94b700dab770e968c761736a98f6256801754` and its original completion time,
17:06:40 UTC. Its evidence files were not modified.

## Exact scope and results

| Added locked package | Role | Advisory lookup result |
|---|---|---|
| `anthropic==1.11.0` | Development SDK used by the Anthropic compatibility audit | No known matches; not skipped |
| `docstring-parser==0.18.0` | Dependency of the development SDK | No known matches; not skipped |

`pip-audit 2.10.1` queried the PyPI advisory API with strict collection,
dependency resolution disabled, and a new empty HTTP cache. It returned exit
code **0** and exactly the two expected package results. This run did not
query the unchanged 228 pins again or install the project dependencies.

## Frozen export verification

The committed lock moved from revision 3 to revision 5. `uv 0.12.19` exported
the revised lock successfully with `--offline --frozen`, all groups and all
extras. After removing environment markers and deduplicating exact pins, the
export matched **all 230 registry package/version pairs** in the revised lock.
Comparing that inventory with the original scan's lock found exactly the two
additions above, **228 unchanged pins**, and no removed or changed versions.
The copied project manifest and lockfile remained byte-for-byte unchanged.

| Source file | Committed Git blob |
|---|---|
| `pyproject.toml` | `9b47010e1684d4d67086da6f1ce95ba51814532b` |
| `uv.lock` | `d1022fa7a4af7e2cb7e49aa88300b2752574dab2` |

The revised `uv.lock` SHA-256 is
`bf8e4106179129659602326b7ace9f650c302ed05c9967851861f3eb0bcd484f`.

## Commands

The export used isolated copies of the two committed source files:

```sh
uv export --offline --frozen --all-groups --all-extras --no-emit-project \
  --format requirements.txt --no-hashes --no-header --no-annotate \
  --project <snapshot> -o <scratch>/locked_with_markers.txt
```

The exported pins were normalized in Python and checked for exact equality
with the lock's registry package inventory. The supplemental requirements
file contained only these two lines:

```text
anthropic==1.11.0
docstring-parser==0.18.0
```

The advisory command used a newly created, empty cache directory:

```sh
uvx --no-env-file --from pip-audit==2.10.1 pip-audit \
  --strict --disable-pip --no-deps --vulnerability-service pypi \
  --cache-dir <fresh-empty-cache> --progress-spinner off --timeout 10 \
  -r <scratch>/new_locked_pins.txt --format json \
  --output <scratch>/pip_audit.json
```

The tool warned that fully hashed requirements are preferable. No package
artifact installation or integrity verification was part of this lookup.
The [pip-audit documentation](https://pypi.org/project/pip-audit/2.10.1/)
describes the advisory service, exact-version mode, and exit semantics.

## Evidence and limits

[Supplemental raw results and provenance](dependency-audit-pr2767-2026-10-05.json)
include the exact command, both raw package results, warnings, source hashes,
the complete 230-pin export inventory, and the original scan's provenance.

These results are an advisory lookup for the two exact versions at the
recorded time. They do not establish that the dependencies or repository are
free of vulnerabilities, and they do not inspect source defects, exploit
reachability, malicious packages, or unpublished issues. Frozen export was
verified with uv 0.12.19; compatibility with older uv readers and a full
dependency installation were not tested.
