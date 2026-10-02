# Helper scripts

Operational scripts that aren't part of the installed packages. Prefer the
`fx1` / `dipcatcher` (`quant`) CLIs for day-to-day work — these are one-off
or bench utilities.

| Script | Purpose |
| --- | --- |
| `gmm_smoke.py` | GMM estimator smoke run |
| `merge_gmm.py` | Merge GMM score outputs |
| `new3_smoke.py` | NEW3 release smoke run |
| `qar_smoke.py` | QAR estimator smoke run |
| `run_gmm_pass.py` / `run_rest_pass.py` / `run_skt_pass.py` | Remote fleet bench-pass launchers (Windows host only) |
| `skt_smoke.py` | SKT estimator smoke run |
| `secret_scan.py` | Pre-commit hook: rejects staged secrets |

Run the smoke/utility scripts with `uv run python scripts/<name>.py` inside the
synced env. The `run_*_pass.py` drivers spawn `D:\evalenv\Scripts\python.exe`
against hard-coded `D:\dipcatcher` shard paths, so they are only runnable on
the remote Windows fleet host.
