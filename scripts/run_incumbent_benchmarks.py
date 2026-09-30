"""Run incumbent benchmarks in fresh subprocesses and write provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BARS = [ROOT / "data/raw/sources" / name for name in ("btcusdt_1d.parquet", "ethusdt_1d.parquet", "solusdt_1d.parquet")]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_new(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing path: {path}")


def run_one(*, label: str, interpreter: Path, script: Path, bars: list[Path], reps: int, run_dir: Path, run_number: int) -> dict[str, Any]:
    receipt = run_dir / f"{label}-{run_number:02d}.json"
    stdout_path = run_dir / f"{label}-{run_number:02d}.stdout.txt"
    stderr_path = run_dir / f"{label}-{run_number:02d}.stderr.txt"
    for path in (receipt, stdout_path, stderr_path):
        require_new(path)
    command = [str(interpreter), str(script), *[arg for bar in bars for arg in ("--bars", str(bar))], "--reps", str(reps), "--out", str(receipt)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    result: dict[str, Any] = {
        "label": label, "run_number": run_number, "command": command, "cwd": str(ROOT),
        "interpreter": str(interpreter), "script": str(script), "stdout": str(stdout_path),
        "stderr": str(stderr_path), "receipt": str(receipt), "exit_code": completed.returncode,
        "stdout_sha256": sha256(stdout_path), "stderr_sha256": sha256(stderr_path),
    }
    if completed.returncode == 0:
        result["receipt_sha256"] = sha256(receipt)
    else:
        result["error"] = "benchmark subprocess failed; receipt hash omitted"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True, help="interpreter containing vectorbt and qlib")
    parser.add_argument("--bars", type=Path, nargs="+", default=DEFAULT_BARS)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--runs", type=int, default=2, help="fresh subprocesses per incumbent")
    parser.add_argument("--run-id", default=None, help="unique output name; never overwritten")
    parser.add_argument("--out-dir", type=Path, default=ROOT / ".dsh-24x7" / "incumbent-runs")
    args = parser.parse_args()
    if args.reps < 1 or args.runs < 1:
        parser.error("--reps and --runs must be positive")
    interpreter = args.python.resolve()
    bars = [bar.resolve() for bar in args.bars]
    scripts = {name: (ROOT / "scripts" / name).resolve() for name in ("incumbent_bench_vectorbt.py", "incumbent_bench_qlib.py")}
    labels = {"incumbent_bench_vectorbt.py": "vectorbt", "incumbent_bench_qlib.py": "qlib"}
    scripts = {labels[name]: path for name, path in scripts.items()}
    for path in [interpreter, *bars, *scripts.values()]:
        if not path.is_file():
            raise SystemExit(f"required file is missing: {path}")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (args.out_dir / run_id).resolve()
    if run_dir.exists():
        raise SystemExit(f"refusing to use existing output directory: {run_dir}")
    run_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": "incumbent_runs.v1", "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "research_only": True,
        "live_pnl_claim": False, "cwd": str(ROOT), "interpreter": str(interpreter),
        "interpreter_sha256": sha256(interpreter), "python_version": platform.python_version(),
        "platform": platform.platform(), "reps_per_process": args.reps,
        "fresh_processes_per_incumbent": args.runs,
        "bars": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in bars],
        "benchmark_scripts": {name: {"path": str(path), "sha256": sha256(path)} for name, path in scripts.items()},
        "runs": [], "disclaimer": "Fixed real Binance daily diagnostic; not production, live-P&L, universal superiority, or SOTA proof. Fresh subprocesses are not clean-environment or independent third-party validation.",
    }
    for run_number in range(1, args.runs + 1):
        for label, script in scripts.items():
            print(f"running {label} fresh process {run_number}/{args.runs}", flush=True)
            result = run_one(label=label, interpreter=interpreter, script=script, bars=bars, reps=args.reps, run_dir=run_dir, run_number=run_number)
            manifest["runs"].append(result)
            print(f"{label} exit_code={result['exit_code']} receipt={result['receipt']}", flush=True)
            if result["exit_code"] != 0:
                break
        if manifest["runs"][-1]["exit_code"] != 0:
            break
    manifest_path = run_dir / "manifest.json"
    require_new(manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failures = [item for item in manifest["runs"] if item["exit_code"] != 0]
    print(f"manifest: {manifest_path}")
    print(f"runs: {len(manifest['runs'])} failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
