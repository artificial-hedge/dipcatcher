"""``dipcatcher explain`` sub-typer — model explainability reports.

Wraps :mod:`quant_fund.research.explainability`, which shipped fully documented
with 16 tests and no operator surface at all: permutation/SHAP attribution,
partial dependence, attribution drift, the Markdown/HTML/JSON report writer, and
the additive receipt sidecar that binds a report to sealed evidence without
ever modifying it.

Honesty contract: attribution is always measured as *degradation of a proper
score* (pinball / CRPS / Brier) — never accuracy, never Sharpe, never P&L. The
``demo`` command fits a least-squares head on a seeded SYNTHETIC planted-signal
panel and says so; ``report`` on user-supplied artifacts relabels the evidence
as REAL but that is a label, not a performance claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

explainability_app = typer.Typer(
    help="Model explainability: proper-score attribution, partial dependence, drift, "
    "and receipt sidecars. Research-only; never a performance claim."
)

#: ``shap`` is an optional extra and is excluded on darwin/x86_64 (see the
#: pyproject comment), so a missing import is a typed skip, not a traceback.
_SHAP_UNAVAILABLE_EXIT = 3


def _data_label(*, synthetic: bool, data_source: str) -> str:
    """Always-printed DATA_LABEL line.

    Mirrors :func:`quant_fund.cli.support.format_data_label`; duplicated rather
    than imported because the ``library-no-cli-imports`` architecture guard
    forbids a research-layer module from importing the CLI even lazily.
    ``tests/unit/research/test_explainability_cli.py`` pins the two in agreement.
    """
    return f"DATA_LABEL={'SYNTHETIC' if synthetic else data_source}"


def _echo_report(report: Any, *, top_k: int) -> None:
    typer.echo(
        f"model={report.model_name} family={report.model_family} scoring={report.scoring} "
        f"rows={report.n_rows} features={len(report.feature_names)} seed={report.seed}"
    )
    typer.echo(f"baseline_score={report.baseline_score:.6f} (lower is better)")
    typer.echo(f"attribution method={report.attribution.method}")
    for item in report.attribution.top(top_k):
        typer.echo(
            f"  #{item.rank:<2d} {item.feature:<20} importance={item.importance_mean:+.6f} "
            f"std={item.importance_std:.6f}"
        )
    if report.shap is not None:
        shap_top = report.shap.top(min(top_k, len(report.shap.attributions)))
        typer.echo(f"shap method={report.shap.method}")
        for item in shap_top:
            typer.echo(
                f"  #{item.rank:<2d} {item.feature:<20} mean_abs_shap={item.importance_mean:.6f}"
            )
    if report.pd_curves:
        typer.echo(f"partial_dependence curves={len(report.pd_curves)}")
        for curve in report.pd_curves:
            values = curve.mean_curve
            spread = (max(values) - min(values)) if values else float("nan")
            typer.echo(
                f"  {curve.feature:<20} points={len(curve.grid)} mean_curve_spread={spread:.6f}"
            )
    drift = report.drift
    if drift is not None:
        typer.echo(
            f"drift blocks={drift.n_blocks} min_spearman={drift.min_spearman:.4f} "
            f"max_js={drift.max_js:.4f} mean_topk_overlap={drift.mean_topk_overlap:.4f} "
            f"flagged={str(drift.drift_flagged).lower()}"
        )
    for warning in (
        *report.attribution.warnings,
        *(drift.warnings if drift else ()),
        *report.warnings,
    ):
        typer.echo(f"warning: {warning}")


class _LeastSquaresHead:
    """Deterministic least-squares head used by ``explain demo``.

    Deliberately minimal so the demo exercises the explainability pack rather
    than a model family. With ``taus=None`` it is a point forecaster (the
    conditional median, which is what ``pinball(0.5)`` is strictly proper for).
    With ``taus`` it emits an ``(n, k)`` Gaussian quantile fan around that
    median at the fitted residual scale — which is what a multi-tau CRPS score
    requires, and a real predictive distribution rather than a point forecast
    dressed up as one.
    """

    def __init__(
        self, coef: Any, *, sigma: float = 0.0, taus: tuple[float, ...] | None = None
    ) -> None:
        self.coef = coef
        self.sigma = float(sigma)
        self.taus = None if taus is None else tuple(float(t) for t in taus)

    def predict(self, x: Any) -> Any:
        import numpy as np

        point = np.asarray(x, dtype=float) @ self.coef
        if self.taus is None:
            return point
        from scipy.stats import norm

        z = np.asarray(norm.ppf(self.taus), dtype=float)
        return point[:, None] + self.sigma * z[None, :]


def _demo_taus(scoring: str) -> tuple[float, ...] | None:
    """Quantile grid the demo head must emit for ``--scoring``, or None.

    A point score needs a point forecast; a multi-tau CRPS needs the matching
    grid, or ``crps_from_quantiles`` is handed the wrong number of columns.
    Brier is rejected outright: the demo target is continuous, and Brier needs
    binary labels with predictions in [0, 1].
    """
    from quant_fund.research.explainability.scoring import DEFAULT_CRPS_TAUS

    name, _, arg = scoring.partition(":")
    if name.strip().lower() == "brier":
        raise ValueError(
            "the demo panel has a continuous target; --scoring brier needs binary "
            "labels — use 'explain report' with a probability head instead"
        )
    if name.strip().lower() != "crps":
        return None
    if not arg.strip():
        return DEFAULT_CRPS_TAUS
    try:
        taus = tuple(float(part) for part in arg.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"--scoring crps levels must be comma-separated floats: {exc}") from exc
    if len(taus) < 2:
        raise ValueError("--scoring crps needs at least two quantile levels")
    if any(t <= 0.0 or t >= 1.0 for t in taus):
        raise ValueError("--scoring crps levels must lie strictly inside (0, 1)")
    if any(b <= a for a, b in zip(taus, taus[1:], strict=False)):
        raise ValueError("--scoring crps levels must be strictly increasing")
    return taus


def _demo_panel(
    seed: int, n_rows: int, noise: float, *, taus: tuple[float, ...] | None = None
) -> tuple[Any, Any, list[str], Any]:
    """Seeded SYNTHETIC planted-signal design: ``y = X @ w + eps``.

    ``signal_core`` carries the signal and ``noise_a``/``noise_b`` carry none,
    so a correct attribution run must rank ``signal_core`` first and the noise
    columns near zero — that contrast is what the demo is for.
    """
    import numpy as np

    names = ["mom_20", "reversal_1", "vol_20", "signal_core", "noise_a", "noise_b"]
    weights = np.asarray([0.4, 0.2, -0.1, 4.0, 0.0, 0.0], dtype=float)
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n_rows, len(names)))
    y = x @ weights + rng.normal(0.0, noise, size=n_rows)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    coef = np.asarray(coef, dtype=float)
    sigma = float(np.std(y - x @ coef))
    return x, y, names, _LeastSquaresHead(coef, sigma=sigma, taus=taus)


def _load_frame(path: Path) -> Any:
    import polars as pl

    if not path.is_file():
        raise ValueError(f"file not found: {path}")
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def _load_xy(
    features: Path, labels: Path | None, label_col: str | None
) -> tuple[Any, Any, list[str]]:
    """Build the (x, y, feature_names) triple from user-supplied artifacts."""
    import numpy as np

    frame = _load_frame(features)
    numeric = [c for c in frame.columns if frame[c].dtype.is_numeric()]
    if labels is not None and label_col is not None:
        raise ValueError("pass either --labels or --label-col, not both")
    if label_col is not None:
        if label_col not in frame.columns:
            raise ValueError(f"--label-col {label_col!r} is not a column of {features.name}")
        y = frame[label_col].cast(float).to_numpy()
        names = [c for c in numeric if c != label_col]
    elif labels is not None:
        if not labels.is_file():
            raise ValueError(f"labels file not found: {labels}")
        y = np.load(labels) if labels.suffix == ".npy" else _load_frame(labels).to_numpy().ravel()
        names = numeric
    else:
        raise ValueError("pass --labels or --label-col so attribution has a target")
    x = frame.select(names).cast(float).to_numpy()
    if x.shape[0] != np.asarray(y).ravel().shape[0]:
        raise ValueError(
            f"feature rows ({x.shape[0]}) and label rows ({np.asarray(y).ravel().shape[0]}) differ"
        )
    if x.shape[0] < 16:
        raise ValueError(f"need at least 16 rows for attribution, got {x.shape[0]}")
    return x, np.asarray(y, dtype=float).ravel(), names


def _maybe_shap(shap: bool) -> bool:
    """Resolve --shap into include_shap, or exit 3 with a typed skip message."""
    if not shap:
        return False
    try:
        import shap as _shap  # noqa: F401
    except ImportError:
        typer.echo(
            "EXPLAINABILITY_SHAP_SKIP: the optional 'shap' extra is not installed "
            "(it is excluded on darwin/x86_64 — see pyproject); permutation "
            "attribution still runs. Install with: uv sync --extra explainability"
        )
        raise typer.Exit(code=_SHAP_UNAVAILABLE_EXIT) from None
    return True


def _write_and_echo(report: Any, out_dir: Path) -> None:
    from quant_fund.research.explainability import write_report

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = write_report(report, out_dir)
    for kind in ("markdown", "html", "json"):
        if kind in paths:
            typer.echo(f"{kind}={paths[kind]}")
    for key, path in sorted(paths.items()):
        if key not in ("markdown", "html", "json"):
            typer.echo(f"{key}={path}")


@explainability_app.command("demo")
def demo_cmd(
    seed: int = typer.Option(2026, help="SYNTHETIC panel seed."),
    n_rows: int = typer.Option(400, help="SYNTHETIC panel rows."),
    noise: float = typer.Option(0.3, help="Label noise scale."),
    scoring: str = typer.Option("pinball", help="Proper score: pinball[:tau], crps[:taus], brier."),
    top_k: int = typer.Option(8, help="Features to echo / include in partial dependence."),
    n_blocks: int = typer.Option(4, help="Time-ordered blocks for the drift statistic."),
    shap: bool = typer.Option(False, "--shap", help="Also run the optional SHAP explainer."),
    no_drift: bool = typer.Option(False, "--no-drift", help="Skip the attribution-drift block."),
    out_dir: Path = typer.Option(
        Path("reports/explainability"), help="Directory for markdown/html/json + sha256 sidecars."
    ),
) -> None:
    """Explain a least-squares head fitted on a seeded SYNTHETIC panel.

    Self-contained and offline: the panel plants all signal in ``signal_core``
    and none in ``noise_a``/``noise_b``, so a correct run ranks ``signal_core``
    first. Attribution is proper-score degradation, never accuracy or P&L.
    """
    from quant_fund.research.explainability import build_report

    include_shap = _maybe_shap(shap)
    try:
        x, y, names, model = _demo_panel(seed, n_rows, noise, taus=_demo_taus(scoring))
        report = build_report(
            model,
            x,
            y,
            feature_names=names,
            scoring=scoring,
            top_k=top_k,
            n_blocks=n_blocks,
            seed=seed,
            synthetic=True,
            model_name="demo_least_squares",
            model_family="synthetic_demo",
            include_shap=include_shap,
            drift=not no_drift,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(_data_label(synthetic=True, data_source="SYNTHETIC"))
    _echo_report(report, top_k=top_k)
    _write_and_echo(report, out_dir)


@explainability_app.command("report")
def report_cmd(
    model_path: Path = typer.Option(
        ..., exists=True, dir_okay=False, help="joblib-pickled fitted model with predict(x)."
    ),
    features: Path = typer.Option(
        ..., exists=True, dir_okay=False, help="Parquet/CSV feature matrix."
    ),
    labels: Path | None = typer.Option(
        None, dir_okay=False, help="Label vector (.npy, or single-column parquet/CSV)."
    ),
    label_col: str | None = typer.Option(
        None, help="Label column inside --features (alternative to --labels)."
    ),
    model_name: str | None = typer.Option(None, help="Override the reported model name."),
    model_family: str | None = typer.Option(None, help="Override the reported model family."),
    scoring: str = typer.Option("pinball", help="Proper score: pinball[:tau], crps[:taus], brier."),
    top_k: int = typer.Option(8, help="Features to echo / include in partial dependence."),
    n_blocks: int = typer.Option(4, help="Time-ordered blocks for the drift statistic."),
    seed: int = typer.Option(42, help="Permutation shuffle seed."),
    max_rows: int | None = typer.Option(None, help="Subsample cap for attribution."),
    shap: bool = typer.Option(False, "--shap", help="Also run the optional SHAP explainer."),
    no_drift: bool = typer.Option(False, "--no-drift", help="Skip the attribution-drift block."),
    out_dir: Path = typer.Option(
        Path("reports/explainability"), help="Directory for markdown/html/json + sha256 sidecars."
    ),
) -> None:
    """Explain a user-supplied fitted model on a user-supplied matrix.

    Rows are treated as time-ordered for the drift block. Output is labeled
    REAL because the inputs are the operator's own — that labels provenance, it
    is not evidence of live performance.
    """
    import joblib

    from quant_fund.research.explainability import build_report

    include_shap = _maybe_shap(shap)
    try:
        x, y, names = _load_xy(features, labels, label_col)
        model = joblib.load(model_path)
        report = build_report(
            model,
            x,
            y,
            feature_names=names,
            scoring=scoring,
            top_k=top_k,
            n_blocks=n_blocks,
            seed=seed,
            synthetic=False,
            model_name=model_name,
            model_family=model_family,
            include_shap=include_shap,
            drift=not no_drift,
            max_rows=max_rows,
        )
    except (ValueError, TypeError, OSError, AttributeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(_data_label(synthetic=False, data_source=features.name))
    _echo_report(report, top_k=top_k)
    _write_and_echo(report, out_dir)


@explainability_app.command("attach-sidecar")
def attach_sidecar_cmd(
    receipt: Path = typer.Argument(..., exists=True, dir_okay=False, help="Sealed receipt JSON."),
    reports: Path = typer.Argument(
        ..., exists=True, help="Report directory, or a single report file."
    ),
) -> None:
    """Bind explainability reports to a sealed receipt, additively.

    Writes ``<receipt stem>.explainability.json`` beside the receipt (see
    :func:`explainability_sidecar_path`). The receipt file is opened read-only
    and never modified — the sidecar is evidence *about* the receipt, not a
    reseal of it.

    Reports must live under the receipt's own directory tree: the sidecar binds
    them by path relative to the receipt root, so a report outside it is
    rejected as ``report_outside_receipt_root``. Generate into a sibling
    directory, e.g. ``explain demo --out-dir receipts/run42/explainability``
    then ``explain attach-sidecar receipts/run42/receipt.json
    receipts/run42/explainability``.
    """
    from quant_fund.research.explainability import attach_explainability_sidecar

    try:
        path = attach_explainability_sidecar(receipt, reports)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"sidecar={path}")


@explainability_app.command("verify-sidecar")
def verify_sidecar_cmd(
    receipt: Path = typer.Argument(..., exists=True, dir_okay=False, help="Sealed receipt JSON."),
    sidecar: Path | None = typer.Argument(
        None, dir_okay=False, help="Sidecar path (default: <receipt>.explainability.json)."
    ),
) -> None:
    """Re-check that a sidecar still binds the report bytes on disk.

    Additive check only: it says nothing about the receipt's own validity
    beyond hash binding. Exits non-zero on any error.
    """
    import json

    from quant_fund.research.explainability import verify_explainability_sidecar

    result = verify_explainability_sidecar(receipt, sidecar)
    typer.echo(json.dumps(result, indent=2))
    raise typer.Exit(code=0 if result["valid"] else 1)
