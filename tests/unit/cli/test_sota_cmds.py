"""CLI bridge for the SOTA and forward-evidence lanes.

``prospective_sota`` and ``forward_shadow`` were reachable only as
``python -m`` argparse entry points — invisible to ``dipcatcher --help`` and
untyped at the boundary. ``sota_protocol`` and ``sota_evidence`` were reachable
only as in-process imports from the ``scripts/*_col.py`` collectors. These tests
drive the new groups end-to-end and pin the honesty contract: prospective clocks
are controlled fixtures, every result is a JSON evidence document, and nothing
here asserts a live or market-performance outcome.

The ``python -m`` mains and their subprocess contract tests are untouched — the
CLI calls the same library functions, so the two surfaces cannot drift.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.research import forward_shadow as shadow
from quant_fund.research import prospective_sota as journal

runner = CliRunner()

_HEX_A = "a" * 64
_HEX_B = "b" * 64
_HEX_C = "c" * 64
_HEX_D = "d" * 64
_HEX_E = "e" * 64
ORIGIN = datetime(2026, 10, 1, tzinfo=UTC)

GROUPS = {
    "prospective-sota": [
        "commitment",
        "prepare",
        "forecast",
        "settle",
        "interrupt",
        "verify",
    ],
    "forward-shadow": ["freeze", "decide", "miss", "settle", "reconcile", "dump", "plan"],
    "sota": ["show", "run", "validate-bars"],
}


@pytest.mark.parametrize("group", sorted(GROUPS))
def test_group_help_lists_every_subcommand(group: str) -> None:
    result = runner.invoke(app, [group, "--help"])
    assert result.exit_code == 0, result.output
    for sub in GROUPS[group]:
        assert sub in result.output


@pytest.mark.parametrize(
    "argv", [(g, s) for g, subs in sorted(GROUPS.items()) for s in subs], ids=lambda a: str(a)
)
def test_subcommand_help(argv: tuple[str, str]) -> None:
    result = runner.invoke(app, [*argv, "--help"])
    assert result.exit_code == 0, f"{' '.join(argv)} --help exited {result.exit_code}"
    assert result.output.strip()


def _protocol() -> dict:
    return {
        "schema": journal.SCHEMA,
        "source_id": "synthetic_closes",
        "asset_ids": ["A", "B"],
        "bar_interval_seconds": 86400,
        "history_returns": 750,
        "first_origin_time": ORIGIN.isoformat(),
        "candidate": {
            "model_id": "new_candidate",
            "artifact_sha256": _HEX_A,
            "adapter_sha256": _HEX_B,
        },
        "published": {
            "model_id": "published_model_v1",
            "artifact_sha256": _HEX_C,
            "adapter_sha256": _HEX_D,
            "canonical_reference": "doi:synthetic-fixture-only",
        },
        "minimum_paired_origins": 10,
        "planning_effect_crps": 1.0,
        "planning_long_run_sd_crps": 0.01,
        "planning_source_sha256": _HEX_E,
        "alpha": 0.05,
        "power": 0.8,
        "hac_lags": 2,
        "coverage_rule": "all_assets_all_models_or_block",
        "primary_metric": "equal_weight_asset_crps_by_target_date",
    }


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_prospective_commitment_binds_the_protocol(tmp_path: Path) -> None:
    protocol = _write(tmp_path / "protocol.json", _protocol())
    result = runner.invoke(app, ["prospective-sota", "commitment", str(protocol)])
    assert result.exit_code == 0, result.output
    assert f"DATA_LABEL={protocol.name}" in result.output

    body = json.loads(result.output.split("DATA_LABEL=", 1)[1].split("\n", 1)[1])
    assert len(body["commitment_sha256"]) == 64
    assert len(body["protocol_sha256"]) == 64
    # The commitment binds the implementation, not just the protocol text.
    assert "journal" in body["source_sha256"]

    # Same protocol, same binding — the digest is reproducible.
    again = runner.invoke(app, ["prospective-sota", "commitment", str(protocol)])
    assert again.exit_code == 0, again.output
    assert body["commitment_sha256"] in again.output


def test_prospective_commitment_output_can_be_written(tmp_path: Path) -> None:
    protocol = _write(tmp_path / "protocol.json", _protocol())
    out = tmp_path / "commitment.json"
    result = runner.invoke(
        app, ["prospective-sota", "commitment", str(protocol), "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert f"written={out}" in result.output
    assert len(json.loads(out.read_text())["commitment_sha256"]) == 64


def test_prospective_prepare_then_verify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A journal prepared before the first origin verifies as a chain."""
    monkeypatch.setattr(journal, "_clock", lambda _now=None: ORIGIN - timedelta(days=1))
    protocol_path = _write(tmp_path / "protocol.json", _protocol())
    commitment = journal.commitment(_protocol())
    anchor = _write(
        tmp_path / "anchor.json",
        {
            "recorded_at": (ORIGIN - timedelta(days=2)).isoformat(),
            "issuer": "synthetic-test",
            "reference": "synthetic-fixture-anchor",
            "commitment_sha256": commitment["commitment_sha256"],
        },
    )
    run = tmp_path / "run"

    prepared = runner.invoke(
        app,
        ["prospective-sota", "prepare", str(run), str(protocol_path), str(anchor)],
    )
    assert prepared.exit_code == 0, prepared.output
    assert run.is_dir()

    verified = runner.invoke(app, ["prospective-sota", "verify", str(run)])
    assert verified.exit_code == 0, verified.output


def test_prospective_interrupt_requires_a_reason(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    result = runner.invoke(app, ["prospective-sota", "interrupt", str(run), "   "])
    assert result.exit_code != 0
    assert "non-empty reason" in result.output


def test_prospective_commitment_rejects_a_malformed_protocol(tmp_path: Path) -> None:
    protocol = _write(tmp_path / "protocol.json", {"schema": journal.SCHEMA})
    result = runner.invoke(app, ["prospective-sota", "commitment", str(protocol)])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


@pytest.fixture
def shadow_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict, Path]:
    """A frozen-spec shadow journal built on a controlled SYNTHETIC clock."""
    start = datetime(2030, 3, 1, 16, tzinfo=UTC)
    monkeypatch.setattr(shadow, "_now", lambda: start - timedelta(hours=1))
    rng = np.random.default_rng(41)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, (30, 4)), axis=0))
    dates = [start + timedelta(days=i - 20) for i in range(30)]
    bars = [
        {
            "event_time": t.isoformat(),
            "available_time": t.isoformat(),
            "close": prices[i].tolist(),
            "volume": [1e6] * 4,
        }
        for i, t in enumerate(dates)
    ]
    spec = {
        "strategy": {"name": "momentum", "family": "momentum"},
        "benchmark": {"name": "equal", "family": "equal_weight"},
        "execution": {},
        "assets": ["a", "b", "c", "d"],
        "source_url": "https://example.org/generated-fixture",
        "price_basis": "raw_price_return",
        "selection_basis": "generated contract test",
        "lead_seconds": 60,
        "sessions": [
            {
                "signal_time": dates[i].isoformat(),
                "execution_time": (dates[i + 1] - timedelta(hours=1)).isoformat(),
            }
            for i in range(20, 29)
        ],
        "calibration": {
            "end_time": dates[19].isoformat(),
            "source": "generated calibration",
            "net_differences": rng.normal(0, 0.001, 100).tolist(),
        },
        "evidence": {"effect_bps": 5.0, "lag": 3},
    }
    spec_path = _write(tmp_path / "spec.json", spec)
    bootstrap_path = _write(tmp_path / "bootstrap.json", bars[:20])
    return spec_path, spec, bootstrap_path


def test_forward_shadow_freeze_dump_reconcile(
    tmp_path: Path, shadow_spec: tuple[Path, dict, Path]
) -> None:
    spec_path, _spec, bootstrap_path = shadow_spec
    run = tmp_path / "forward.sqlite"

    frozen = runner.invoke(
        app,
        [
            "forward-shadow",
            "freeze",
            "--spec",
            str(spec_path),
            "--bootstrap",
            str(bootstrap_path),
            "--run",
            str(run),
        ],
    )
    assert frozen.exit_code == 0, frozen.output
    assert f"DATA_LABEL={spec_path.name}" in frozen.output
    assert run.is_file()

    dumped = runner.invoke(app, ["forward-shadow", "dump", str(run)])
    assert dumped.exit_code == 0, dumped.output
    events = json.loads(dumped.output)
    assert isinstance(events, list) and events

    reconciled = runner.invoke(
        app,
        ["forward-shadow", "reconcile", "--run", str(run), "--output", str(tmp_path / "r.json")],
    )
    assert reconciled.exit_code == 0, reconciled.output
    report = json.loads((tmp_path / "r.json").read_text())
    # The heavy per-event bodies stay out of the summary.
    assert "events" not in report and "state" not in report


def test_forward_shadow_reconcile_detects_a_forged_head(
    tmp_path: Path, shadow_spec: tuple[Path, dict, Path]
) -> None:
    spec_path, _spec, bootstrap_path = shadow_spec
    run = tmp_path / "forward.sqlite"
    frozen = runner.invoke(
        app,
        [
            "forward-shadow",
            "freeze",
            "--spec",
            str(spec_path),
            "--bootstrap",
            str(bootstrap_path),
            "--run",
            str(run),
        ],
    )
    assert frozen.exit_code == 0, frozen.output

    result = runner.invoke(
        app,
        ["forward-shadow", "reconcile", "--run", str(run), "--expected-head", "f" * 64],
    )
    assert result.exit_code != 0


def test_forward_shadow_plan_sizes_the_experiment(
    tmp_path: Path, shadow_spec: tuple[Path, dict, Path]
) -> None:
    """The HAC power plan must state its own assumption and floor the sample."""
    _spec_path, spec, _bootstrap = shadow_spec
    calibration = _write(tmp_path / "calibration.json", spec["calibration"]["net_differences"])
    result = runner.invoke(
        app,
        [
            "forward-shadow",
            "plan",
            "--calibration",
            str(calibration),
            "--effect-bps",
            "5.0",
            "--lag",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.output.split("DATA_LABEL=", 1)[1].split("\n", 1)[1])
    assert body["required_sessions"] >= 30
    assert body["method"] == "one_sided_normal_approximation_with_Bartlett_HAC"
    assert "no power guarantee" in body["assumption"]
    assert body["long_run_variance"] > 0.0


def test_forward_shadow_plan_rejects_an_invalid_effect(tmp_path: Path) -> None:
    calibration = _write(tmp_path / "calibration.json", [0.001] * 50)
    result = runner.invoke(
        app,
        [
            "forward-shadow",
            "plan",
            "--calibration",
            str(calibration),
            "--effect-bps",
            "0",
            "--lag",
            "2",
        ],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_sota_show_fingerprints_the_committed_protocol(tmp_path: Path) -> None:
    protocol = Path("configs/sota_protocol.yaml")
    if not protocol.is_file():
        pytest.skip("configs/sota_protocol.yaml missing")
    result = runner.invoke(app, ["sota", "show", str(protocol)])
    assert result.exit_code == 0, result.output
    line = next(line for line in result.output.splitlines() if line.startswith("protocol_sha256="))
    digest = line.split("=", 1)[1]
    assert len(digest) == 64

    from quant_fund.research.sota_protocol import load_sota_protocol, protocol_sha256

    assert digest == protocol_sha256(load_sota_protocol(protocol))


def test_sota_show_rejects_an_unfrozen_or_unknown_field(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("protocol_id: x\nnot_a_real_field: 1\n")
    result = runner.invoke(app, ["sota", "show", str(bad)])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def _bars(n: int = 40, *, available_offset_seconds: int = 1) -> pl.DataFrame:
    """A panel that satisfies ``validate_bars``: one security, regular spacing,
    causal availability, and internally consistent OHLC."""
    step = timedelta(days=1)
    close = np.array([100.0 + i for i in range(n)])
    return pl.DataFrame(
        {
            "security_id": ["A"] * n,
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC) + step * i for i in range(n)],
            "available_time": [
                datetime(2026, 1, 1, tzinfo=UTC)
                + step * i
                + timedelta(seconds=available_offset_seconds)
                for i in range(n)
            ],
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 5_000.0),
        }
    )


def test_sota_validate_bars_accepts_a_clean_panel(tmp_path: Path) -> None:
    frame = _bars()
    from quant_fund.research.sota_evidence import validate_bars

    _times, interval = validate_bars(frame)
    assert interval == 86_400 * 10**9  # one day, in nanoseconds

    path = tmp_path / "bars.parquet"
    frame.write_parquet(path)
    result = runner.invoke(app, ["sota", "validate-bars", str(path)])
    assert result.exit_code == 0, result.output
    assert f"DATA_LABEL={path.name}" in result.output
    assert "bars=40" in result.output
    assert "bar_interval_ns=86400000000000" in result.output


def test_sota_validate_bars_rejects_a_point_in_time_violation(tmp_path: Path) -> None:
    """A bar available only after the next bar opens must fail the gate."""
    frame = _bars(available_offset_seconds=2 * 86_400)
    path = tmp_path / "late.parquet"
    frame.write_parquet(path)
    result = runner.invoke(app, ["sota", "validate-bars", str(path)])
    assert result.exit_code != 0
    assert "point-in-time violation" in result.output
    assert "Traceback" not in result.output


def test_sota_validate_bars_rejects_inconsistent_ohlc(tmp_path: Path) -> None:
    """``high`` below ``close`` breaks the bar identity, so the gate must fail."""
    frame = _bars().with_columns((pl.col("close") - 5.0).alias("high"))
    path = tmp_path / "bad_ohlc.parquet"
    frame.write_parquet(path)
    result = runner.invoke(app, ["sota", "validate-bars", str(path)])
    assert result.exit_code != 0
    assert "inconsistent OHLC" in result.output


# --------------------------------------------------------------------------- #
# sota effects / block-sensitivity — paired inference over a loss panel
# --------------------------------------------------------------------------- #


def _loss_panel(tmp_path: Path, n: int = 120, seed: int = 11) -> Path:
    """A chronological panel where ``cand`` is planted strictly better.

    ``cand = 0.80 * base`` and ``weak = 1.25 * base`` share the same ``base``
    draw, so the relative CRPS reduction of cand against weak is exactly
    ``(0.80 - 1.25) / 0.80`` — a closed form the test asserts rather than a
    tolerance around a simulation.
    """
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 0.5, n)
    frame = pl.DataFrame(
        {
            "cand": base * 0.80,
            "peer": base * 1.00 + rng.normal(0.0, 0.01, n),
            "weak": base * 1.25,
        }
    )
    path = tmp_path / "losses.parquet"
    frame.write_parquet(path)
    return path


def _json_after_label(output: str) -> object:
    """Parse the JSON document a command prints after its DATA_LABEL/summary lines."""
    lines = output.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(("{", "["))),
        None,
    )
    assert start is not None, f"no JSON document in output:\n{output}"
    return json.loads("\n".join(lines[start:]))


def test_sota_effects_recovers_the_planted_reduction(tmp_path: Path) -> None:
    panel = _loss_panel(tmp_path)
    result = runner.invoke(
        app,
        ["sota", "effects", "--losses", str(panel), "--targets", "cand", "--n-boot", "300"],
    )
    assert result.exit_code == 0, result.output
    assert f"DATA_LABEL={panel.name}" in result.output
    assert "models=3 targets=1 rows=120" in result.output

    body = _json_after_label(result.output)
    assert body["status"] == "computed"
    assert body["sign"] == "positive_favors_comparator"
    comparisons = body["comparisons"]["cand"]

    # Closed form: (0.80 - 1.25) / 0.80 = -0.5625 exactly.
    assert comparisons["weak"]["relative_crps_reduction"] == pytest.approx(-0.5625, abs=1e-12)
    # cand beats both comparators, so target-minus-comparator is negative.
    assert comparisons["weak"]["mean_target_minus_comparator"] < 0.0
    assert comparisons["peer"]["mean_target_minus_comparator"] < 0.0


def test_sota_effects_simultaneous_interval_is_at_least_as_wide(tmp_path: Path) -> None:
    """Simultaneous intervals pay for the family, so they cannot be narrower."""
    panel = _loss_panel(tmp_path)
    result = runner.invoke(
        app,
        ["sota", "effects", "--losses", str(panel), "--targets", "cand", "--n-boot", "300"],
    )
    assert result.exit_code == 0, result.output
    body = _json_after_label(result.output)
    assert body["family_size"] == 2
    for comparator in body["comparisons"]["cand"].values():
        pointwise = comparator
        if not isinstance(pointwise, dict):
            continue
        pw_lo, pw_hi = pointwise["pointwise_percentile_ci"]
        sm_lo, sm_hi = pointwise["simultaneous_ci"]
        assert (sm_hi - sm_lo) >= (pw_hi - pw_lo) - 1e-12


def test_sota_effects_requires_two_models(tmp_path: Path) -> None:
    single = tmp_path / "one.parquet"
    pl.DataFrame({"only": [1.0, 2.0, 3.0]}).write_parquet(single)
    result = runner.invoke(app, ["sota", "effects", "--losses", str(single), "--targets", "only"])
    assert result.exit_code != 0
    assert "at least 2 numeric loss columns" in result.output


def test_sota_effects_rejects_an_unknown_target(tmp_path: Path) -> None:
    panel = _loss_panel(tmp_path)
    result = runner.invoke(
        app, ["sota", "effects", "--losses", str(panel), "--targets", "not_a_model"]
    )
    assert result.exit_code != 0
    assert "not_a_model" in result.output
    assert "cand" in result.output  # remediation lists the panel's columns


def test_sota_effects_rejects_negative_losses(tmp_path: Path) -> None:
    path = tmp_path / "negative.parquet"
    pl.DataFrame({"a": [-1.0, 2.0, 3.0] * 10, "b": [1.0, 2.0, 3.0] * 10}).write_parquet(path)
    result = runner.invoke(app, ["sota", "effects", "--losses", str(path), "--targets", "a"])
    assert result.exit_code != 0
    assert "nonnegative" in result.output


def test_sota_block_sensitivity_reports_every_declared_length(tmp_path: Path) -> None:
    panel = _loss_panel(tmp_path)
    result = runner.invoke(
        app,
        [
            "sota",
            "block-sensitivity",
            "--losses",
            str(panel),
            "--targets",
            "cand",
            "--blocks",
            "1,5,20",
            "--n-boot",
            "200",
        ],
    )
    assert result.exit_code == 0, result.output
    # Regression: the echo used to join the raw option string character-wise.
    assert "blocks=1,5,20" in result.output
    assert ",,," not in result.output

    body = _json_after_label(result.output)
    assert isinstance(body, list)
    assert [entry["block_length"] for entry in body] == [1.0, 5.0, 20.0]
    assert body[0]["resampling"] == "iid"
    # The planted winner survives MCS at every declared length.
    for entry in body:
        assert entry["mcs_included"]["cand"] is True


def test_sota_block_sensitivity_rejects_bad_blocks(tmp_path: Path) -> None:
    panel = _loss_panel(tmp_path)
    result = runner.invoke(
        app,
        [
            "sota",
            "block-sensitivity",
            "--losses",
            str(panel),
            "--targets",
            "cand",
            "--blocks",
            "1,-5",
        ],
    )
    assert result.exit_code != 0
    assert "--blocks must be positive" in result.output
