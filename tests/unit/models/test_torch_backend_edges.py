"""robinhood+ torch backend edge paths: gating errors, predictor cache,
future-stamp edges, and the cross-section forecast matrix. No torch/Kronos
weights are loaded — every heavyweight object is stubbed."""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.models.robinhood_plus import torch_backend

pytestmark = pytest.mark.synthetic


def _config(**overrides: Any) -> AppConfig:
    rh: dict[str, Any] = {"backend": "torch", "lookback": 8, "pred_len": 3, "sample_count": 2}
    rh.update(overrides.pop("robinhood_plus", {}))
    return AppConfig.model_validate(
        {
            "data": {"root": "/tmp/x", "source": "synthetic"},
            "paper": {"enable_shadow": False},
            "robinhood_plus": rh,
            **overrides,
        }
    )


class TestAvailabilityAndPaths:
    def test_torch_available_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
        assert torch_backend.torch_available() is False

    def test_local_dirs_none_for_bad_variant(self) -> None:
        assert torch_backend.local_kronos_weight_dirs("bogus") is None

    def test_local_dirs_none_when_missing(self) -> None:
        assert torch_backend.local_kronos_weight_dirs("mini") is None

    def test_ensure_kronos_on_path_with_checkout(self, tmp_path: Path) -> None:
        # third_party/kronos ships with the repo, so sys.path gains the entry.
        before = list(sys.path)
        try:
            torch_backend._ensure_kronos_on_path()
            root = Path(torch_backend.__file__).resolve().parents[4] / "third_party"
            for name in ("kronos_src", "Kronos", "kronos"):
                if (root / name / "model" / "kronos.py").is_file():
                    assert str(root / name) in sys.path
                    break
        finally:
            sys.path[:] = before


class TestLoadPredictor:
    def test_unknown_variant_rejected(self) -> None:
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="unknown"):
            torch_backend.load_pretrained_predictor("bogus")

    def test_hub_download_refused_without_network(self) -> None:
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="allow_network"):
            torch_backend.load_pretrained_predictor("mini")

    def test_missing_torch_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch_backend, "torch_available", lambda: False)
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="torch"):
            torch_backend.load_pretrained_predictor(
                "mini", tokenizer_path=str(tmp_path), model_path=str(tmp_path)
            )

    def test_kronos_not_importable_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Intent: an ImportError from the Kronos classes must be wrapped in
        # RobinhoodPlusTorchError, not propagate. The original premise —
        # third_party/kronos's deps (huggingface_hub/torch) absent from the
        # test env — no longer holds after `uv sync --all-extras` installs the
        # [nn] extra, so the un-importability is simulated deterministically:
        # a None sys.modules entry makes `from model.kronos import ...` raise
        # ImportError on every machine, and torch_available is forced True so
        # the run reaches the import branch (the torch-absent branch has its
        # own test above).
        monkeypatch.setattr(torch_backend, "torch_available", lambda: True)
        monkeypatch.setitem(sys.modules, "model.kronos", None)
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="Kronos"):
            torch_backend.load_pretrained_predictor(
                "mini", tokenizer_path=str(tmp_path), model_path=str(tmp_path)
            )

    def test_happy_path_via_stub_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, Any] = {}

        class _Tok:
            @classmethod
            def from_pretrained(cls, path: str) -> str:
                calls["tok"] = path
                return "tok-obj"

        class _Model:
            @classmethod
            def from_pretrained(cls, path: str) -> str:
                calls["model"] = path
                return "model-obj"

        class _Predictor:
            def __init__(self, model: Any, tokenizer: Any, **kw: Any) -> None:
                calls["predictor"] = (model, tokenizer, kw)

        fake = types.ModuleType("model.kronos")
        fake.Kronos = _Model  # type: ignore[attr-defined]
        fake.KronosTokenizer = _Tok  # type: ignore[attr-defined]
        fake.KronosPredictor = _Predictor  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "model", types.ModuleType("model"))
        monkeypatch.setitem(sys.modules, "model.kronos", fake)
        torch_backend.load_pretrained_predictor(
            "mini",
            tokenizer_path=str(tmp_path),
            model_path=str(tmp_path),
            device="cpu",
            max_context=64,
        )
        assert calls["predictor"][2]["max_context"] == 64
        assert calls["predictor"][2]["device"] == "cpu"


class TestPredictorCache:
    def test_cache_hit_and_eviction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch_backend._PREDICTOR_CACHE.clear()
        loads: list[str] = []

        def _fake_load(variant: str, **kw: Any) -> str:
            loads.append(variant)
            return f"pred-{variant}"

        monkeypatch.setattr(torch_backend, "load_pretrained_predictor", _fake_load)
        cfg = _config()
        assert torch_backend._cached_predictor(cfg) == "pred-mini"
        assert torch_backend._cached_predictor(cfg) == "pred-mini"
        assert loads == ["mini"]
        for i in range(5):
            alt = _config(robinhood_plus={"tokenizer_path": f"/t{i}"})
            torch_backend._cached_predictor(alt)
        assert len(torch_backend._PREDICTOR_CACHE) <= 5
        torch_backend._PREDICTOR_CACHE.clear()


class TestFutureStamps:
    def test_degenerate_and_regular_deltas(self) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        assert torch_backend._future_stamps([], 2) == [
            datetime(2020, 1, 3, tzinfo=UTC),
            datetime(2020, 1, 4, tzinfo=UTC),
        ]
        same = [t0, t0]
        out = torch_backend._future_stamps(same, 2)
        assert out[0] - same[-1] == timedelta(days=1)
        reg = [t0, t0 + timedelta(hours=4)]
        assert torch_backend._future_stamps(reg, 2)[0] == t0 + timedelta(hours=8)


class _StubPredictor:
    def __init__(self, rows: int = 3) -> None:
        self.rows = rows

    def predict(self, df: Any, **kw: Any) -> pd.DataFrame:
        cols = ["open", "high", "low", "close", "volume", "amount"]
        return pd.DataFrame(np.ones((self.rows, 6)), columns=cols)


class TestKlinePaths:
    def test_paths_stacked_per_sample(self) -> None:
        kline = np.ones((5, 6))
        times = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=d) for d in range(5)]
        paths = torch_backend._kline_paths_from_kronos(
            _StubPredictor(3),
            kline,
            times,
            pred_len=3,
            sample_count=2,
            temperature=1.0,
            top_p=0.9,
        )
        assert paths.shape == (2, 3, 6)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="shape"):
            torch_backend._kline_paths_from_kronos(
                _StubPredictor(2),
                np.ones((5, 6)),
                [datetime(2024, 1, 1, tzinfo=UTC)] * 5,
                pred_len=3,
                sample_count=1,
                temperature=1.0,
                top_p=0.9,
            )


def _frame(n: int = 10, sids: tuple[str, ...] = ("AAA",)) -> pl.DataFrame:
    times = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=d) for d in range(n)]
    rows = []
    for sid in sids:
        for t in times:
            rows.append(
                {
                    "security_id": sid,
                    "event_time": t,
                    "open_split_adjusted": 1.0,
                    "high_split_adjusted": 1.1,
                    "low_split_adjusted": 0.9,
                    "close_split_adjusted": 1.0,
                    "volume": 1e6,
                    "amount": 1e6,
                }
            )
    return pl.DataFrame(rows)


class TestCrossSection:
    def test_missing_kline_columns_returns_empty(self) -> None:
        assert (
            torch_backend.forecast_cross_section_torch(
                pl.DataFrame({"event_time": [], "security_id": []}),
                datetime(2024, 1, 5, tzinfo=UTC),
                _config(),
                ["AAA"],
            )
            == {}
        )

    def test_missing_event_time_raises(self) -> None:
        frame = pl.DataFrame(
            {
                "open_split_adjusted": [1.0],
                "high_split_adjusted": [1.0],
                "low_split_adjusted": [1.0],
                "close_split_adjusted": [1.0],
                "volume": [1.0],
            }
        )
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="event_time"):
            torch_backend.forecast_cross_section_torch(
                frame, datetime(2024, 1, 5, tzinfo=UTC), _config(), ["AAA"]
            )

    def test_insufficient_history_marks_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch_backend, "_cached_predictor", lambda _c: _StubPredictor())
        frame = _frame(1)
        out = torch_backend.forecast_cross_section_torch(
            frame, datetime(2025, 1, 1, tzinfo=UTC), _config(), ["AAA"]
        )
        assert set(out) == {"AAA"}

    def test_inference_failure_wraps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Boom:
            def predict(self, *_a: Any, **_k: Any) -> Any:
                raise RuntimeError("cuda oom")

        monkeypatch.setattr(torch_backend, "_cached_predictor", lambda _c: _Boom())
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="cuda oom"):
            torch_backend.forecast_cross_section_torch(
                _frame(12), datetime(2025, 1, 1, tzinfo=UTC), _config(), ["AAA"]
            )

    def test_happy_path_returns_forecast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch_backend, "_cached_predictor", lambda _c: _StubPredictor(3))
        out = torch_backend.forecast_cross_section_torch(
            _frame(12, ("AAA", "ZZZ")),
            datetime(2025, 1, 1, tzinfo=UTC),
            _config(),
            ["AAA", "ZZZ"],
        )
        assert set(out) == {"AAA", "ZZZ"}


# Offline Hub cache holding tiny real Kronos weights ONLY under the pinned
# VARIANT_HUB commits (no refs/main): a loader that resolves a moving branch
# cannot find them, and HF_HUB_OFFLINE=1 keeps the run off the network.
_PINNED_HUB_LOAD = """
import json, os, sys
from pathlib import Path
from quant_fund.models.robinhood_plus import torch_backend
from quant_fund.models.robinhood_plus.constants import VARIANT_HUB
torch_backend._ensure_kronos_on_path()
from model.kronos import Kronos, KronosTokenizer
cache = Path(os.environ["HF_HUB_CACHE"])
spec = VARIANT_HUB["mini"]
tok = KronosTokenizer(6, 8, 2, 16, 1, 1, 0.0, 0.0, 0.0, 2, 2, 0.05, 1.0, 1.1, 0.05, 2)
mdl = Kronos(2, 2, 1, 8, 2, 16, 0.0, 0.0, 0.0, 0.0, True)
for obj, repo, rev in ((tok, spec["tokenizer"], spec["tokenizer_rev"]),
                       (mdl, spec["model"], spec["model_rev"])):
    obj.save_pretrained(str(cache / ("models--" + repo.replace("/", "--")) / "snapshots" / rev))
pred = torch_backend.load_pretrained_predictor("mini", allow_network=True, device="cpu")
print(json.dumps([type(pred.tokenizer).__name__, type(pred.model).__name__]))
"""

# A regular FILE at a repo-id-shaped relative path is not a local source:
# huggingface_hub's ModelHubMixin reads only os.path.isdir() sources from disk
# and fetches anything else as a Hub repo id. The offline cache serves a
# d_model=16 pair under refs/main (a moving branch, commit "b"*40) and a
# d_model=8 pair only at the pinned VARIANT_HUB commits; the cwd holds empty
# regular files at every repo-id path.
_FILE_SHADOW_LOAD = """
import json, os, sys
from pathlib import Path
from quant_fund.models.robinhood_plus import torch_backend as tb
from quant_fund.models.robinhood_plus.constants import VARIANT_HUB
tb._ensure_kronos_on_path()
from model.kronos import Kronos, KronosTokenizer
cache, spec = Path(os.environ["HF_HUB_CACHE"]), VARIANT_HUB["mini"]
def put(repo, rev, d_model, main=False):
    if "Tokenizer" in repo or repo.endswith("-tok"):
        obj = KronosTokenizer(6, d_model, 2, 16, 1, 1, 0.0, 0.0, 0.0, 2, 2, 0.05, 1.0, 1.1, 0.05, 2)
    else:
        obj = Kronos(2, 2, 1, d_model, 2, 16, 0.0, 0.0, 0.0, 0.0, True)
    base = cache / ("models--" + repo.replace("/", "--"))
    obj.save_pretrained(str(base / "snapshots" / rev))
    if main:
        (base / "refs").mkdir(parents=True, exist_ok=True)
        (base / "refs" / "main").write_text(rev)
    Path(repo).parent.mkdir(parents=True, exist_ok=True)
    Path(repo).write_text("")
put(spec["tokenizer"], spec["tokenizer_rev"], 8)
put(spec["model"], spec["model_rev"], 8)
for repo in (spec["tokenizer"], spec["model"], "someone/fork-tok", "someone/fork-model"):
    put(repo, "b" * 40, 16, main=True)
def outcome(fn, **kw):
    try:
        out = fn("mini", **kw)
    except tb.RobinhoodPlusTorchError as exc:
        return "refused: " + str(exc)
    if isinstance(out, dict):
        return out
    return ["loaded", out.tokenizer.d_model, out.model.d_model]
fork = {"tokenizer_path": "someone/fork-tok", "model_path": "someone/fork-model"}
print(json.dumps({
    "pinned_revisions": outcome(tb.kronos_weight_revisions),
    "pinned_load": outcome(tb.load_pretrained_predictor, allow_network=True, device="cpu"),
    "pinned_offline": outcome(tb.load_pretrained_predictor, allow_network=False),
    "fork_revisions": outcome(tb.kronos_weight_revisions, **fork),
    "fork_load": outcome(tb.load_pretrained_predictor, allow_network=True, **fork),
    "fork_offline": outcome(tb.load_pretrained_predictor, allow_network=False, **fork),
}))
"""


def _offline_hub_run(tmp_path: Path, script: str) -> Any:
    """Run ``script`` in a scratch cwd against an empty offline Hub cache."""
    import json
    import os
    import subprocess

    cache = tmp_path / "hub"
    cache.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_CACHE": str(cache)}
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestHubRevisionPins:
    def test_hub_fetch_loads_the_pinned_commit(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        assert _offline_hub_run(tmp_path, _PINNED_HUB_LOAD) == ["KronosTokenizer", "Kronos"]

    def test_regular_file_at_hub_id_path_is_not_local(self, tmp_path: Path) -> None:
        # A stray file at the repo-id path must neither drop the pin (load
        # refs/main) nor be recorded as "local", and must not get a foreign
        # id or an allow_network=false run past the guards.
        pytest.importorskip("torch")
        from quant_fund.models.robinhood_plus.constants import VARIANT_HUB

        spec = VARIANT_HUB["mini"]
        out = _offline_hub_run(tmp_path, _FILE_SHADOW_LOAD)
        assert out["pinned_revisions"] == {
            "tokenizer_revision": spec["tokenizer_rev"],
            "model_revision": spec["model_rev"],
        }
        assert out["pinned_load"] == ["loaded", 8, 8]  # the pinned pair, not refs/main
        for key in ("pinned_offline", "fork_offline"):
            assert str(out[key]).startswith("refused: torch backend refused Hub download"), out
        unpinned = "refused: refusing unpinned Hub fetch of 'someone/"
        for key in ("fork_revisions", "fork_load"):
            assert str(out[key]).startswith(unpinned), out

    def test_unpinned_hub_id_refused(self) -> None:
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="unpinned Hub fetch"):
            torch_backend.load_pretrained_predictor(
                "mini", tokenizer_path="someone/unpinned-fork", allow_network=True
            )
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="unpinned Hub fetch"):
            torch_backend.load_pretrained_predictor(
                "mini", model_path="someone/unpinned-fork", allow_network=True
            )

    def test_forecast_diagnostics_record_revisions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from quant_fund.models.robinhood_plus.constants import VARIANT_HUB

        monkeypatch.setattr(torch_backend, "_cached_predictor", lambda _c: _StubPredictor(3))
        asof = datetime(2025, 1, 1, tzinfo=UTC)
        hub = torch_backend.forecast_cross_section_torch(_frame(12), asof, _config(), ["AAA"])
        diag = hub["AAA"].forecast.diagnostics
        assert diag["tokenizer_revision"] == VARIANT_HUB["mini"]["tokenizer_rev"]
        assert diag["model_revision"] == VARIANT_HUB["mini"]["model_rev"]
        local_cfg = _config(
            robinhood_plus={"tokenizer_path": str(tmp_path), "model_path": str(tmp_path)}
        )
        local = torch_backend.forecast_cross_section_torch(_frame(12), asof, local_cfg, ["AAA"])
        assert local["AAA"].forecast.diagnostics["model_revision"] == "local"
        assert local["AAA"].forecast.diagnostics["tokenizer_revision"] == "local"
        # A regular file is a Hub id to huggingface_hub, never "local".
        weights = tmp_path / "kronos.safetensors"
        weights.write_bytes(b"")
        file_cfg = _config(
            robinhood_plus={"tokenizer_path": str(weights), "model_path": str(weights)}
        )
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="unpinned Hub fetch"):
            torch_backend.forecast_cross_section_torch(_frame(12), asof, file_cfg, ["AAA"])
