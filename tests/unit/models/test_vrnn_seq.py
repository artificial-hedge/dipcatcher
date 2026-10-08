import numpy as np
import pytest

torch = pytest.importorskip("torch")  # noqa: E402
from quant_fund.models import vrnn_seq  # noqa: E402


def test_eval_uses_holdout_not_train() -> None:
    """Training samples from X[:nseq-n_te]; eval must use the disjoint
    tail (previously eval re-ran sequences 0..19 = the training set)."""
    import inspect

    src = inspect.getsource(vrnn_seq.bench_vrnn_seq)
    assert "rng.integers(0, nseq - n_te)" in src
    assert "range(nseq - n_te, nseq)" in src


def test_bench_reports_gain() -> None:
    out = vrnn_seq.bench_vrnn_seq(steps=150)
    assert np.isfinite(out["synthetic_vrnn_step_nll"])
