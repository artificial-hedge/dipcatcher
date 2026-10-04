import pytest

from quant_fund.research import benches_w1677


@pytest.mark.parametrize(
    "fam",
    [
        "bench_antheia_qa_studies_family",
        "bench_aurae_qa_studies_family",
        "bench_camenae_qa_studies_family",
        "bench_fauns_qa_studies_family",
        "bench_limoniad_qa_studies_family",
        "bench_numina_qa_studies_family",
    ],
)
def test_benches_w1677(fam):
    out = getattr(benches_w1677, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
