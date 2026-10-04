import pytest

from quant_fund.research import benches_w1608


@pytest.mark.parametrize(
    "fam",
    [
        "bench_buffalo_qa_studies_family",
        "bench_kob_qa_studies_family",
        "bench_lechwe_qa_studies_family",
        "bench_rhino_qa_studies_family",
        "bench_roan_qa_studies_family",
        "bench_warthog_qa_studies_family",
    ],
)
def test_benches_w1608(fam):
    out = getattr(benches_w1608, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
