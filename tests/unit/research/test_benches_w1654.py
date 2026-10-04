import pytest

from quant_fund.research import benches_w1654


@pytest.mark.parametrize(
    "fam",
    [
        "bench_amphisbaena_qa_studies_family",
        "bench_bonnacon_qa_studies_family",
        "bench_cerastes_qa_studies_family",
        "bench_leucrotta_qa_studies_family",
        "bench_parandrus_qa_studies_family",
        "bench_questing_qa_studies_family",
    ],
)
def test_benches_w1654(fam):
    out = getattr(benches_w1654, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
