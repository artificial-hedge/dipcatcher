import pytest

from quant_fund.research import benches_w1719


@pytest.mark.parametrize(
    "fam",
    [
        "bench_basajaun_qa_studies_family",
        "bench_eguzki_qa_studies_family",
        "bench_lamiak_qa_studies_family",
        "bench_mairu_qa_studies_family",
        "bench_mari_qa_studies_family",
        "bench_sugaar_qa_studies_family",
    ],
)
def test_benches_w1719(fam):
    out = getattr(benches_w1719, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
