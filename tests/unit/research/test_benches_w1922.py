import pytest

from quant_fund.research import benches_w1922


@pytest.mark.parametrize(
    "fam",
    [
        "bench_kalma_qa_studies_family",
        "bench_kratti_qa_studies_family",
        "bench_loviatar_qa_studies_family",
        "bench_nakki_qa_studies_family",
        "bench_painajainen_qa_studies_family",
        "bench_tursas_qa_studies_family",
    ],
)
def test_benches_w1922(fam):
    out = getattr(benches_w1922, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
