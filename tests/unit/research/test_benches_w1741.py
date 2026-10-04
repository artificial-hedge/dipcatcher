import pytest

from quant_fund.research import benches_w1741


@pytest.mark.parametrize(
    "fam",
    [
        "bench_hina_qa_studies_family",
        "bench_kanaloa_qa_studies_family",
        "bench_kane_qa_studies_family",
        "bench_ku_qa_studies_family",
        "bench_lono_qa_studies_family",
        "bench_pele_qa_studies_family",
    ],
)
def test_benches_w1741(fam):
    out = getattr(benches_w1741, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
