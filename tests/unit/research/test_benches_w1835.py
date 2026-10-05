import pytest

from quant_fund.research import benches_w1835


@pytest.mark.parametrize(
    "fam",
    [
        "bench_hannahanna2_qa_studies_family",
        "bench_istanuwa2_qa_studies_family",
        "bench_iyarri2_qa_studies_family",
        "bench_kamrusepa2_qa_studies_family",
        "bench_runtija2_qa_studies_family",
        "bench_tarhunza2_qa_studies_family",
    ],
)
def test_benches_w1835(fam):
    out = getattr(benches_w1835, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
