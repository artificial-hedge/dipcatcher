import pytest

from quant_fund.research import benches_w1794


@pytest.mark.parametrize(
    "fam",
    [
        "bench_maru2_qa_studies_family",
        "bench_pere_qa_studies_family",
        "bench_rongomai_qa_studies_family",
        "bench_tuhi2_qa_studies_family",
        "bench_uenuku2_qa_studies_family",
        "bench_wairere_qa_studies_family",
    ],
)
def test_benches_w1794(fam):
    out = getattr(benches_w1794, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
