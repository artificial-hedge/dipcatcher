import pytest

from quant_fund.research import benches_w1726


@pytest.mark.parametrize(
    "fam",
    [
        "bench_dalnim_qa_studies_family",
        "bench_dangun_qa_studies_family",
        "bench_haenim_qa_studies_family",
        "bench_hwanin_qa_studies_family",
        "bench_hwanung_qa_studies_family",
        "bench_samshin_qa_studies_family",
    ],
)
def test_benches_w1726(fam):
    out = getattr(benches_w1726, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
