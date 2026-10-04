import pytest

from quant_fund.research import benches_w1635


@pytest.mark.parametrize(
    "fam",
    [
        "bench_kappa_qa_studies_family",
        "bench_kitsune_2_qa_studies_family",
        "bench_oni_qa_studies_family",
        "bench_tanuki_2_qa_studies_family",
        "bench_tengu_qa_studies_family",
        "bench_tsukumogami_qa_studies_family",
    ],
)
def test_benches_w1635(fam):
    out = getattr(benches_w1635, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
