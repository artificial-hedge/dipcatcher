import pytest

from quant_fund.research import benches_w1954


@pytest.mark.parametrize(
    "fam",
    [
        "bench_flauros_qa_studies_family",
        "bench_kimaris_qa_studies_family",
        "bench_oriens_qa_studies_family",
        "bench_valac_qa_studies_family",
        "bench_vapula_qa_studies_family",
        "bench_zagan_qa_studies_family",
    ],
)
def test_benches_w1954(fam):
    out = getattr(benches_w1954, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
