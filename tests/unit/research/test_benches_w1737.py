import pytest

from quant_fund.research import benches_w1737


@pytest.mark.parametrize(
    "fam",
    [
        "bench_coatlicue_qa_studies_family",
        "bench_coyolxauhqui_qa_studies_family",
        "bench_huitzilopochtli_qa_studies_family",
        "bench_mictlantecuhtli_qa_studies_family",
        "bench_tlaloc_qa_studies_family",
        "bench_tonatiuh_qa_studies_family",
    ],
)
def test_benches_w1737(fam):
    out = getattr(benches_w1737, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
