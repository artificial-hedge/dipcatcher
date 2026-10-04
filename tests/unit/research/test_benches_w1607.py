import pytest

from quant_fund.research import benches_w1607


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bamboo_lemur_qa_studies_family",
        "bench_bearded_saki_qa_studies_family",
        "bench_owl_monkey_qa_studies_family",
        "bench_pale_titi_qa_studies_family",
        "bench_uakari_2_qa_studies_family",
        "bench_woolly_lemur_qa_studies_family",
    ],
)
def test_benches_w1607(fam):
    out = getattr(benches_w1607, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
