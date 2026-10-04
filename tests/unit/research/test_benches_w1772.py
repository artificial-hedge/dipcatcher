import pytest

from quant_fund.research import benches_w1772


@pytest.mark.parametrize(
    "fam",
    [
        "bench_hecate_qa_studies_family",
        "bench_helios_qa_studies_family",
        "bench_hypnos_qa_studies_family",
        "bench_selene_qa_studies_family",
        "bench_thanatos_qa_studies_family",
        "bench_zephyrus_qa_studies_family",
    ],
)
def test_benches_w1772(fam):
    out = getattr(benches_w1772, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
