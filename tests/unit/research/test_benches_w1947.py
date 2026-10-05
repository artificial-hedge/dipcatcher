import pytest

from quant_fund.research import benches_w1947


@pytest.mark.parametrize(
    "fam",
    [
        "bench_baron_samedi_qa_studies_family",
        "bench_damballa_qa_studies_family",
        "bench_ezili_dantor_qa_studies_family",
        "bench_ogou_feray_qa_studies_family",
        "bench_papa_legba_qa_studies_family",
        "bench_simbi_qa_studies_family",
    ],
)
def test_benches_w1947(fam):
    out = getattr(benches_w1947, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
