import pytest

from quant_fund.research import benches_w1593


@pytest.mark.parametrize(
    "fam",
    [
        "bench_colobus_qa_studies_family",
        "bench_drill_qa_studies_family",
        "bench_gelada_qa_studies_family",
        "bench_guenon_qa_studies_family",
        "bench_mandrill_qa_studies_family",
        "bench_mangabey_qa_studies_family",
    ],
)
def test_benches_w1593(fam):
    out = getattr(benches_w1593, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
