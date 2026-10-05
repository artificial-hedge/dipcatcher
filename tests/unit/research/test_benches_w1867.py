import pytest

from quant_fund.research import benches_w1867


@pytest.mark.parametrize(
    "fam",
    [
        "bench_antenociticus_qa_studies_family",
        "bench_ares_lusitani_qa_studies_family",
        "bench_braciaca_qa_studies_family",
        "bench_deiba_qa_studies_family",
        "bench_nantosuelta_qa_studies_family",
        "bench_ognios_qa_studies_family",
    ],
)
def test_benches_w1867(fam):
    out = getattr(benches_w1867, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
