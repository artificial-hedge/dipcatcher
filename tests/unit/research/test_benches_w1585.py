import pytest

from quant_fund.research import benches_w1585


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aardvark_qa_studies_family",
        "bench_elephant_shrew_qa_studies_family",
        "bench_golden_mole_qa_studies_family",
        "bench_gymnure_qa_studies_family",
        "bench_solenodon_qa_studies_family",
        "bench_tenrec_qa_studies_family",
    ],
)
def test_benches_w1585(fam):
    out = getattr(benches_w1585, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
