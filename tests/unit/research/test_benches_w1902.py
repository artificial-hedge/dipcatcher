import pytest

from quant_fund.research import benches_w1902


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ao_andon_qa_studies_family",
        "bench_hannya_oni_qa_studies_family",
        "bench_kamaitachi_qa_studies_family",
        "bench_nurarihyon_qa_studies_family",
        "bench_shuten_doji_qa_studies_family",
        "bench_tsuchigumo_qa_studies_family",
    ],
)
def test_benches_w1902(fam):
    out = getattr(benches_w1902, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
