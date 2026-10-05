import pytest

from quant_fund.research import benches_w1896


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ereshkigal_namtar_qa_studies_family",
        "bench_lama_demon_qa_studies_family",
        "bench_mukil_qa_studies_family",
        "bench_mukil_res_lemuttim_qa_studies_family",
        "bench_nergal_demon_qa_studies_family",
        "bench_rabisu_hursag_qa_studies_family",
    ],
)
def test_benches_w1896(fam):
    out = getattr(benches_w1896, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
