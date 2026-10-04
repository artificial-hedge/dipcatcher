import pytest


@pytest.mark.parametrize(
    "name",
    [
        "needle_haystack_studies",
        "ruler_studies",
        "infinite_bench_studies",
        "longmem_studies",
        "truthful_qa_studies",
        "halu_eval_studies",
    ],
)
def test_w1308_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
