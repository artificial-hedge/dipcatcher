import pytest


@pytest.mark.parametrize(
    "name",
    [
        "swe_dev_studies",
        "swe_multimodal_studies",
        "code_rag_studies",
        "codegen_universal_studies",
        "odex_eval_studies",
        "long_code_bench_studies",
    ],
)
def test_w1335_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
