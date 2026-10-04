import pytest


@pytest.mark.parametrize(
    "name",
    [
        "aqua_lite_studies",
        "num_glue_studies",
        "math_qa_studies",
        "tab_fact_studies",
        "tat_qa_studies",
        "fin_qa_studies",
    ],
)
def test_w1378_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
