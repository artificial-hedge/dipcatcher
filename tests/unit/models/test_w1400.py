import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cwq_lite_studies",
        "kgqa_lite_studies",
        "grailqa_studies",
        "pweb_qa_studies",
        "qald_lite_studies",
        "cronqa_lite_studies",
    ],
)
def test_w1400_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
