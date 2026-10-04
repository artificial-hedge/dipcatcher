import pytest


@pytest.mark.parametrize(
    "name",
    [
        "clarq_lite_studies",
        "duread_qa_studies",
        "doqa_lite_studies",
        "orchid_qa_studies",
        "qrecc_lite_studies",
        "canard_lite_studies",
    ],
)
def test_w1407_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
