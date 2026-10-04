import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ammon_qa_studies",
        "milcom_qa_studies",
        "el2_qa_studies",
        "baalis2_qa_studies",
        "sodom2_qa_studies",
        "moloch2_qa_studies",
    ],
)
def test_w1842_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
