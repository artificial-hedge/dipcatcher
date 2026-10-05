import pytest


@pytest.mark.parametrize(
    "name",
    [
        "wekufe_qa_studies",
        "kalku_qa_studies",
        "cherufe_qa_studies",
        "colo_colo_qa_studies",
        "peuchen_qa_studies",
        "chonchon_qa_studies",
    ],
)
def test_w1934_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
