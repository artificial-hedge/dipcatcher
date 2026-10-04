import pytest


@pytest.mark.parametrize(
    "name",
    [
        "century_qa_studies",
        "decade_qa_studies",
        "date_qa_studies",
        "epoch_qa_studies",
        "era_qa_studies",
        "calendar_qa_studies",
    ],
)
def test_w1422_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
