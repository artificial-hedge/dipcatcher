import pytest


@pytest.mark.parametrize(
    "name",
    [
        "kinnara_qa_studies",
        "vidyadhara_qa_studies",
        "pisacha_qa_studies",
        "yakshini_qa_studies",
        "vetala_qa_studies",
        "uraga_qa_studies",
    ],
)
def test_w1676_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
