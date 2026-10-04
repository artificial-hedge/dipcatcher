import pytest


@pytest.mark.parametrize(
    "name",
    [
        "sabazios2_qa_studies",
        "attis2_qa_studies",
        "cybele2_qa_studies",
        "agdistis2_qa_studies",
        "men2_qa_studies",
        "papas2_qa_studies",
    ],
)
def test_w1808_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
