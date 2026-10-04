import pytest


@pytest.mark.parametrize(
    "name",
    [
        "romulus2_qa_studies",
        "remus2_qa_studies",
        "aeneas2_qa_studies",
        "lavinia2_qa_studies",
        "turnus2_qa_studies",
        "evander2_qa_studies",
    ],
)
def test_w1800_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
