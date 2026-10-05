import pytest


@pytest.mark.parametrize(
    "name",
    [
        "olwen_qa_studies",
        "geraint_qa_studies",
        "enid_qa_studies",
        "owen_qa_studies",
        "rheged_qa_studies",
        "gwalchmei_qa_studies",
    ],
)
def test_w1865_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
