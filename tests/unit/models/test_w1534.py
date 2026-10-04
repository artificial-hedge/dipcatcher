import pytest


@pytest.mark.parametrize(
    "name",
    [
        "koel_qa_studies",
        "nightjar_qa_studies",
        "roadrunner_qa_studies",
        "nighthawk_qa_studies",
        "frogmouth_qa_studies",
        "cuckoo_qa_studies",
    ],
)
def test_w1534_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
