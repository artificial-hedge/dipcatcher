import pytest


@pytest.mark.parametrize(
    "name",
    [
        "monitor_lizard_qa_studies",
        "chuckwalla_qa_studies",
        "tegu_qa_studies",
        "frilled_lizard_qa_studies",
        "uromastyx_qa_studies",
        "agama_qa_studies",
    ],
)
def test_w1550_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
