import pytest


@pytest.mark.parametrize(
    "name",
    [
        "koshchey_qa_studies",
        "indrik_qa_studies",
        "psoglav_qa_studies",
        "belun_qa_studies",
        "mavka_qa_studies",
        "triglav_qa_studies",
    ],
)
def test_w1912_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
