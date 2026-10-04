import pytest


@pytest.mark.parametrize(
    "name",
    [
        "isis_qa_studies",
        "osiris_qa_studies",
        "horus_qa_studies",
        "set_qa_studies",
        "geb_qa_studies",
        "shu_qa_studies",
    ],
)
def test_w1782_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
