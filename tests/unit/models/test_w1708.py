import pytest


@pytest.mark.parametrize(
    "name",
    [
        "numit_qa_studies",
        "numishi_qa_studies",
        "naveluz_qa_studies",
        "piryani_qa_studies",
        "yejmun_qa_studies",
        "metsik_qa_studies",
    ],
)
def test_w1708_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
