import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ulgen2_qa_studies",
        "erlik2_qa_studies",
        "kydyr2_qa_studies",
        "alkarisi2_qa_studies",
        "baiyz2_qa_studies",
        "tenger2_qa_studies",
    ],
)
def test_w1826_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
