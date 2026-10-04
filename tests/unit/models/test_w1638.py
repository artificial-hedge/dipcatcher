import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ubume_qa_studies",
        "nurikabe_qa_studies",
        "hitodama_qa_studies",
        "shikigami_qa_studies",
        "akaname_qa_studies",
        "ittanmomen_qa_studies",
    ],
)
def test_w1638_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
