import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hiruko_qa_studies",
        "ebisu_qa_studies",
        "kisshoten_qa_studies",
        "kikuzuki_qa_studies",
        "morinaga_qa_studies",
        "senju_qa_studies",
    ],
)
def test_w1760_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
