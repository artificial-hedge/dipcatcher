import pytest


@pytest.mark.parametrize(
    "name",
    [
        "tessub2_qa_studies",
        "hebat2_qa_studies",
        "sarruma2_qa_studies",
        "simige2_qa_studies",
        "kusuh2_qa_studies",
        "tasmisu2_qa_studies",
    ],
)
def test_w1834_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
