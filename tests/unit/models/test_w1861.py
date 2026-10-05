import pytest


@pytest.mark.parametrize(
    "name",
    [
        "lancelot_qa_studies",
        "galahad_qa_studies",
        "percival_qa_studies",
        "gawain_qa_studies",
        "bedivere_qa_studies",
        "tristan_qa_studies",
    ],
)
def test_w1861_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
