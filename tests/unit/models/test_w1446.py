import pytest


@pytest.mark.parametrize(
    "name",
    [
        "drum_qa_studies",
        "guitar_qa_studies",
        "flute_qa_studies",
        "piano_qa_studies",
        "violin_qa_studies",
        "cello_qa_studies",
    ],
)
def test_w1446_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
