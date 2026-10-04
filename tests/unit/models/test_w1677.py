import pytest


@pytest.mark.parametrize(
    "name",
    [
        "fauns_qa_studies",
        "camenae_qa_studies",
        "limoniad_qa_studies",
        "antheia_qa_studies",
        "aurae_qa_studies",
        "numina_qa_studies",
    ],
)
def test_w1677_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
