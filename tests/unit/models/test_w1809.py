import pytest


@pytest.mark.parametrize(
    "name",
    [
        "manannan2_qa_studies",
        "lugh2_qa_studies",
        "dagda2_qa_studies",
        "brigid2_qa_studies",
        "ogma2_qa_studies",
        "nuada2_qa_studies",
    ],
)
def test_w1809_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
