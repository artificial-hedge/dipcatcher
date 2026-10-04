import pytest


@pytest.mark.parametrize(
    "name",
    [
        "kumarbi2_qa_studies",
        "tesub2_qa_studies",
        "hannahanna2_qa_studies",
        "kamrusepa2_qa_studies",
        "pirinkir2_qa_studies",
        "ilib2_qa_studies",
    ],
)
def test_w1807_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
