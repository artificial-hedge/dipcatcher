import pytest


@pytest.mark.parametrize(
    "name",
    [
        "mukil_qa_studies",
        "ereshkigal_namtar_qa_studies",
        "nergal_demon_qa_studies",
        "mukil_res_lemuttim_qa_studies",
        "rabisu_hursag_qa_studies",
        "lama_demon_qa_studies",
    ],
)
def test_w1896_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
