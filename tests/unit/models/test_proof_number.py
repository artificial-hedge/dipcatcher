from quant_fund.models.proof_number import bench_proof_number, proof_number_search


def test_prove_n_position():
    proved, _ = proof_number_search(9)  # 9 % 4 = 1 → forced win
    assert proved is True


def test_disprove_p_position():
    proved, _ = proof_number_search(8)  # P-position
    assert proved is False


def test_bench():
    out = bench_proof_number(seed=5)
    assert out["synthetic_exact_status"] == 1.0
