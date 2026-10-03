import numpy as np

from quant_fund.models.lda_topics import bench_lda_topics, lda_gibbs


def test_lda_recovers_topics():
    rng = np.random.default_rng(0)
    vocab = 20
    phi_true = np.zeros((2, vocab)) + 0.01
    phi_true[0, :8] = 0.12
    phi_true[1, 8:16] = 0.12
    phi_true /= phi_true.sum(axis=1, keepdims=True)
    docs = []
    for i in range(40):
        theta = np.array([0.9, 0.1]) if i < 20 else np.array([0.1, 0.9])
        zs = rng.choice(2, 30, p=theta)
        docs.append(np.asarray([rng.choice(vocab, p=phi_true[k]) for k in zs]))
    m = lda_gibbs(docs, n_topics=2, vocab=vocab, it=120, seed=0)
    phi = np.asarray(m["phi"])
    assert phi.shape == (2, vocab)
    # each planted profile is matched by some learned topic
    for t in range(2):
        assert min(np.abs(phi - phi_true[t]).sum(axis=1).min() for _ in [0]) < 0.6


def test_lda_theta_rows_sum_one():
    rng = np.random.default_rng(1)
    docs = [rng.integers(0, 10, 20) for _ in range(10)]
    m = lda_gibbs(docs, n_topics=3, vocab=10, it=10, seed=1)
    theta = np.asarray(m["theta"])
    assert np.allclose(theta.sum(axis=1), 1.0)


def test_bench_lda_topics():
    out = bench_lda_topics(seed=556)
    assert out["synthetic_lda_phi_l1_min"] < 0.4
