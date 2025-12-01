import numpy as np
import sys

sys.path.append("./python")

from needle.autograd import SparseTensor
from needle.nn.nn_transformer_sparse import MultiHeadAttention


def _dense_reference(q, k, v, mask, causal):
    bsz, num_head, q_len, dim = q.shape
    _, _, k_len, _ = k.shape

    scores = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) / np.sqrt(dim)
    mask_reshaped = mask.reshape(1, 1, q_len, k_len)
    large_neg = -np.finfo(np.float32).max
    scores = np.where(mask_reshaped > 0, scores, large_neg)

    if causal:
        causal_mask = np.triu(np.ones((q_len, k_len), dtype=bool), 1)
        scores = np.where(causal_mask, large_neg, scores)

    scores_shifted = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores_shifted)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    probs = probs * mask_reshaped

    weighted = probs[..., None] * v[:, :, None, :, :]
    return weighted.sum(axis=3)


def _run_attention_and_dense(mha, q_np, k_np, v_np):
    q = SparseTensor(q_np)
    k = SparseTensor(k_np)
    v = SparseTensor(v_np)
    out_sparse, probs = mha(q, k, v)
    mask = mha._build_sparse_pattern(q_np.shape[2], k_np.shape[2])
    expected = _dense_reference(q_np, k_np, v_np, mask, mha.causal)
    return out_sparse, probs, mask, expected


def test_local_window_attention_matches_dense():
    rng = np.random.default_rng(0)
    batch_size, num_head, seq_len, dim = 1, 2, 6, 3

    q_np = rng.standard_normal((batch_size, num_head, seq_len, dim), dtype=np.float32)
    k_np = rng.standard_normal((batch_size, num_head, seq_len, dim), dtype=np.float32)
    v_np = rng.standard_normal((batch_size, num_head, seq_len, dim), dtype=np.float32)

    mha = MultiHeadAttention(
        dropout=0.0,
        causal=False,
        window_size=1,
        num_global_tokens=0,
        block_stride=0,
        random_tokens=0,
        sparsity_threshold=0.0,
        mask_seed=0,
    )

    out_sparse, _, _, expected = _run_attention_and_dense(mha, q_np, k_np, v_np)

    np.testing.assert_allclose(
        out_sparse.numpy(), expected, atol=1e-4, rtol=1e-4
    )


def test_global_token_attention_matches_dense():
    rng = np.random.default_rng(123)
    batch_size, num_head, seq_len, dim = 1, 2, 6, 4

    q_np = rng.standard_normal((batch_size, num_head, seq_len, dim), dtype=np.float32)
    k_np = rng.standard_normal((batch_size, num_head, seq_len, dim), dtype=np.float32)
    v_np = rng.standard_normal((batch_size, num_head, seq_len, dim), dtype=np.float32)

    mha = MultiHeadAttention(
        dropout=0.0,
        causal=False,
        window_size=1,
        num_global_tokens=2,
        block_stride=0,
        random_tokens=0,
        sparsity_threshold=0.0,
        mask_seed=7,
    )

    out_sparse, _, mask, expected = _run_attention_and_dense(mha, q_np, k_np, v_np)

    # Ensure that all tokens can attend to the global columns.
    assert np.all(mask[:, :2] == 1.0)

    np.testing.assert_allclose(
        out_sparse.numpy(), expected, atol=1e-4, rtol=1e-4
    )


def test_random_attention_matches_dense():
    rng = np.random.default_rng(999)
    batch_size, num_head, seq_len, dim = 2, 1, 7, 3

    q_np = rng.standard_normal((batch_size, num_head, seq_len, dim), dtype=np.float32)
    k_np = rng.standard_normal((batch_size, num_head, seq_len, dim), dtype=np.float32)
    v_np = rng.standard_normal((batch_size, num_head, seq_len, dim), dtype=np.float32)

    mha = MultiHeadAttention(
        dropout=0.0,
        causal=False,
        window_size=0,
        num_global_tokens=0,
        block_stride=0,
        random_tokens=3,
        sparsity_threshold=0.0,
        mask_seed=5,
    )

    out_sparse, _, mask, expected = _run_attention_and_dense(mha, q_np, k_np, v_np)

    # Each row should include the diagonal plus the sampled random tokens.
    row_nonzeros = mask.sum(axis=1)
    assert np.all(row_nonzeros >= 1)  # at least the diagonal survives

    np.testing.assert_allclose(
        out_sparse.numpy(), expected, atol=1e-4, rtol=1e-4
    )


def test_sparse_attention_respects_causal_mask():
    rng = np.random.default_rng(1)
    batch_size, num_head, seq_len, dim = 1, 1, 5, 2

    data = rng.standard_normal((batch_size, num_head, seq_len, dim), dtype=np.float32)
    mha = MultiHeadAttention(
        dropout=0.0,
        causal=True,
        window_size=3,
        num_global_tokens=0,
        block_stride=0,
        random_tokens=0,
        sparsity_threshold=0.0,
        mask_seed=42,
    )

    out_sparse, probs, mask, expected = _run_attention_and_dense(mha, data, data, data)

    probs_np = probs.numpy()
    upper = np.triu(np.ones((seq_len, seq_len)), 1).astype(bool)
    assert np.allclose(probs_np[0, 0][upper], 0)

    np.testing.assert_allclose(
        out_sparse.numpy(), expected, atol=1e-4, rtol=1e-4
    )

