import importlib
from needle.autograd import Tensor, SparseTensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
from needle.ops import sparse_ops as sops
import needle.init as init
import numpy as np
try:
    _nn_sequence = importlib.import_module("needle.nn.nn_sequence")
    Embedding = getattr(_nn_sequence, "Embedding")
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Embedding = None
from .nn_basic import (
    Parameter,
    Module,
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential,
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
        window_size = 32,
        num_global_tokens = 0,
        block_stride = 0,
        random_tokens = 0,
        sparsity_threshold = 1e-6,
        mask_seed = 0,
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout) if dropout > 0 else None
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.block_stride = block_stride
        self.random_tokens = random_tokens
        self.sparsity_threshold = sparsity_threshold
        self.mask_seed = mask_seed
        self._pattern_cache = {}
        self._pattern_sparse_cache = {}

    def _build_sparse_pattern(self, q_len: int, kv_len: int) -> np.ndarray:
        """
        Construct a boolean mask that encodes the sparse attention pattern.
        Combines local windowed, global token, and random attention mechanisms.
        """
        mask = np.zeros((q_len, kv_len), dtype=bool)
        window = kv_len if self.window_size is None else max(0, self.window_size)

        for i in range(q_len):
            left = max(0, i - window)
            right = min(kv_len, i + window + 1)
            mask[i, left:right] = True

            if self.block_stride and self.block_stride > 0:
                start = i % self.block_stride
                mask[i, start::self.block_stride] = True

        diag = min(q_len, kv_len)
        mask[np.arange(diag), np.arange(diag)] = True

        if self.num_global_tokens:
            num_global = min(self.num_global_tokens, kv_len)
            mask[: min(self.num_global_tokens, q_len), :] = True  # global queries
            mask[:, :num_global] = True  # every token can see global keys

        if self.random_tokens:
            rng = np.random.default_rng(self.mask_seed)
            count = min(self.random_tokens, kv_len)
            if count > 0:
                for i in range(q_len):
                    rand_idx = rng.choice(kv_len, size=count, replace=False)
                    mask[i, rand_idx] = True

        if self.causal:
            causal = np.tril(np.ones((q_len, kv_len), dtype=bool))
            mask = np.logical_and(mask, causal)

        return mask.astype(np.float32)

    def _get_pattern_tensor(self, q_len: int, kv_len: int, device) -> Tensor:
        device_key = getattr(device, "name", str(device))
        key = (device_key, q_len, kv_len)
        target_device = device if device is not None else ndarray.default_device()
        if key not in self._pattern_cache:
            pattern = self._build_sparse_pattern(q_len, kv_len)
            pattern_nd = ndarray.array(pattern, device=target_device, dtype=self.dtype)
            self._pattern_cache[key] = Tensor(
                pattern_nd,
                device=target_device,
                dtype=self.dtype,
                requires_grad=False,
            )
        return self._pattern_cache[key]

    def _apply_pattern_mask(self, logits: Tensor, mask: Tensor) -> Tensor:
        broadcast_mask = mask.broadcast_to(logits.shape)
        device = broadcast_mask.device
        ones = init.ones(
            *broadcast_mask.shape,
            device=device,
            dtype=self.dtype,
            requires_grad=False,
        )
        neg_value = -np.finfo(np.float32).max
        neg_val = init.ones(
            *broadcast_mask.shape,
            device=device,
            dtype=self.dtype,
            requires_grad=False,
        ) * neg_value
        inverse_mask = ones - broadcast_mask
        return logits * broadcast_mask + inverse_mask * neg_val

    def _tensor_to_sparse(self, tensor: Tensor, threshold: float | None = None):
        """
        Helper to convert a dense Tensor into SparseTensor while avoiding CSR conversion
        for higher-rank tensors (CSR backend currently supports 2-D best).
        """
        thresh = self.sparsity_threshold if threshold is None else threshold
        use_csr = len(tensor.shape) <= 2
        return sops.dense_to_sparse(tensor, threshold=thresh, use_csr=use_csr)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        """
        upper = np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1
        )
        mask = np.where(upper > 0, -np.inf, 0.0).astype(np.float32)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        """
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        return (a * b_transpose).sum(len(a.shape) - 1)

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        if isinstance(q, SparseTensor):
            q = sops.sparse_to_dense(q)
        if isinstance(k, SparseTensor):
            k = sops.sparse_to_dense(k)
        if isinstance(v, SparseTensor):
            v = sops.sparse_to_dense(v)

        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        scores = self.matmul(q, k)
        scores = scores / (q_dim ** 0.5)

        pattern = self._get_pattern_tensor(queries_len, keys_values_len, q.device)
        pattern = pattern.reshape((1, 1, queries_len, keys_values_len))
        scores = self._apply_pattern_mask(scores, pattern)

        if self.causal:
            mask = self.create_causal_mask(queries_len, keys_values_len, scores.device)
            mask = mask.broadcast_to(scores.shape)
            scores = scores + Tensor(
                mask, device=scores.device, dtype=self.dtype, requires_grad=False
            )

        probs = self.softmax(scores)
        if self.dropout is not None:
            probs = self.dropout(probs)
        probs = probs * pattern.broadcast_to(probs.shape)

        probs_sparse = self._tensor_to_sparse(probs)
        probs_sparse = probs_sparse.reshape(
            (batch_size, num_head, queries_len, keys_values_len, 1)
        ).broadcast_to((batch_size, num_head, queries_len, keys_values_len, v_dim))

        values_view = v.reshape((batch_size, num_head, keys_values_len, v_dim))
        values_sparse = self._tensor_to_sparse(values_view, threshold=0.0)
        values_sparse = values_sparse.reshape(
            (batch_size, num_head, 1, keys_values_len, v_dim)
        ).broadcast_to((batch_size, num_head, queries_len, keys_values_len, v_dim))

        weighted = sops.sparse_multiply(probs_sparse, values_sparse)
        aggregated = sops.sparse_sum(weighted, axes=3)
        result = sops.sparse_to_dense(aggregated)
        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        # Pre-normalization
        q_norm = self.prenorm_q(q.reshape((batch_size * queries_len, q_dim)))
        k_norm = self.prenorm_k(k.reshape((batch_size * keys_values_len, k_dim)))
        v_norm = self.prenorm_v(v.reshape((batch_size * keys_values_len, v_dim)))

        q_w = self.q_projection(q_norm) # (bs * queries_len, d * head)
        k_w = self.k_projection(k_norm) # (bs * keys_values_len, d * head)
        v_w = self.v_projection(v_norm) # (bs * keys_values_len, d * head)

        q_w = q_w.reshape((batch_size, queries_len, self.num_head, self.dim_head))
        k_w = k_w.reshape((batch_size, keys_values_len, self.num_head, self.dim_head))
        v_w = v_w.reshape((batch_size, keys_values_len, self.num_head, self.dim_head))

        q_w = q_w.transpose(axes=(1, 2)) # (bs, num_head, queries_len, d)
        k_w = k_w.transpose(axes=(1, 2)) # (bs, num_head, keys_values_len, d)
        v_w = v_w.transpose(axes=(1, 2)) # (bs, num_head, keys_values_len, d)
        out, probs = self.attn(q_w, k_w, v_w) # (bs, num_head, queries_len, d)
        out = out.transpose(axes=(1, 2)).reshape((batch_size * queries_len, self.num_head * self.dim_head)) # (bs, queries_len, d * head)

        result = self.out_projection(out).reshape((batch_size, queries_len, self.out_features)) # (bs, q_len, out_dim)
        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.self_attn = AttentionLayer(
            q_features=q_features,
            num_head=num_head,
            dim_head=dim_head,
            dropout=dropout,
            causal=causal,
            device=device,
            dtype=dtype,
        )
        self.norm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.linear1 = Linear(q_features, hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(hidden_size, q_features, device=device, dtype=dtype)
        self.dropout = Dropout(dropout)
        self.relu = ReLU()
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        x = x + self.dropout(self.self_attn(x))

        y = self.norm(x.reshape((batch_size * seq_len, x_dim)))
        y = self.dropout(self.relu(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        x = x + y.reshape((batch_size, seq_len, x_dim))

        

        ### END YOUR SOLUTION

        return x
class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first
        self.sequence_len = sequence_len

        if Embedding is None:
            raise ImportError(
                "Transformer requires needle.nn.nn_sequence.Embedding, "
                "which is not available in this build.",
            )

        ### BEGIN YOUR SOLUTION
        self.position_embedding = Embedding(sequence_len, embedding_size, device=device, dtype=dtype)
        self.transformer_layers = Sequential(*[
            TransformerLayer(
                embedding_size,
                num_head=num_head,
                dim_head=dim_head,
                hidden_size=hidden_size,
                dropout=dropout,
                causal=causal,
                device=device,
                dtype=dtype,
            ) for _ in range(num_layers)
        ])
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1)) # (bs, seq_len, input_dim)

        batch_size, seq_len, input_dim = x.shape
        
        ### BEGIN YOUR SOLUTION
        positions = np.arange(seq_len).reshape(seq_len, 1)
        positions = ndarray.array(positions, device=self.device).broadcast_to((seq_len, batch_size))
        positions = Tensor(positions, device=self.device, dtype=self.dtype)

        pos_embedding = self.position_embedding(positions) # (seq_len, bs, input_dim)
        pos_embedding = ops.transpose(pos_embedding, axes=(0, 1)) # (bs, seq_len, input_dim)

        x = x + pos_embedding # (bs, seq_len, input_dim)
        x = self.transformer_layers(x) # (bs, seq_len, input_dim)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)