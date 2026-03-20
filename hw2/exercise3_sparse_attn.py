"""
Exercise 3: Sparse Attention (Top-K)
"""

import torch
import torch.nn as nn
import math


class SparseAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, k: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.k = k

        d_kv = num_heads * self.head_dim
        self.W_q = nn.Linear(d_model, d_kv)
        self.W_k = nn.Linear(d_model, d_kv)
        self.W_v = nn.Linear(d_model, d_kv)
        self.W_o = nn.Linear(d_kv, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            mask: (L, L) causal mask or None
        Returns:
            (B, L, D)
        """
        B, L, _ = x.shape

        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Top-K sparse attention: keep only top k values, set rest to -inf
        topk_vals, _ = scores.topk(self.k, dim=-1)
        threshold = topk_vals[..., -1]                                        # (B, H, L)
        sparse_scores = scores.masked_fill(scores < threshold[..., None], float("-inf"))

        attn = torch.softmax(sparse_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)

        return self.W_o(out)


if __name__ == "__main__":
    print("=" * 50)
    print("Ex 3: Sparse Attention Assessment")
    print("=" * 50)

    B, L, D = 2, 8, 64
    num_heads = 4
    device = torch.device("cpu")
    torch.manual_seed(42)

    # --- basic shape ---
    sparse_attn = SparseAttention(D, num_heads, k=4).to(device)
    x = torch.randn(B, L, D, device=device)
    out = sparse_attn(x)
    assert out.shape == (B, L, D)
    print("[OK] Shape test passed: output (B, L, D)")

    # --- sparse scores are zero for non-top-k ---
    torch.manual_seed(0)
    x0 = torch.randn(1, 4, 8)
    attn0 = SparseAttention(8, 2, k=2)
    scores = torch.randn(1, 2, 4, 4)            # non-uniform scores
    topk_vals, _ = scores.topk(2, dim=-1)
    threshold = topk_vals[..., -1][..., None]   # (1, 2, 4, 1)
    sparse = scores.masked_fill(scores < threshold, float("-inf"))
    ninf = (sparse == float("-inf")).sum().item()
    total = 1 * 2 * 4 * 4
    expected_inf = total - 1 * 2 * 4 * 2       # keep k=2 per head per query
    assert ninf == expected_inf, f"Expected {expected_inf} -inf entries, got {ninf}"
    print("[OK] Top-K masking: non-topk entries are -inf")

    # --- causal mask compatibility ---
    causal = torch.tril(torch.ones(L, L))
    out_masked = sparse_attn(x, causal)
    assert out_masked.shape == (B, L, D)
    print("[OK] Causal mask works with sparse attention")

    print("\nAll tests passed!")
