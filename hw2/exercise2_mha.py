"""
Exercise 2: Multi-Head Attention (MHA)
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

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

        # QKV projection: (B, L, D) -> (B, L, H*D_k)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Split heads: (B, L, H*D_k) -> (B, H, L, D_k)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product: (B, H, L, D_k) @ (B, H, D_k, L) -> (B, H, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum: (B, H, L, L) @ (B, H, L, D_k) -> (B, H, L, D_k)
        out = torch.matmul(attn, v)

        # Concat heads: (B, H, L, D_k) -> (B, L, H*D_k)
        out = out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)

        # Output projection: (B, L, H*D_k) -> (B, L, D)
        return self.W_o(out)


if __name__ == "__main__":
    print("=" * 50)
    print("Ex 2: Multi-Head Attention Assessment")
    print("=" * 50)

    B, L, D = 2, 8, 64
    num_heads = 4
    device = torch.device("cpu")
    torch.manual_seed(42)

    mha = MultiHeadAttention(D, num_heads).to(device)
    x = torch.randn(B, L, D, device=device)

    out = mha(x)
    assert out.shape == (B, L, D), f"Shape mismatch: {out.shape}"
    print("[OK] Shape test passed: output (B, L, D)")

    # --- causal mask ---
    causal = torch.tril(torch.ones(L, L))
    out_masked = mha(x, causal)
    assert out_masked.shape == (B, L, D)
    print("[OK] Causal mask works")

    # --- deterministic seed ---
    torch.manual_seed(0)
    x0 = torch.randn(1, 4, 8)
    mha0 = MultiHeadAttention(8, 2)
    out0 = mha0(x0)
    out0_2 = mha0(x0)
    assert torch.allclose(out0, out0_2), "Non-deterministic output"
    print(f"[OK] Deterministic output (sum={out0.sum().item():.4f})")

    print("\nAll tests passed!")
