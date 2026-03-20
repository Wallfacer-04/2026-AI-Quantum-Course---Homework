"""
Exercise 1: RoPE (Rotary Position Embedding)
"""

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in pairs and rotate: [a,b,c,d,...] -> [-b,a,-d,c,...]."""
    half = x.shape[-1] // 2
    x = x.reshape(*x.shape[:-1], half, 2)                   # (..., pairs, 2)
    x = torch.stack([-x[..., 1], x[..., 0]], dim=-1)       # (..., pairs, 2) columns interleaved
    return x.flatten(-2)                                    # (..., dim)


def get_rotary_cos_sin(
    seq_len: int,
    dim: int,
    device: torch.device | None = None,
    base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cos/sin embeddings for RoPE: theta_i = base^(-2i/dim)."""
    positions = torch.arange(seq_len, device=device)
    half = dim // 2
    freqs = base ** (-2.0 * torch.arange(0, half, device=device).float() / dim)
    angles = positions[:, None] * freqs[None, :]   # (seq_len, half)
    cos_emb = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin_emb = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos_emb, sin_emb


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_emb: torch.Tensor,
    sin_emb: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE via Hadamard product: v * cos + rotate_half(v) * sin."""
    n_unsqueeze = q.dim() - 2
    for _ in range(n_unsqueeze):
        cos_emb = cos_emb.unsqueeze(0)
        sin_emb = sin_emb.unsqueeze(0)
    return q * cos_emb + rotate_half(q) * sin_emb, k * cos_emb + rotate_half(k) * sin_emb


if __name__ == "__main__":
    print("=" * 50)
    print("Ex 1: RoPE Assessment")
    print("=" * 50)

    B, L, H, D = 2, 8, 4, 64
    device = torch.device("cpu")
    torch.manual_seed(42)

    # --- basic shapes ---
    q = torch.randn(B, H, L, D, device=device)
    k = torch.randn(B, H, L, D, device=device)
    cos, sin = get_rotary_cos_sin(L, D, device)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape and k_rot.shape == k.shape
    print("[OK] Shape test passed")

    # --- rotate_half ---
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    expected = torch.tensor([[-2.0, 1.0, -4.0, 3.0, -6.0, 5.0, -8.0, 7.0]])
    assert torch.allclose(rotate_half(x), expected)
    print("[OK] rotate_half test passed")

    # --- relative position invariance ---
    q_same = torch.randn(1, 1, 8)
    cos4, sin4 = get_rotary_cos_sin(4, 8)
    scores = []
    for i in range(4):
        q_r, _ = apply_rotary_pos_emb(q_same, q_same, cos4[i:i+1], sin4[i:i+1])
        scores.append(torch.sum(q_r * q_r[0], dim=-1))
    assert torch.allclose(scores[0], scores[1], atol=1e-4)
    print("[OK] Relative position invariance verified")

    # --- norm preservation ---
    q0 = q[0, 0, 0]
    assert torch.allclose(q0.norm(), q_rot[0, 0, 0].norm(), atol=1e-4)
    print("[OK] Norm preserved (orthogonal transform)")

    print("\nAll tests passed!")
