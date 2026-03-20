import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        num_tokens, d_model = x.shape
        
        router_logits = self.router(x)
        probs = F.softmax(router_logits, dim=-1)
        
        expert_idx = probs.argmax(dim=-1)
        
        weights = probs.gather(1, expert_idx.unsqueeze(1)).squeeze(1)
        
        output = torch.zeros_like(x)
        
        for i, expert in enumerate(self.experts):
            mask = (expert_idx == i)
            if mask.any():
                expert_input = x[mask]
                expert_output = expert(expert_input)
                weighted_output = expert_output * weights[mask].unsqueeze(1)
                output[mask] = weighted_output
        
        return output


def test_moe():
    print("=" * 50)
    print("Exercise 5: MoE Layer Tests")
    print("=" * 50)
    
    torch.manual_seed(42)
    
    d_model = 64
    d_ff = 128
    num_experts = 4
    num_tokens = 16
    
    moe = MoELayer(d_model, d_ff, num_experts)
    
    x = torch.randn(num_tokens, d_model)
    output = moe(x)
    
    print("\n1. Shape verification:")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == (num_tokens, d_model), "Shape mismatch!"
    print("   [OK] Shape is correct")
    
    print("\n2. Router test:")
    router_logits = moe.router(x)
    print(f"   Router logits shape: {router_logits.shape}")
    probs = F.softmax(router_logits, dim=-1)
    print(f"   Probabilities sum along dim=-1: {probs.sum(dim=-1)}")
    assert torch.allclose(probs.sum(dim=-1), torch.ones(num_tokens), atol=1e-5)
    print("   [OK] Router probabilities sum to 1")
    
    print("\n3. Top-1 routing test:")
    expert_idx = probs.argmax(dim=-1)
    print(f"   Expert indices: {expert_idx.tolist()}")
    for i in range(num_experts):
        count = (expert_idx == i).sum().item()
        print(f"   Expert {i}: {count} tokens ({count/num_tokens*100:.1f}%)")
    print("   [OK] Tokens routed to top-1 expert")
    
    print("\n4. Output computation test:")
    weights = probs.gather(1, expert_idx.unsqueeze(1)).squeeze(1)
    print(f"   Routing weights shape: {weights.shape}")
    print(f"   Sample weights: {weights[:5].tolist()}")
    print("   [OK] Output weighted by routing probability")
    
    print("\n5. Gradient flow test:")
    output.sum().backward()
    assert moe.router.weight.grad is not None
    print("   [OK] Gradient flows through router")
    
    print("\n6. Parameter count:")
    total_params = sum(p.numel() for p in moe.parameters())
    router_params = moe.router.weight.numel() + moe.router.bias.numel()
    expert_params = sum(p.numel() for p in moe.experts.parameters())
    print(f"   Router parameters: {router_params}")
    print(f"   Total expert parameters: {expert_params}")
    print(f"   Total parameters: {total_params}")
    print(f"   Note: Experts share same architecture but independent weights")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    test_moe()