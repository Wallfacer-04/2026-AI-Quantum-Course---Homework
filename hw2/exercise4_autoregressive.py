import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from itertools import product


class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_emb = nn.Embedding(vocab_size + 1, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.0,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, vocab_size + 1)
        self.bos_idx = vocab_size
    
    def forward(self, x):
        B, L = x.shape
        emb = self.token_emb(x) + self.pos_emb[:, :L, :]
        
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        
        output = self.transformer(emb, mask=mask)
        logits = self.output_proj(output)
        return logits


def compute_joint_prob(model, sequence):
    log_prob = 0.0
    bos = torch.tensor([[model.bos_idx]], dtype=torch.long)
    for i in range(len(sequence)):
        context = torch.cat([bos, torch.tensor([sequence[:i+1]], dtype=torch.long)], dim=1)
        logits = model(context)
        probs = F.softmax(logits[0, i], dim=-1)
        probs_valid = probs[:model.vocab_size] / probs[:model.vocab_size].sum()
        log_prob += torch.log(probs_valid[sequence[i]] + 1e-10).item()
    return math.exp(log_prob)


def task_4_1():
    print("=" * 50)
    print("Task 4.1: Compute Joint Probability for All Sequences")
    print("=" * 50)
    
    torch.manual_seed(42)
    model = CausalTransformer(
        vocab_size=2,
        d_model=64,
        num_heads=4,
        num_layers=2,
        max_seq_len=10
    )
    model.eval()
    
    probabilities = []
    for seq in product([0, 1], repeat=4):
        prob = compute_joint_prob(model, list(seq))
        probabilities.append(prob)
        print(f"Sequence {seq}: P = {prob:.6f}")
    
    total = sum(probabilities)
    print("-" * 50)
    print(f"Sum of probabilities: {total:.6f}")
    print(f"Verification: {'PASS' if abs(total - 1.0) < 1e-3 else 'FAIL'}")
    return total


def task_4_2():
    print("\n" + "=" * 50)
    print("Task 4.2: GHZ State Generation")
    print("=" * 50)
    
    torch.manual_seed(42)
    model = CausalTransformer(
        vocab_size=2,
        d_model=32,
        num_heads=2,
        num_layers=1,
        max_seq_len=10
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model.train()
    
    print("Training on GHZ dataset: [(0,0,0), (1,1,1)]")
    for epoch in range(200):
        for first_token in [0, 1]:
            for step in [1, 2]:
                bos = torch.tensor([[model.bos_idx]], dtype=torch.long)
                x = torch.tensor([[first_token] * step], dtype=torch.long)
                context = torch.cat([bos, x], dim=1)
                target = torch.tensor([first_token], dtype=torch.long)
                logits = model(context)
                logits_valid = logits[0, -1, :2]
                loss = F.cross_entropy(logits_valid.unsqueeze(0), target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}")
    
    model.eval()
    
    print("\nSampling 100 sequences...")
    samples = []
    with torch.no_grad():
        for _ in range(100):
            first_token = torch.randint(0, 2, (1,)).item()
            bos = torch.tensor([[model.bos_idx]], dtype=torch.long)
            seq = [first_token]
            for i in range(2):
                context = torch.cat([bos, torch.tensor([seq], dtype=torch.long)], dim=1)
                logits = model(context)
                probs = F.softmax(logits[0, -1, :2], dim=-1)
                probs_valid = probs / probs.sum()
                token = torch.multinomial(probs_valid, 1).item()
                seq.append(token)
            samples.append(tuple(seq))
    
    from collections import Counter
    counts = Counter(samples)
    print("\nEmpirical distribution:")
    for seq, count in sorted(counts.items()):
        print(f"  {seq}: {count} ({count}%)")
    
    ghz_count = counts.get((0, 0, 0), 0) + counts.get((1, 1, 1), 0)
    print(f"\nGHZ states (000, 111): {ghz_count}%")
    print(f"Other states: {100 - ghz_count}%")


if __name__ == "__main__":
    task_4_1()
    task_4_2()