# AGENTS.md

This is a collection of AI & Quantum Science course homework assignments (hw1, hw2, hw3).
Code is primarily in Python with PyTorch and NumPy. The working directory is `hw2/`.

---

## Project Structure

```
homework/
├── hw1/          # HMM (Hidden Markov Model) exercises
├── hw2/          # Transformer exercises (current focus)
│   ├── exercise1_rope.py       # RoPE implementation
│   └── transformer_exercises_EN.ipynb  # Main notebook with all exercises
├── hw3/          # Tutorial files
└── AGENTS.md     # This file
```

---

## Running Code

### Python Scripts
```bash
python hw2/exercise1_rope.py
```

### Jupyter Notebooks
```bash
jupyter notebook hw2/transformer_exercises_EN.ipynb
```

### Running Single Test
For the RoPE exercise, tests are embedded in `if __name__ == "__main__"`:
```bash
python -c "
import torch
from exercise1_rope import rotate_half, get_rotary_cos_sin, apply_rotary_pos_emb

# Quick test
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
expected = torch.tensor([[-2.0, 1.0, -4.0, 3.0]])
assert torch.allclose(rotate_half(x), expected), 'rotate_half failed'
print('All tests passed!')
"
```

---

## Code Style Guidelines

### General Principles
- Write clean, readable code with descriptive variable names
- Include docstrings for all functions and classes
- Use type hints where beneficial
- Keep functions focused (single responsibility)

### Imports
```python
# Standard library first
import math
from typing import Optional

# Third-party libraries
import torch
import torch.nn as nn
import numpy as np
from scipy.special import logsumexp
```

### Formatting
- 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Use blank lines to separate logical sections
- Put space after commas: `x, y` not `x,y`

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Functions | snake_case | `rotate_half`, `apply_rotary_pos_emb` |
| Classes | PascalCase | `RoPE`, `MultiHeadAttention` |
| Variables | snake_case | `seq_len`, `head_dim`, `cos_emb` |
| Constants | UPPER_SNAKE | `MAX_SEQ_LEN`, `BASE` |
| Type variables | PascalCase | `torch.Tensor` |

### Type Hints
```python
def get_rotary_cos_sin(
    seq_len: int,
    dim: int,
    device: torch.device | None = None,
    base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
```

### Docstrings
Use Google-style docstrings:
```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half of the hidden dimensions.
    
    For dimensions [d_0, d_1, d_2, d_3, ...], this transforms to:
    [-d_1, d_0, -d_3, d_2, ...]
    
    Args:
        x: Input tensor of shape (..., dim) where dim is even
        
    Returns:
        Rotated tensor with same shape as input
    """
```

### PyTorch Conventions
- Use `torch.no_grad()` for inference
- Set random seeds: `torch.manual_seed(42)`
- Move to device: `tensor.to(device)` or `tensor.to(device=device)`
- Use `.to(dtype)` for type conversion
- Use `torch.allclose()` for floating-point comparisons with `atol`
- Register buffers with `self.register_buffer('name', tensor, persistent=False)`

### Numerical Stability
- Add epsilon for log operations: `np.log(x + 1e-10)`
- Check for NaN/Inf: `assert not torch.isnan(tensor).any()`
- Use `torch.isclose()` or `torch.allclose()` for float comparisons

### Error Handling
- Use assertions for invariants: `assert q.shape == expected_shape`
- Raise `NotImplementedError` for skeleton code
- Include descriptive error messages

### Testing
- Embed tests in `if __name__ == "__main__"` block
- Print test names and status: `print("[OK] Test passed!")`
- Use consistent separator lines: `print("-" * 40)`

---

## Transformer Exercise Reference

The main exercises in `transformer_exercises_EN.ipynb`:

1. **RoPE** (`exercise1_rope.py`): Implement `rotate_half`, `get_rotary_cos_sin`, `apply_rotary_pos_emb`
2. **Multi-Head Attention**: Implement QKV projection, scaled dot-product attention
3. **Sparse Attention**: Top-K selection for attention scores
4. **Autoregressive Transformer**: GHZ state generation with causal masking
5. **Mixture of Experts**: Top-1 routing with multiple FFN experts

Key shapes for MHA:
- Input: `(B, L, D)`
- QKV projection: `(B, L, H * D_k)`
- Reshape heads: `(B, H, L, D_k)`
- Attention output: `(B, H, L, D_k)`
- Concat: `(B, L, H * D_k)`
- Output: `(B, L, D)`

---

## Dependencies

```
torch>=2.0
numpy>=1.20
scipy
matplotlib
jupyter
```

Install via:
```bash
pip install torch numpy scipy matplotlib jupyter
```
