# AlphaZero GoBang - Refactored (Stage 1)

Clean refactor of the AlphaZero implementation with bug fixes and training optimizations.

## Quick Start

```bash
# Training (requires Ray)
python train.py

# Play against AI
python play.py --model models/best.pt --port 8080
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Centralized hyperparameters |
| `game.py` | 5-channel GoBang game logic |
| `model.py` | ResNet-10 policy-value network |
| `mcts.py` | Generator-based MCTS with bug fixes |
| `train.py` | Ray distributed training |
| `play.py` | Flask backend interface |

## Bug Fixes Applied

### 1. torch.rot90() Return Value (train.py)
```python
# BEFORE (bug):
torch.rot90(s, r)      # Return value discarded!

# AFTER (fixed):
s = torch.rot90(s, r, dims=(0, 1))
```

### 2. MCTS Value Sign (mcts.py)
```python
# BEFORE (bug):
wsa[sk][a] += v
return -v * discount  # Double negation issue

# AFTER (fixed):
wsa[sk][action] += -value  # Store from current player's view
return -value              # Negate for parent
```

### 3. Temperature Logic (mcts.py)
```python
# BEFORE (bug):
if tau == 1.0:
    pi = nsa ** (1/tau)  # 1/1 = 1, no effect!
else:
    # greedy

# AFTER (fixed):
if tau < 0.01:  # Near-zero = greedy
    pi[best] = 1.0
else:
    pi = counts ** (1.0 / tau)  # Proper temperature scaling
```

### 4. Consistent Value Targets (train.py)
```python
# BEFORE (inconsistent):
v = -v * 0.8  # Arbitrary discount

# AFTER (standard):
v = -v  # No discount (standard AlphaZero)
```

## Optimizations Added

- **Cosine LR schedule**: `CosineAnnealingLR` for smooth learning rate decay
- **Gradient clipping**: `clip_grad_norm_(1.0)` prevents exploding gradients
- **Kaiming init**: Proper weight initialization for ReLU networks
- **Recency sampling**: Newer samples weighted higher in replay buffer
- **Centralized config**: All hyperparameters in `config.py`

## Architecture

### Neural Network (ResNet-10)
```
Input: (batch, 5, 15, 15)
  → Conv2d(5→128) + BN + ReLU
  → 10× ResidualBlock(128)
  → Policy: Conv→Linear→Softmax (225)
  → Value: Conv→Linear→ReLU→Linear→Tanh (1)
```

### MCTS (Generator-based)
```python
def search(state, stones, level):
    if leaf_node:
        policy, value = yield state  # Batch externally
        return -value

    action = select_ucb(state)
    value = yield from search(next_state, stones+1, level+1)

    update_stats(state, action, value)
    return -value
```

### Training Pipeline (Ray)
```
Self-Play Workers (32×)
        ↓ yield states
Inference Servers (5×)
        ↓ samples
Training Server
        ↓ weights
Inference Servers (broadcast)
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_simulations` | 800 | MCTS playouts per move |
| `c_puct` | 1.5 | Exploration constant |
| `dirichlet_alpha` | 0.3 | Noise concentration (adaptive) |
| `batch_size` | 2048 | Training batch size |
| `lr` | 2e-3 | Initial learning rate |
| `replay_buffer_size` | 100K | Max samples stored |

## Preserved Designs

- Generator-based MCTS for async batching
- Stone-count indexing for efficient tree management
- Dihedral symmetry (8-fold) at inference and training
- Adaptive Dirichlet noise scaling
- 5-channel state encoding with history
- Ray distributed training architecture
