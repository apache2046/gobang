# AlphaZero GoBang - Refactored

Clean, bug-fixed AlphaZero implementation for GoBang (Gomoku/Five-in-a-Row).

## Features

- **Generator-based MCTS** - Async batching via `yield from` pattern
- **Dihedral symmetry** - 8-fold augmentation at inference and training
- **Adaptive Dirichlet noise** - Scales with board occupancy
- **Ray distributed training** - Parallel self-play with batched GPU inference
- **ResNet-10 architecture** - 6.4M parameters, 128 channels

## Quick Start

```bash
# Activate environment
conda activate torch2.9

# Training (distributed with Ray)
python train.py

# Play against AI
python play.py --model models/epoch_0001.pt --port 8080

# Then open http://localhost:8080
```

## Bug Fixes (vs original 2_alphazero/)

| Issue | Original Bug | Fix Applied |
|-------|--------------|-------------|
| Data augmentation | `torch.rot90(s, r)` - return discarded | `s = torch.rot90(s, r, dims=(0,1))` |
| MCTS backprop | Value sign inconsistent | Proper negation for minimax |
| Temperature | `tau=1` had no effect | Correct `counts^(1/tau)` scaling |
| Value targets | Inconsistent 0.8 discount | Standard no-discount (AlphaZero) |

## Architecture

### Neural Network
```
Input (5, 15, 15)
    ├── Black stones
    ├── White stones
    ├── Previous black
    ├── Previous white
    └── Current player
         ↓
Conv2d(5→128) + BN + ReLU
         ↓
10× ResidualBlock(128)
         ↓
    ┌────┴────┐
Policy Head  Value Head
(225 probs)  (scalar)
```

### Training Pipeline
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Self-Play (4×) │────▶│ Inference (2×)  │────▶│ Training (1×)   │
│  32 games each  │     │ GPU batching    │     │ Replay buffer   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               Weight broadcast
```

## Configuration

All hyperparameters in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_simulations` | 800 | MCTS playouts per move |
| `c_puct` | 1.5 | Exploration constant |
| `batch_size` | 2048 | Training batch |
| `lr` | 2e-3 | Initial learning rate |
| `replay_buffer_size` | 100K | Max stored samples |
| `num_res_blocks` | 10 | ResNet depth |
| `num_filters` | 128 | Channel width |

## Files

| File | Lines | Description |
|------|-------|-------------|
| `config.py` | 60 | Hyperparameters dataclass |
| `game.py` | 130 | GoBang rules, 5-channel state |
| `model.py` | 160 | ResNet-10 policy-value net |
| `mcts.py` | 330 | Generator-based MCTS |
| `train.py` | 450 | Ray distributed training |
| `play.py` | 150 | Flask backend interface |

## Training Output

### Fast Training (train_fast.py) - Recommended
```bash
python train_fast.py  # Batched MCTS, optimized
```

**10-Minute Optimized Training Results:**
```
======================================================================
AlphaZero GoBang - Batched MCTS Training
======================================================================
Settings:
  MCTS simulations: 200
  MCTS batch size: 128
  Train batch size: 2048
----------------------------------------------------------------------
[    0s] Game 1: 224 moves, winner=B, 6.2s, 7217 sims/s
[   72s] Game 11: 216 moves, winner=B, 4.1s, 10638 sims/s
...
[  600s] Game 61: 216 moves, winner=B, 4.1s, 10481 sims/s

SUMMARY: 61 games, 13357 samples, 1040 train steps, 10K sims/s
```

**Loss Curve (Optimized):**
| Game | Policy Loss | Value Loss |
|------|-------------|------------|
| 10 | 14.2 → 5.7 | 2.1 → 2.0 |
| 20 | 2.0 → 1.7 | 2.0 |
| 40 | 0.99 → 0.90 | 2.0 |
| 60 | 0.74 → 0.70 | 2.0 |

**Key Improvement:** Policy loss dropped from 14.2 → 0.70 (95% reduction!)

---

### Simple Training (train_simple.py)
```bash
python train_simple.py  # Single-process, slower but simpler
```

**10-Minute Simple Training Results:**
```
============================================================
AlphaZero GoBang Training (Simple Mode)
============================================================
Device: cuda:0

Settings: sims=50, batch=256, train_every=5, train_steps=20
------------------------------------------------------------
[   0.0s] Game 1: 80 moves, winner=Black, buffer=80, time=8.3s
[  31.9s] Game 5: 77 moves, winner=Black, buffer=405, time=7.2s

======================================== Training ========================================
  Step  1/20: policy_loss=13.4273, value_loss=1.8708
  Step 16/20: policy_loss=6.1740, value_loss=2.0781
  Saved: models/game_0005.pt
================================================================================

[ 106.5s] Game 10: 219 moves, winner=Black, buffer=1503, time=17.4s

======================================== Training ========================================
  Step  1/20: policy_loss=6.8752, value_loss=2.1406
  Step 16/20: policy_loss=5.6742, value_loss=1.8906
================================================================================

[ 282.8s] Game 20: 218 moves, winner=Black, buffer=3700, time=16.4s

======================================== Training ========================================
  Step  1/20: policy_loss=5.5371, value_loss=2.0000
  Step 16/20: policy_loss=5.3822, value_loss=2.0781
================================================================================

[ 470.8s] Game 30: 222 moves, winner=Black, buffer=5899, time=19.4s

======================================== Training ========================================
  Step  1/20: policy_loss=5.0647, value_loss=1.9531
  Step 16/20: policy_loss=5.1188, value_loss=2.2344
================================================================================

============================================================
TRAINING SUMMARY
============================================================
Total time: 621.4s (10.4 min)
Games played: 37
Training steps: 140
Final buffer size: 7425
Samples/game: 200.7
```

**Loss Curve Summary:**
| Training Round | Policy Loss | Value Loss |
|---------------|-------------|------------|
| Round 1 (game 5) | 13.4 → 6.2 | 1.9 → 2.1 |
| Round 2 (game 10) | 6.9 → 5.7 | 2.1 → 1.9 |
| Round 4 (game 20) | 5.5 → 5.4 | 2.0 → 2.1 |
| Round 6 (game 30) | 5.1 → 5.1 | 2.0 → 2.2 |
| Round 7 (game 35) | 5.0 → 5.2 | 2.0 → 2.1 |

**Observations:**
- Policy loss dropped from 13.4 to ~5.0 (62% reduction)
- Value loss stable around 2.0 (expected for ±1 targets)
- ~17-19 seconds per game with 50 MCTS simulations
- Buffer accumulates ~200 samples per game

### Distributed Training (train.py)
```bash
python train.py  # Ray distributed, full training
```

## Requirements

- Python 3.10+
- PyTorch 2.x
- Ray
- NumPy

## License

MIT
