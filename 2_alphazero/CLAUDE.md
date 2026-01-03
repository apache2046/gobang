# AlphaZero GoBang Implementation

## Overview
GoBang (Gomoku/Five-in-a-Row) AI using AlphaZero algorithm: neural network + Monte Carlo Tree Search on a 15x15 board.

## Directory Structure

```
2_alphazero/
├── game*.py          # Game logic (board state, win detection)
├── mcts*.py          # MCTS variants (13 versions)
├── model*.py         # Neural networks (9 versions)
├── sim*.py           # Self-play & training (Ray distributed)
├── evaluate*.py      # Pattern-based board evaluation
├── infer_srv.py      # PyTorch inference server
├── onnx_trtinfer_server_int8.py  # TensorRT INT8 inference
├── play4.py          # Game interface for playing
├── exportonnx.py     # ONNX model export
├── models*/          # Saved checkpoints
└── log*/             # Training logs
```

## Key Files

| File | Purpose |
|------|---------|
| `game.py` | Core game: 15x15 board, 4-channel state, win detection |
| `mcts7.py` | Best MCTS: Dirichlet noise, dihedral symmetry, generator-based |
| `model6.py` | ResNet-10 with 5-channel input, 128 filters |
| `sim8.py` | Full training pipeline with Ray actors |

## Architecture

### Neural Network (model6.py)
```
Input: (batch, 5, 15, 15)
  → Conv2d(5→128) + BatchNorm + ReLU
  → 10x ResidualBlock(128)
  → Policy Head: Conv→Linear→Softmax (225 outputs)
  → Value Head: Conv→Linear→Tanh (1 output)
```

### MCTS Formula
```
UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (N(s,a) + 1)
```

### Training Pipeline
1. Self-play generates (state, policy, outcome) samples
2. Replay buffer stores up to 10K samples
3. Train every 200 episodes, 50 iterations per batch
4. 8-fold data augmentation (rotations + flips)

## Important Constants

- Board: 15x15 = 225 positions
- MCTS playouts: 100K-10K (configurable)
- c_puct: 0.001-5.0 (exploration constant)
- Dirichlet alpha: 0.05-0.3
- Temperature: 0.8 (early) → 0.05 (late game)
- Batch size: 1024
- Learning rate: 1e-4 (AdamW)

## Running

### Training (Ray)
```bash
python sim8.py
```

### Playing
```python
from play4 import AI
ai = AI()
move = ai.play(board_state)
```

### Export to ONNX
```bash
python exportonnx.py
```

## File Versions

- `game.py` (4-ch), `game2.py` (5-ch), `game3.py` (3-ch): Different state encodings
- `mcts.py` → `mcts8.py`: Progressive improvements (noise, symmetry, generators)
- `model.py` → `model9.py`: CNN → ResNet evolution

## Dependencies
- PyTorch
- Ray (distributed training)
- NumPy
- ONNX, TensorRT (optional, for deployment)
