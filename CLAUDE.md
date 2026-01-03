# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GoBang (Gomoku/Five-in-a-Row) AI system with three implementation approaches: heuristic, AlphaZero, and MuZero. Includes a web frontend and Flask backend.

## Build & Run Commands

### Frontend
```bash
cd front_end
npm install
npm run build          # Production build
npm run build-dev      # Development build (webpack)
```

### Running the Game with AI

**Heuristic AI:**
```bash
cd 1_heuristic
python play.py         # Starts Flask server on port 8080
```

**AlphaZero AI:**
```bash
cd 2_alphazero
python play4.py        # Loads model from /home/apache/ray_run/models.10/28.pt
```

### AlphaZero Training (Ray distributed)
```bash
cd 2_alphazero
python sim8.py         # Requires Ray cluster
```

### Model Export
```bash
cd 2_alphazero
python exportonnx.py   # Export to ONNX format
```

## Architecture

```
Frontend (Preact)  ──HTTP──▶  Backend (Flask)  ──callbacks──▶  AI Engine
     │                              │
     └── dist/                      └── serv(boardsize_cb, clearboard_cb,
         built by webpack                    play_cb, genmove_cb)
```

### REST API Endpoints
- `POST /boardsize` - Initialize board
- `POST /clearboard` - Reset game
- `POST /play` - Place stone (params: `actor`, `pos`)
- `POST /genmove` - Generate AI move (params: `actor`)

### AI Callback Interface
Each AI module (`1_heuristic/play.py`, `2_alphazero/play4.py`) implements:
```python
board_size(size)      # Initialize board
clearboard()          # Reset board
play(actor, pos)      # Place stone
genmove(actor)        # Generate AI move → returns position
```

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `1_heuristic/` | Pattern-based evaluation AI (`board.py`, `evaluate.py`, `play.py`) |
| `2_alphazero/` | Neural network + MCTS AI (see `2_alphazero/CLAUDE.md` for details) |
| `3_muzero/` | Model-based RL (placeholder) |
| `back_end/` | Flask server with callback architecture |
| `front_end/` | Preact UI using @sabaki/shudan board component |

## AlphaZero Key Files

| File | Purpose |
|------|---------|
| `game.py` | Game logic: 15x15 board, 4-channel state, win detection |
| `mcts7.py` | Best MCTS: Dirichlet noise, dihedral symmetry, generator-based |
| `model6.py` | ResNet-10 with 5-channel input, 128 filters |
| `sim8.py` | Full Ray distributed training pipeline |
| `play4.py` | Game interface for inference |

### Neural Network (model6.py)
```
Input: (batch, 5, 15, 15) → Conv2d(5→128) → 10x ResidualBlock → Policy(225) + Value(1)
```

### MCTS Formula
```
UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (N(s,a) + 1)
```

## Key Constants

- Board: 15x15 = 225 positions
- Win condition: 5 consecutive stones
- MCTS playouts: 100K (training), 10K (inference)
- c_puct: 0.001-5.0
- Temperature: 0.8 (early) → 0.05 (late game)
- Training batch: 1024, LR: 1e-4 (AdamW)

## Dependencies

- Python: flask, torch, ray, numpy, numba
- Frontend: preact, @sabaki/shudan, axios, webpack
