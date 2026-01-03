"""Centralized configuration for AlphaZero GoBang."""

from dataclasses import dataclass


@dataclass
class Config:
    """AlphaZero training and inference configuration."""

    # Board
    board_size: int = 15
    win_length: int = 5

    # State encoding
    num_channels: int = 5  # [black, white, prev_black, prev_white, turn]

    # Neural network
    num_res_blocks: int = 10
    num_filters: int = 128

    # MCTS
    num_simulations: int = 800
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25

    # Training
    batch_size: int = 2048
    lr: float = 2e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # Replay buffer
    replay_buffer_size: int = 100_000
    min_buffer_size: int = 10_000  # Min samples before training starts

    # Training schedule
    train_every_n_games: int = 100
    train_steps_per_update: int = 50
    save_checkpoint_every: int = 100

    # Temperature schedule
    tau_initial: float = 1.0
    tau_final: float = 0.1
    tau_decay_moves: int = 30

    # Self-play
    num_parallel_games: int = 128
    num_inference_workers: int = 5

    # Device
    device: str = "cuda:0"

    @property
    def board_positions(self) -> int:
        return self.board_size * self.board_size


# Default configuration instance
config = Config()
