"""Game interface for playing against the AI via Flask backend."""

import sys
import os
import numpy as np
import torch

# Add parent directory for back_end import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from game import GoBang
from model import PolicyValueNet, load_model
from mcts import MCTS


class GoBangAI:
    """AI player interface compatible with Flask backend."""

    def __init__(self, model_path: str = None, num_simulations: int = 800):
        """Initialize AI with model.

        Args:
            model_path: Path to model checkpoint. If None, uses random policy.
            num_simulations: MCTS simulations per move.
        """
        self.game = GoBang()
        self.num_simulations = num_simulations
        self.state = None

        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = load_model(model_path, device=config.device)
        else:
            print("No model loaded, using random initialization")
            self.model = PolicyValueNet().to(config.device)
            self.model.eval()

    def board_size(self, size: int):
        """Initialize board with given size.

        Args:
            size: Board size (must be 15)
        """
        assert size == config.board_size, f"Only size {config.board_size} supported"
        self.state = self.game.start_state()
        print(f"Board initialized: {size}x{size}")

    def clearboard(self):
        """Reset the board to initial state."""
        self.state = self.game.start_state()
        print("Board cleared")

    def play(self, actor: int, pos: list) -> list:
        """Apply a player move.

        Args:
            actor: Player identifier (1 = black, -1 = white)
            pos: Position [x, y]

        Returns:
            [actor, pos, value, patterns]
        """
        if self.state is None:
            self.state = self.game.start_state()

        x, y = pos
        action = y * config.board_size + x

        # Validate move
        valid = self.game.valid_actions(self.state)
        if not valid[action]:
            print(f"Invalid move: ({x}, {y})")
            return [actor, pos, 0, []]

        self.state, done, reward = self.game.next_state(self.state, action)

        # Get value estimate
        _, value = self.model.infer(self.state)

        print(f"Player {actor} plays ({x}, {y}), value={value:.3f}, done={done}")

        return [actor, pos, float(value), []]

    def genmove(self, actor: int) -> list:
        """Generate AI move.

        Args:
            actor: Player identifier requesting move

        Returns:
            [actor, pos, value, patterns]
        """
        if self.state is None:
            self.state = self.game.start_state()

        # Run MCTS
        mcts = MCTS(
            self.game,
            c_puct=config.c_puct,
            selfplay=False,
            use_symmetry=True,
        )

        stones = self.game.stone_count(self.state)

        # Run simulations with inline inference
        for _ in range(self.num_simulations):
            gen = mcts.search(self.state, stones)
            try:
                infer_state = next(gen)
                while True:
                    policy, value = self.model.infer(infer_state)
                    infer_state = gen.send((policy, value))
            except StopIteration:
                pass

        # Get greedy policy (low temperature for play)
        policy = mcts.get_policy(self.state, tau=0.1)
        action = int(np.argmax(policy))

        x = action % config.board_size
        y = action // config.board_size
        pos = [x, y]

        # Apply move
        self.state, done, reward = self.game.next_state(self.state, action)

        # Get value estimate
        _, value = self.model.infer(self.state)

        print(f"AI plays ({x}, {y}), confidence={policy[action]:.3f}, value={value:.3f}")

        return [actor, pos, float(value), []]

    def get_board_state(self) -> np.ndarray:
        """Return current board state."""
        return self.state


def main():
    """Start the game server."""
    import argparse

    parser = argparse.ArgumentParser(description="GoBang AI Server")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="models/best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--simulations", "-s",
        type=int,
        default=800,
        help="MCTS simulations per move"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Server port"
    )
    args = parser.parse_args()

    # Import backend
    from back_end import serv

    # Create AI
    ai = GoBangAI(
        model_path=args.model,
        num_simulations=args.simulations,
    )

    print(f"Starting server on port {args.port}...")
    print(f"Model: {args.model}")
    print(f"Simulations: {args.simulations}")

    # Start server
    serv(
        port=args.port,
        boardsize_cb=ai.board_size,
        clearboard_cb=ai.clearboard,
        play_cb=ai.play,
        genmove_cb=ai.genmove,
    )


if __name__ == "__main__":
    main()
