"""GoBang (Gomoku) game logic with 5-channel state encoding."""

import numpy as np
from typing import Tuple, Optional
from config import config


def _check_line(board: np.ndarray, x: int, y: int, dx: int, dy: int, player: int) -> int:
    """Count consecutive stones in one direction."""
    count = 0
    size = config.board_size
    nx, ny = x + dx, y + dy
    while 0 <= nx < size and 0 <= ny < size and board[ny, nx] == player:
        count += 1
        nx += dx
        ny += dy
    return count


def _has_five(board: np.ndarray, x: int, y: int, player: int) -> bool:
    """Check if placing at (x, y) creates 5-in-a-row for player."""
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # horizontal, vertical, diagonals

    for dx, dy in directions:
        count = 1  # Count the placed stone
        count += _check_line(board, x, y, dx, dy, player)
        count += _check_line(board, x, y, -dx, -dy, player)
        if count >= config.win_length:
            return True
    return False


class GoBang:
    """GoBang game with 5-channel state encoding.

    State channels:
        0: Black stones (1 where black, 0 elsewhere)
        1: White stones (1 where white, 0 elsewhere)
        2: Previous black stones (before last move)
        3: Previous white stones (before last move)
        4: Current player (1 = black's turn, 0 = white's turn)
    """

    def __init__(self):
        self.size = config.board_size
        self.num_actions = self.size * self.size

    def start_state(self) -> np.ndarray:
        """Return initial empty board state. Black plays first."""
        state = np.zeros((self.size, self.size, 5), dtype=np.float32)
        state[:, :, 4] = 1  # Black's turn
        return state

    def next_state(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, bool, float]:
        """Apply action and return (new_state, is_terminal, reward).

        Args:
            state: Current board state (15, 15, 5)
            action: Position index (0-224), where action = y * size + x

        Returns:
            new_state: Updated board state
            done: True if game ended
            reward: 1.0 if current player won, 0.0 otherwise
        """
        state = state.copy()
        y, x = divmod(action, self.size)

        # Save previous board positions
        state[:, :, 2] = state[:, :, 0]  # prev_black = black
        state[:, :, 3] = state[:, :, 1]  # prev_white = white

        # Determine current player
        is_black = state[0, 0, 4] > 0.5

        if is_black:
            state[y, x, 0] = 1  # Place black stone
            player_board = state[:, :, 0]
        else:
            state[y, x, 1] = 1  # Place white stone
            player_board = state[:, :, 1]

        # Check for win
        win = _has_five(player_board, x, y, 1)

        # Check for draw (board full)
        occupied = state[:, :, 0] + state[:, :, 1]
        draw = np.all(occupied > 0)

        # Switch player
        state[:, :, 4] = 0 if is_black else 1

        done = win or draw
        reward = 1.0 if win else 0.0

        return state, done, reward

    def valid_actions(self, state: np.ndarray) -> np.ndarray:
        """Return boolean mask of valid (empty) positions."""
        occupied = state[:, :, 0] + state[:, :, 1]
        return (occupied == 0).flatten()

    def valid_action_indices(self, state: np.ndarray) -> np.ndarray:
        """Return array of valid action indices."""
        return np.where(self.valid_actions(state))[0]

    def state2key(self, state: np.ndarray) -> bytes:
        """Convert state to hashable key for MCTS cache."""
        # Only use black/white channels (0,1) - turn is implicit in stone count
        board = (state[:, :, 0] > 0.5).astype(np.uint8) + \
                (state[:, :, 1] > 0.5).astype(np.uint8) * 2
        return board.tobytes()

    def current_player(self, state: np.ndarray) -> int:
        """Return current player: 1 for black, -1 for white."""
        return 1 if state[0, 0, 4] > 0.5 else -1

    def stone_count(self, state: np.ndarray) -> int:
        """Return number of stones on the board."""
        return int(np.sum(state[:, :, 0]) + np.sum(state[:, :, 1]))

    def render(self, state: np.ndarray) -> str:
        """Render board as ASCII string."""
        lines = []
        lines.append("   " + " ".join(f"{i:2d}" for i in range(self.size)))
        for y in range(self.size):
            row = f"{y:2d} "
            for x in range(self.size):
                if state[y, x, 0] > 0.5:
                    row += " X "
                elif state[y, x, 1] > 0.5:
                    row += " O "
                else:
                    row += " . "
            lines.append(row)
        player = "Black (X)" if state[0, 0, 4] > 0.5 else "White (O)"
        lines.append(f"Next: {player}")
        return "\n".join(lines)
