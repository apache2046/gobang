"""Generator-based Monte Carlo Tree Search with bug fixes.

Key features preserved from original:
- Generator pattern for async batched inference
- Stone-count indexing for efficient tree management
- Dihedral symmetry exploitation (8-fold)
- Adaptive Dirichlet noise

Bug fixes applied:
- Correct value sign in backpropagation
- Fixed temperature-based policy selection
- Proper Dirichlet noise on all valid positions
"""

import numpy as np
import random
from collections import defaultdict
from typing import Generator, Tuple, Optional
from config import config


class MCTS:
    """Monte Carlo Tree Search with neural network guidance.

    Uses generator pattern to yield states for batched inference.
    """

    def __init__(
        self,
        game,
        c_puct: float = config.c_puct,
        dirichlet_alpha: float = config.dirichlet_alpha,
        dirichlet_weight: float = config.dirichlet_weight,
        selfplay: bool = True,
        use_symmetry: bool = True,
    ):
        self.game = game
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.selfplay = selfplay
        self.use_symmetry = use_symmetry

        num_positions = config.board_size * config.board_size

        # Statistics indexed by stone count for efficient partial resets
        # psa[stones][state_key] = prior probabilities (225,)
        # wsa[stones][state_key] = cumulative values (225,)
        # nsa[stones][state_key] = visit counts (225,)
        # ns[stones][state_key] = total visits for state
        self.psa = [dict() for _ in range(num_positions)]
        self.wsa = [dict() for _ in range(num_positions)]
        self.nsa = [dict() for _ in range(num_positions)]
        self.ns = [defaultdict(int) for _ in range(num_positions)]

    def clear_statistics(self, stones: Optional[int] = None):
        """Clear MCTS statistics.

        Args:
            stones: If provided, only clear statistics for this stone count.
                   If None, clear all statistics.
        """
        num_positions = config.board_size * config.board_size
        if stones is not None:
            self.psa[stones] = dict()
            self.wsa[stones] = dict()
            self.nsa[stones] = dict()
            self.ns[stones] = defaultdict(int)
        else:
            self.psa = [dict() for _ in range(num_positions)]
            self.wsa = [dict() for _ in range(num_positions)]
            self.nsa = [dict() for _ in range(num_positions)]
            self.ns = [defaultdict(int) for _ in range(num_positions)]

    def search(
        self,
        state: np.ndarray,
        stones: int,
        level: int = 0
    ) -> Generator[np.ndarray, Tuple[np.ndarray, float], float]:
        """MCTS search using generator pattern.

        Yields states for neural network inference, receives (policy, value) via send().

        Args:
            state: Current board state (15, 15, 5)
            stones: Number of stones on board
            level: Tree depth (0 = root)

        Yields:
            state: Board state needing inference (possibly transformed)

        Returns:
            value: Value from current player's perspective
        """
        sk = self.game.state2key(state)

        psa = self.psa[stones]
        wsa = self.wsa[stones]
        nsa = self.nsa[stones]
        ns = self.ns[stones]

        # Leaf node - needs neural network evaluation
        if ns[sk] == 0:
            # Apply dihedral symmetry transformation
            if self.use_symmetry:
                transform_state, rot, flip_h, flip_v = self._random_transform(state)
            else:
                transform_state = state
                rot, flip_h, flip_v = 0, False, False

            # Yield state for batched inference
            policy, value = yield transform_state

            # Inverse transform policy
            if self.use_symmetry:
                policy = self._inverse_transform_policy(policy, rot, flip_h, flip_v)

            # Initialize statistics
            psa[sk] = policy.astype(np.float32)
            wsa[sk] = np.zeros(config.board_positions, dtype=np.float32)
            nsa[sk] = np.zeros(config.board_positions, dtype=np.float32)

            # Add Dirichlet noise at root for exploration
            if level == 0 and self.selfplay:
                self._add_dirichlet_noise(state, sk, psa)

            ns[sk] = 1

            # Return value from current player's perspective
            return -value  # Negate because value is from NN's perspective

        # Internal node - select action via UCB
        valid_mask = self.game.valid_actions(state)
        visit_counts = nsa[sk]
        total_visits = ns[sk]

        # UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        q_values = np.where(
            visit_counts > 0,
            wsa[sk] / (visit_counts + 1e-8),
            0.0
        )
        exploration = self.c_puct * psa[sk] * np.sqrt(total_visits) / (1 + visit_counts)
        ucb = q_values + exploration

        # Mask invalid actions
        ucb[~valid_mask] = -np.inf

        # Select action with highest UCB
        action = int(np.argmax(ucb))

        # Apply action
        next_state, done, reward = self.game.next_state(state, action)

        if done:
            # Terminal state - reward is from perspective of player who just moved
            value = reward
        else:
            # Recurse
            value = yield from self.search(next_state, stones + 1, level + 1)

        # Backpropagate
        # value is from child's perspective, negate for current player
        wsa[sk][action] += -value
        nsa[sk][action] += 1
        ns[sk] += 1

        # Return value for parent (negated)
        return -value

    def get_policy(self, state: np.ndarray, tau: float = 1.0) -> np.ndarray:
        """Get action probabilities based on visit counts.

        Args:
            state: Current board state
            tau: Temperature. tau=1.0 for exploration, tau->0 for greedy.

        Returns:
            policy: Action probabilities (225,)
        """
        stones = self.game.stone_count(state)
        sk = self.game.state2key(state)

        if sk not in self.nsa[stones]:
            # No visits - uniform over valid actions
            valid = self.game.valid_actions(state)
            return valid.astype(np.float32) / valid.sum()

        counts = self.nsa[stones][sk]
        valid = self.game.valid_actions(state)

        if tau < 0.01:
            # Greedy: select most visited action
            masked_counts = np.where(valid, counts, -np.inf)
            best = np.argmax(masked_counts)
            policy = np.zeros_like(counts)
            policy[best] = 1.0
        else:
            # Temperature-scaled softmax over visit counts
            # pi(a) proportional to N(s,a)^(1/tau)
            masked_counts = np.where(valid, counts, 0)
            if tau != 1.0:
                # Avoid overflow for very small tau
                masked_counts = np.power(masked_counts + 1e-10, 1.0 / tau)
            policy = masked_counts / (masked_counts.sum() + 1e-10)

        return policy.astype(np.float32)

    def _add_dirichlet_noise(self, state: np.ndarray, sk: bytes, psa: dict):
        """Add Dirichlet noise to prior for exploration at root.

        Uses adaptive alpha scaling based on number of valid positions.
        """
        valid_mask = self.game.valid_actions(state)
        num_valid = valid_mask.sum()

        if num_valid == 0:
            return

        # Adaptive alpha: more noise when board is empty
        board_size = config.board_size
        alpha = (board_size * board_size / num_valid) * self.dirichlet_alpha

        # Generate noise for valid positions
        noise = np.random.dirichlet(alpha * np.ones(num_valid))

        # Apply noise: (1 - weight) * prior + weight * noise
        prior = psa[sk].copy()
        prior[valid_mask] = (
            (1 - self.dirichlet_weight) * prior[valid_mask] +
            self.dirichlet_weight * noise
        )
        psa[sk] = prior

    def _random_transform(self, state: np.ndarray) -> Tuple[np.ndarray, int, bool, bool]:
        """Apply random dihedral transformation (8-fold symmetry).

        Returns:
            transformed_state: Transformed board state
            rot: Number of 90-degree rotations (0-3)
            flip_h: Whether horizontal flip was applied
            flip_v: Whether vertical flip was applied
        """
        transformed = state.copy()
        rot = random.randint(0, 3)
        flip_h = random.choice([True, False])
        flip_v = random.choice([True, False])

        if rot > 0:
            transformed = np.rot90(transformed, rot, axes=(0, 1))
        if flip_h:
            transformed = np.flip(transformed, axis=1)
        if flip_v:
            transformed = np.flip(transformed, axis=0)

        # Make contiguous copy to avoid negative stride issues with torch
        transformed = np.ascontiguousarray(transformed)

        return transformed, rot, flip_h, flip_v

    def _inverse_transform_policy(
        self,
        policy: np.ndarray,
        rot: int,
        flip_h: bool,
        flip_v: bool
    ) -> np.ndarray:
        """Apply inverse transformation to policy.

        Args:
            policy: Policy from NN (225,) or (15, 15)
            rot: Number of 90-degree rotations that were applied
            flip_h: Whether horizontal flip was applied
            flip_v: Whether vertical flip was applied

        Returns:
            policy: Inverse-transformed policy (225,)
        """
        size = config.board_size
        p = policy.reshape(size, size)

        # Apply inverse transforms in reverse order
        if flip_v:
            p = np.flip(p, axis=0)
        if flip_h:
            p = np.flip(p, axis=1)
        if rot > 0:
            p = np.rot90(p, -rot)  # Negative rotation to undo

        return p.flatten()


def run_mcts(
    state: np.ndarray,
    game,
    model,
    num_simulations: int = config.num_simulations,
    c_puct: float = config.c_puct,
    selfplay: bool = False,
    tau: float = 1.0,
) -> np.ndarray:
    """Run MCTS and return policy.

    Convenience function for single-state MCTS with inline inference.

    Args:
        state: Board state
        game: Game instance
        model: Neural network model
        num_simulations: Number of MCTS simulations
        c_puct: Exploration constant
        selfplay: Whether to add exploration noise
        tau: Temperature for policy

    Returns:
        policy: Action probabilities (225,)
    """
    mcts = MCTS(game, c_puct=c_puct, selfplay=selfplay)
    stones = game.stone_count(state)

    for _ in range(num_simulations):
        gen = mcts.search(state, stones)
        try:
            infer_state = next(gen)
            while True:
                policy, value = model.infer(infer_state)
                infer_state = gen.send((policy, value))
        except StopIteration:
            pass

    return mcts.get_policy(state, tau=tau)
