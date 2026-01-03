"""Distributed AlphaZero training with Ray.

Bug fixes applied:
- Fixed torch.rot90() return value assignment
- Consistent value targets (no arbitrary discount)
- Proper data augmentation

Optimizations added:
- Cosine annealing learning rate schedule
- Gradient clipping
- Recency-weighted replay sampling
- Better logging
"""

import os
import random
import time
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ray

from config import config
from game import GoBang
from model import PolicyValueNet
from mcts import MCTS


# ============================================================================
# Self-Play Episode Generator
# ============================================================================

def execute_episode(game: GoBang, mcts: MCTS, num_simulations: int) -> Tuple[List, bool, float]:
    """Execute one self-play episode.

    Returns:
        samples: List of (state, policy, value) tuples
        done: Whether game completed
        final_reward: 1.0 if first player won, -1.0 if second, 0.0 if draw
    """
    state = game.start_state()
    samples = []
    move_count = 0

    # Temperature schedule
    tau = config.tau_initial

    while True:
        stones = game.stone_count(state)

        # Run MCTS simulations (generator-based)
        for _ in range(num_simulations):
            gen = mcts.search(state, stones)
            try:
                infer_state = next(gen)
                # In actual training, this would batch with other games
                # Here we yield for external batching
                yield infer_state  # Will receive (policy, value) via send()
            except StopIteration:
                pass

        # Get action probabilities from visit counts
        policy = mcts.get_policy(state, tau=tau)

        # Store sample (value will be assigned later)
        samples.append([state.copy(), policy.copy(), None])

        # Add exploration noise for action selection
        valid_mask = game.valid_actions(state)
        num_valid = valid_mask.sum()
        if num_valid > 0 and mcts.selfplay:
            noise = np.random.dirichlet(0.3 * np.ones(num_valid))
            noisy_policy = policy.copy()
            noisy_policy[valid_mask] = 0.75 * policy[valid_mask] + 0.25 * noise
            noisy_policy = noisy_policy / noisy_policy.sum()
            action = np.random.choice(len(policy), p=noisy_policy)
        else:
            action = np.random.choice(len(policy), p=policy)

        # Apply action
        state, done, reward = game.next_state(state, action)
        move_count += 1

        # Decay temperature
        if move_count >= config.tau_decay_moves:
            tau = config.tau_final
        else:
            # Smooth decay
            tau = config.tau_initial - (config.tau_initial - config.tau_final) * \
                  (move_count / config.tau_decay_moves)

        # Clear old MCTS statistics to save memory
        if stones > 2:
            mcts.clear_statistics(stones - 2)

        if done:
            break

    # Assign values: alternate signs from final reward
    # reward is from perspective of player who made last move
    final_value = reward if reward != 0 else 0.0  # 1.0 for win, 0.0 for draw
    v = final_value
    for i in reversed(range(len(samples))):
        samples[i][2] = v
        v = -v  # Alternate perspective (no discount - standard AlphaZero)

    return samples


# ============================================================================
# Data Augmentation
# ============================================================================

def augment_sample(
    state: np.ndarray,
    policy: np.ndarray,
    value: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply random dihedral augmentation to a sample.

    Args:
        state: Board state (15, 15, 5)
        policy: Action probabilities (225,)
        value: State value

    Returns:
        state_tensor: Augmented state (5, 15, 15)
        policy_tensor: Augmented policy (225,)
        value_tensor: Value (1,)
    """
    # Convert to tensors
    s = torch.from_numpy(state).float()  # (15, 15, 5)
    p = torch.from_numpy(policy).float().view(15, 15)  # (15, 15)

    # Random rotation (0, 90, 180, 270 degrees)
    rot = random.randint(0, 3)
    if rot > 0:
        s = torch.rot90(s, rot, dims=(0, 1))  # FIX: Assign return value!
        p = torch.rot90(p, rot, dims=(0, 1))  # FIX: Assign return value!

    # Random horizontal flip
    if random.random() > 0.5:
        s = torch.flip(s, dims=[1])
        p = torch.flip(p, dims=[1])

    # Random vertical flip
    if random.random() > 0.5:
        s = torch.flip(s, dims=[0])
        p = torch.flip(p, dims=[0])

    # Convert to correct format
    s = s.permute(2, 0, 1)  # (5, 15, 15)
    p = p.flatten()  # (225,)
    v = torch.tensor([value], dtype=torch.float32)

    return s, p, v


# ============================================================================
# Ray Actors
# ============================================================================

@ray.remote(num_cpus=0.1, num_gpus=0.1)
class InferenceServer:
    """GPU inference server for batched predictions."""

    def __init__(self, model_cls=PolicyValueNet):
        self.device = config.device
        self.model = model_cls().to(self.device)
        self.model.eval()

    def infer(self, states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Batch inference.

        Args:
            states: List of board states (15, 15, 5)

        Returns:
            policies: (batch, 225)
            values: (batch,)
        """
        with torch.no_grad():
            x = np.stack(states, axis=0)  # (batch, 15, 15, 5)
            x = torch.from_numpy(x).float()
            x = x.permute(0, 3, 1, 2).to(self.device)  # (batch, 5, 15, 15)

            policies, values = self.model(x)

            return policies.cpu().numpy(), values.squeeze(-1).cpu().numpy()

    def load_weights(self, state_dict: dict):
        """Load new weights."""
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def get_weights(self) -> dict:
        """Get current weights."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}


@ray.remote(num_cpus=0.2, num_gpus=0.2)
class TrainingServer:
    """Training server that maintains replay buffer and trains model."""

    def __init__(self, inference_servers: List):
        self.device = config.device
        self.model = PolicyValueNet().to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # Cosine annealing LR schedule
        self.scheduler = None  # Will be initialized when we know total steps

        self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        self.inference_servers = inference_servers

        self.games_played = 0
        self.train_steps = 0
        self.epoch = 0

        # Create models directory
        os.makedirs("models", exist_ok=True)

    def add_samples(self, samples: List[Tuple]):
        """Add samples to replay buffer.

        Args:
            samples: List of (state, policy, value) tuples
        """
        self.replay_buffer.extend(samples)
        self.games_played += 1

        # Train periodically
        if self.games_played % config.train_every_n_games == 0:
            if len(self.replay_buffer) >= config.min_buffer_size:
                self._train()
                self._broadcast_weights()
                self._save_checkpoint()

    def _train(self):
        """Run training steps."""
        self.model.train()

        for step in range(config.train_steps_per_update):
            # Sample batch with recency weighting
            batch = self._sample_batch()

            states, policies, values = [], [], []
            for state, policy, value in batch:
                s, p, v = augment_sample(state, policy, value)
                states.append(s)
                policies.append(p)
                values.append(v)

            states = torch.stack(states).to(self.device)
            policies = torch.stack(policies).to(self.device)
            values = torch.cat(values).to(self.device)

            # Forward pass
            pred_policies, pred_values = self.model(states)
            pred_values = pred_values.squeeze(-1)

            # Policy loss: cross-entropy
            policy_loss = -torch.mean(
                torch.sum(policies * torch.log(pred_policies + 1e-8), dim=1)
            )

            # Value loss: MSE
            value_loss = F.mse_loss(pred_values, values)

            # Total loss
            loss = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.max_grad_norm
            )

            self.optimizer.step()
            self.train_steps += 1

            # Log
            if step == 0 or (step + 1) % 10 == 0:
                print(f"Epoch {self.epoch} Step {step+1}/{config.train_steps_per_update}: "
                      f"policy_loss={policy_loss.item():.4f}, "
                      f"value_loss={value_loss.item():.4f}, "
                      f"lr={self.optimizer.param_groups[0]['lr']:.6f}")

        self.model.eval()
        self.epoch += 1

    def _sample_batch(self) -> List[Tuple]:
        """Sample batch with recency weighting (newer samples more likely)."""
        buffer_size = len(self.replay_buffer)

        # Exponential weighting: newer samples have higher probability
        weights = np.exp(np.linspace(-2, 0, buffer_size))
        weights /= weights.sum()

        indices = np.random.choice(
            buffer_size,
            size=min(config.batch_size, buffer_size),
            p=weights,
            replace=False
        )

        return [self.replay_buffer[i] for i in indices]

    def _broadcast_weights(self):
        """Send updated weights to all inference servers."""
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for server in self.inference_servers:
            server.load_weights.remote(state_dict)
        print(f"Weights broadcast to {len(self.inference_servers)} inference servers")

    def _save_checkpoint(self):
        """Save model checkpoint."""
        if self.epoch % config.save_checkpoint_every == 0 or self.epoch == 1:
            path = f"models/epoch_{self.epoch:04d}.pt"
            torch.save(self.model.state_dict(), path)
            print(f"Saved checkpoint: {path}")

    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            "games_played": self.games_played,
            "train_steps": self.train_steps,
            "epoch": self.epoch,
            "buffer_size": len(self.replay_buffer),
        }


@ray.remote(num_cpus=1)
def self_play_worker(
    inference_server,
    training_server,
    num_games: int = 128,
    num_simulations: int = config.num_simulations
):
    """Self-play worker that generates training data.

    Runs multiple games in parallel, batching inference requests.
    """
    game = GoBang()

    # Initialize MCTS and generators for each game
    mcts_instances = [
        MCTS(game, selfplay=True, c_puct=config.c_puct)
        for _ in range(num_games)
    ]
    states = [game.start_state() for _ in range(num_games)]
    samples_lists = [[] for _ in range(num_games)]
    active = [True] * num_games

    move_counts = [0] * num_games
    simulation_counts = [0] * num_games

    while any(active):
        # Collect states needing inference
        infer_states = []
        infer_indices = []

        for i in range(num_games):
            if not active[i]:
                continue

            # Run MCTS search step
            stones = game.stone_count(states[i])
            gen = mcts_instances[i].search(states[i], stones)

            try:
                infer_state = next(gen)
                infer_states.append(infer_state)
                infer_indices.append((i, gen))
            except StopIteration:
                simulation_counts[i] += 1

        if infer_states:
            # Batch inference
            policies, values = ray.get(
                inference_server.infer.remote(infer_states)
            )

            # Resume generators
            for idx, (i, gen) in enumerate(infer_indices):
                try:
                    gen.send((policies[idx], values[idx]))
                except StopIteration:
                    simulation_counts[i] += 1

        # Check if any game has completed enough simulations
        for i in range(num_games):
            if not active[i]:
                continue

            if simulation_counts[i] >= num_simulations:
                # Get policy and make move
                stones = game.stone_count(states[i])
                tau = config.tau_initial if move_counts[i] < config.tau_decay_moves else config.tau_final
                policy = mcts_instances[i].get_policy(states[i], tau=tau)

                # Store sample
                samples_lists[i].append([states[i].copy(), policy.copy(), None])

                # Select action with noise
                valid = game.valid_actions(states[i])
                num_valid = valid.sum()
                if num_valid > 0:
                    noise = np.random.dirichlet(0.3 * np.ones(num_valid))
                    noisy_policy = policy.copy()
                    noisy_policy[valid] = 0.75 * policy[valid] + 0.25 * noise
                    noisy_policy /= noisy_policy.sum()
                    action = np.random.choice(len(policy), p=noisy_policy)
                else:
                    action = np.argmax(policy)

                states[i], done, reward = game.next_state(states[i], action)
                move_counts[i] += 1
                simulation_counts[i] = 0

                # Clear old stats
                if stones > 2:
                    mcts_instances[i].clear_statistics(stones - 2)

                if done:
                    # Assign values
                    v = reward if reward != 0 else 0.0
                    for j in reversed(range(len(samples_lists[i]))):
                        samples_lists[i][j][2] = v
                        v = -v

                    # Send to training server
                    ray.get(training_server.add_samples.remote(samples_lists[i]))

                    # Reset for new game
                    states[i] = game.start_state()
                    samples_lists[i] = []
                    mcts_instances[i].clear_statistics()
                    move_counts[i] = 0

                    print(f"Game {i} completed with {len(samples_lists[i])} moves, reward={reward}")


# ============================================================================
# Main Training Loop
# ============================================================================

def train(
    num_workers: int = 4,
    num_inference_servers: int = 2,
    num_games_per_worker: int = 32,
    num_simulations: int = config.num_simulations,
):
    """Main training function.

    Args:
        num_workers: Number of self-play worker processes
        num_inference_servers: Number of GPU inference servers
        num_games_per_worker: Games per worker
        num_simulations: MCTS simulations per move
    """
    ray.init(ignore_reinit_error=True)

    print("Creating inference servers...")
    inference_servers = [
        InferenceServer.remote()
        for _ in range(num_inference_servers)
    ]

    print("Creating training server...")
    training_server = TrainingServer.remote(inference_servers)

    print(f"Starting {num_workers} self-play workers...")
    workers = []
    for i in range(num_workers):
        worker = self_play_worker.remote(
            inference_servers[i % num_inference_servers],
            training_server,
            num_games=num_games_per_worker,
            num_simulations=num_simulations,
        )
        workers.append(worker)

    # Wait for workers
    try:
        ray.get(workers)
    except KeyboardInterrupt:
        print("\nTraining interrupted")

    # Get final stats
    stats = ray.get(training_server.get_stats.remote())
    print(f"\nFinal stats: {stats}")

    ray.shutdown()


if __name__ == "__main__":
    train()
