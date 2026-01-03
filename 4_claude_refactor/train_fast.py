"""Optimized training with proper batched MCTS.

Key optimization: Batch N simulations at once by collecting leaf states.
"""

import os
import time
import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import config
from game import GoBang
from model import PolicyValueNet
from mcts import MCTS


def batch_infer(model, states: List[np.ndarray], device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Batch inference on multiple states."""
    if not states:
        return np.array([]), np.array([])

    with torch.no_grad():
        x = np.stack(states, axis=0)
        x = torch.from_numpy(x).float()
        x = x.permute(0, 3, 1, 2).to(device)
        policies, values = model(x)
        return policies.cpu().numpy(), values.squeeze(-1).cpu().numpy()


def run_mcts_batched(state, game, model, device, mcts, num_simulations, batch_size=64):
    """Run MCTS with batched leaf evaluation.

    Collects up to batch_size leaf states, evaluates them together.
    """
    stones = game.stone_count(state)
    sims_done = 0

    while sims_done < num_simulations:
        # Collect batch of leaf states
        leaf_states = []
        generators = []

        for _ in range(min(batch_size, num_simulations - sims_done)):
            gen = mcts.search(state, stones)
            try:
                leaf_state = next(gen)
                leaf_states.append(leaf_state)
                generators.append(gen)
            except StopIteration:
                # No leaf node needed (hit cached node)
                sims_done += 1

        if leaf_states:
            # Batch inference
            policies, values = batch_infer(model, leaf_states, device)

            # Resume generators
            for i, gen in enumerate(generators):
                try:
                    gen.send((policies[i], values[i]))
                except StopIteration:
                    pass
                sims_done += 1


def play_game(game, model, device, num_simulations=100, batch_size=64):
    """Play one self-play game."""
    state = game.start_state()
    samples = []
    mcts = MCTS(game, selfplay=True, c_puct=config.c_puct)
    move_count = 0

    while True:
        stones = game.stone_count(state)

        # Run batched MCTS
        run_mcts_batched(state, game, model, device, mcts, num_simulations, batch_size)

        # Get policy
        tau = 1.0 if move_count < 15 else 0.1
        policy = mcts.get_policy(state, tau=tau)

        samples.append([state.copy(), policy.copy(), None])

        # Select action
        if tau > 0.5:
            action = np.random.choice(len(policy), p=policy)
        else:
            action = int(np.argmax(policy))

        state, done, reward = game.next_state(state, action)
        move_count += 1

        # Clear old stats
        if stones > 2:
            mcts.clear_statistics(stones - 2)

        if done:
            break

    # Assign values
    v = reward if reward != 0 else 0.0
    for j in reversed(range(len(samples))):
        samples[j][2] = v
        v = -v

    return samples, reward


def augment_batch(samples: List, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Augment and batch samples."""
    states, policies, values = [], [], []

    for state, policy, value in samples:
        s = torch.from_numpy(state).float()
        p = torch.from_numpy(policy).float().view(15, 15)

        rot = random.randint(0, 3)
        if rot > 0:
            s = torch.rot90(s, rot, dims=(0, 1))
            p = torch.rot90(p, rot, dims=(0, 1))

        if random.random() > 0.5:
            s = torch.flip(s, dims=[1])
            p = torch.flip(p, dims=[1])

        if random.random() > 0.5:
            s = torch.flip(s, dims=[0])
            p = torch.flip(p, dims=[0])

        states.append(s.permute(2, 0, 1))
        policies.append(p.flatten())
        values.append(value)

    states = torch.stack(states).to(device)
    policies = torch.stack(policies).to(device)
    values = torch.tensor(values, dtype=torch.float32).to(device)

    return states, policies, values


def train_step(model, optimizer, states, policies, values):
    """Training step."""
    model.train()
    pred_policies, pred_values = model(states)
    pred_values = pred_values.squeeze(-1)

    policy_loss = -torch.mean(
        torch.sum(policies * torch.log(pred_policies + 1e-8), dim=1)
    )
    value_loss = F.mse_loss(pred_values, values)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return policy_loss.item(), value_loss.item()


def main():
    print("=" * 70)
    print("AlphaZero GoBang - Batched MCTS Training")
    print("=" * 70)

    device = "cuda:0"
    print(f"Device: {device}")

    game = GoBang()
    model = PolicyValueNet().to(device)
    model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # Settings
    num_simulations = 200    # MCTS simulations per move
    mcts_batch_size = 128    # Batch size for MCTS leaf evaluation
    train_batch_size = 2048  # Training batch
    train_every = 2          # Train after N games
    train_steps = 40

    print(f"\nSettings:")
    print(f"  MCTS simulations: {num_simulations}")
    print(f"  MCTS batch size: {mcts_batch_size}")
    print(f"  Train batch size: {train_batch_size}")
    print("-" * 70)

    replay_buffer = deque(maxlen=100000)
    os.makedirs("models", exist_ok=True)

    total_games = 0
    total_samples = 0
    total_steps = 0
    start_time = time.time()

    try:
        while True:
            elapsed = time.time() - start_time

            # Play one game
            game_start = time.time()
            samples, reward = play_game(
                game, model, device,
                num_simulations=num_simulations,
                batch_size=mcts_batch_size
            )
            game_time = time.time() - game_start

            replay_buffer.extend(samples)
            total_games += 1
            total_samples += len(samples)

            winner = "B" if reward > 0 else ("W" if reward < 0 else "D")
            sps = len(samples) * num_simulations / game_time  # sims per second
            print(f"[{elapsed:5.0f}s] Game {total_games}: {len(samples)} moves, "
                  f"winner={winner}, {game_time:.1f}s, {sps:.0f} sims/s")

            # Train
            if total_games % train_every == 0 and len(replay_buffer) >= train_batch_size:
                print(f"\n{'='*25} Training (buffer={len(replay_buffer)}) {'='*25}")

                for step in range(train_steps):
                    batch = random.sample(list(replay_buffer), train_batch_size)
                    states, policies, values = augment_batch(batch, device)

                    pi_loss, v_loss = train_step(model, optimizer, states, policies, values)
                    total_steps += 1

                    if step % 10 == 0:
                        print(f"  Step {step+1:2d}/{train_steps}: "
                              f"pi_loss={pi_loss:.4f}, v_loss={v_loss:.4f}")

                model.eval()
                path = f"models/game_{total_games:04d}.pt"
                torch.save(model.state_dict(), path)
                print(f"  Saved: {path}")
                print("=" * 60 + "\n")

            if elapsed > 600:
                print("\n10 minutes elapsed.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Games: {total_games}")
    print(f"Samples: {total_samples}")
    print(f"Training steps: {total_steps}")
    print(f"Avg sims/s: {total_samples * num_simulations / elapsed:.0f}")
    torch.save(model.state_dict(), "models/final.pt")
    print("Saved: models/final.pt")


if __name__ == "__main__":
    main()
