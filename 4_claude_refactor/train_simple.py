"""Simple single-process training for testing and debugging.

No Ray, just straightforward self-play + training loop.
"""

import os
import time
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from config import config
from game import GoBang
from model import PolicyValueNet
from mcts import MCTS


def augment_sample(state, policy, value):
    """Apply random dihedral augmentation."""
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

    s = s.permute(2, 0, 1)
    p = p.flatten()
    v = torch.tensor([value], dtype=torch.float32)

    return s, p, v


def self_play_game(game, model, num_simulations=100):
    """Play one game and return samples."""
    state = game.start_state()
    samples = []
    move_count = 0

    mcts = MCTS(game, selfplay=True, c_puct=config.c_puct)

    while True:
        stones = game.stone_count(state)

        # Run MCTS
        for _ in range(num_simulations):
            gen = mcts.search(state, stones)
            try:
                infer_state = next(gen)
                while True:
                    policy, value = model.infer(infer_state)
                    infer_state = gen.send((policy, value))
            except StopIteration:
                pass

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
    for i in reversed(range(len(samples))):
        samples[i][2] = v
        v = -v

    return samples, reward


def train_step(model, optimizer, batch, device):
    """Single training step."""
    states, policies, values = [], [], []
    for state, policy, value in batch:
        s, p, v = augment_sample(state, policy, value)
        states.append(s)
        policies.append(p)
        values.append(v)

    states = torch.stack(states).to(device)
    policies = torch.stack(policies).to(device)
    values = torch.cat(values).to(device)

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
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()

    return policy_loss.item(), value_loss.item()


def main():
    print("=" * 60)
    print("AlphaZero GoBang Training (Simple Mode)")
    print("=" * 60)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    game = GoBang()
    model = PolicyValueNet().to(device)
    model.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    replay_buffer = deque(maxlen=50000)
    os.makedirs("models", exist_ok=True)

    # Training settings - optimized for RTX 4090 (24GB VRAM)
    num_simulations = 400  # Stronger play (standard AlphaZero uses 800)
    batch_size = 4096      # Large batch for GPU utilization
    train_every = 3        # Train more frequently
    train_steps = 30       # More steps per training round

    total_games = 0
    total_steps = 0
    start_time = time.time()

    print(f"\nSettings: sims={num_simulations}, batch={batch_size}, "
          f"train_every={train_every}, train_steps={train_steps}")
    print("-" * 60)

    try:
        while True:
            elapsed = time.time() - start_time

            # Self-play
            game_start = time.time()
            samples, reward = self_play_game(game, model, num_simulations)
            game_time = time.time() - game_start

            replay_buffer.extend(samples)
            total_games += 1

            winner = "Black" if reward > 0 else ("White" if reward < 0 else "Draw")
            print(f"[{elapsed:6.1f}s] Game {total_games}: {len(samples)} moves, "
                  f"winner={winner}, buffer={len(replay_buffer)}, "
                  f"time={game_time:.1f}s")

            # Training
            if total_games % train_every == 0 and len(replay_buffer) >= batch_size:
                print(f"\n{'='*40} Training {'='*40}")

                for step in range(train_steps):
                    batch = random.sample(list(replay_buffer), batch_size)
                    pi_loss, v_loss = train_step(model, optimizer, batch, device)
                    total_steps += 1

                    if step % 5 == 0:
                        print(f"  Step {step+1:2d}/{train_steps}: "
                              f"policy_loss={pi_loss:.4f}, value_loss={v_loss:.4f}")

                # Save checkpoint
                path = f"models/game_{total_games:04d}.pt"
                torch.save(model.state_dict(), path)
                print(f"  Saved: {path}")
                print("=" * 80 + "\n")

                model.eval()

            # Stop after 10 minutes
            if elapsed > 600:
                print(f"\n10 minutes elapsed. Stopping training.")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Games played: {total_games}")
    print(f"Training steps: {total_steps}")
    print(f"Final buffer size: {len(replay_buffer)}")
    print(f"Samples/game: {len(replay_buffer)/max(1,total_games):.1f}")

    # Save final model
    final_path = "models/final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")


if __name__ == "__main__":
    main()
