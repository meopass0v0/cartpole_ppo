"""
CartPole PPO 评估脚本
"""

import argparse
import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_latest_model():
    models = [f for f in os.listdir(SAVE_DIR)
               if f.startswith("ppo_cartpole_") and f.endswith(".zip")]
    models.sort()
    return os.path.join(SAVE_DIR, models[-1]) if models else None


def evaluate(model_path, n_episodes=20, device="cuda"):
    model = PPO.load(model_path, device=device)
    env = gym.make("CartPole-v1", render_mode=None)

    rewards, lengths = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = truncated = False
        steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)

        status = "OK" if steps >= 500 else "FAIL"
        print(f"  Ep {ep+1:2d}: reward={total_reward:7.2f}, length={steps:4d} [{status}]")

    env.close()

    print(f"\n[SUMMARY] Avg Reward: {np.mean(rewards):7.2f} | "
          f"Avg Length: {np.mean(lengths):6.1f} | "
          f"Max Reward: {max(rewards):7.2f}")
    return np.mean(rewards), np.mean(lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.checkpoint:
        model_path = args.checkpoint
    else:
        model_path = find_latest_model()

    if not model_path:
        print("[ERROR] No model found!")
        return

    print(f"[Model] {model_path} | device={args.device}")
    print(f"[Eval] {args.episodes} episodes...\n")

    evaluate(model_path, n_episodes=args.episodes, device=args.device)


if __name__ == "__main__":
    main()
