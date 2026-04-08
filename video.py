"""
CartPole PPO 视频录制脚本
"""

import argparse
import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
import imageio
from datetime import datetime

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_latest_model():
    models = [f for f in os.listdir(SAVE_DIR)
               if f.startswith("ppo_cartpole_") and f.endswith(".zip")]
    models.sort()
    return os.path.join(SAVE_DIR, models[-1]) if models else None


def record_episode(model_path, n_max=100, device="cuda"):
    model = PPO.load(model_path, device=device)
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    frames = []
    meta = None

    print("[Record] Looking for success case...")
    for ep in range(n_max):
        obs, _ = env.reset()
        done = truncated = False
        ep_reward = 0

        while not (done or truncated):
            frame = env.render()
            frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

        if ep_reward >= 500:
            meta = {"ep": ep+1, "reward": ep_reward, "length": len(frames)}
            print(f"  [SUCCESS] Ep {ep+1}: reward={ep_reward:.1f}, {len(frames)} steps")
            break

    env.close()

    if frames:
        if len(frames) > 1500:
            frames = frames[::2]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SAVE_DIR, f"video_success_{ts}.mp4")
        print(f"  Encoding {len(frames)} frames -> {path}")
        imageio.mimwrite(path, frames, fps=30, codec="libx264", quality=8)
        print(f"  [Saved] {path}")
        return meta
    else:
        print("  [WARN] No success case found")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.checkpoint:
        model_path = args.checkpoint
    else:
        model_path = find_latest_model()

    if not model_path:
        print("[ERROR] No model found!")
        return

    print(f"[Model] {model_path}")
    record_episode(model_path, device=args.device)


if __name__ == "__main__":
    main()
