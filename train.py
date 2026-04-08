"""
CartPole PPO 训练脚本
使用 stable-baselines3 PPO
"""

import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import imageio

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.total_episodes += 1
                if self.total_episodes % 20 == 0:
                    avg_r = np.mean(self.episode_rewards[-20:])
                    avg_l = np.mean(self.episode_lengths[-20:])
                    print(f"[Ep {self.total_episodes}] AvgR={avg_r:7.2f} | AvgL={avg_l:6.1f}")
        return True

    def get_metrics_df(self):
        df = pd.DataFrame({
            "episode": range(1, len(self.episode_rewards) + 1),
            "reward": self.episode_rewards,
            "length": self.episode_lengths,
        })
        df["reward_smooth"] = df["reward"].rolling(50, min_periods=1).mean()
        df["length_smooth"] = df["length"].rolling(50, min_periods=1).mean()
        return df


def sample_and_record(model, n_max=100):
    """录制一个成功案例"""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    frames = []
    success_meta = None

    print("\n[Sample] Collecting success case...")
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

        if ep_reward >= 500:  # CartPole-v1 最大步数
            success_meta = {"ep": ep+1, "reward": ep_reward, "length": len(frames)}
            print(f"  [SUCCESS] Ep {ep+1}: reward={ep_reward:.1f}, {len(frames)} steps")
            break

    env.close()
    return frames, success_meta


def save_video(frames, path, fps=30):
    if len(frames) > 1500:
        frames = frames[::2]
    print(f"  Encoding {len(frames)} frames -> {path}")
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)


def plot_learning_curves(df, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CartPole PPO Training", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(df["episode"], df["reward"], alpha=0.3, color="steelblue")
    ax.plot(df["episode"], df["reward_smooth"], color="steelblue", linewidth=2)
    ax.axhline(y=500, color="green", linestyle="--", label="Max (500)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Episode Reward (Higher = Better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(df["episode"], df["length"], alpha=0.3, color="orange")
    ax.plot(df["episode"], df["length_smooth"], color="orange", linewidth=2)
    ax.axhline(y=500, color="green", linestyle="--", label="Max (500)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length (steps)")
    ax.set_title("Episode Length (Higher = Better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"learning_curves_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("CartPole PPO Training")
    print("=" * 60)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = Monitor(env)

    print(f"[Env] CartPole-v1 | Action: {env.action_space} | Obs: {env.observation_space.shape}")
    print(f"[Train] {args.steps} timesteps | device={args.device}")

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        verbose=1,
        device=args.device
    )

    metrics_cb = MetricsCallback()

    start = datetime.now()
    model.learn(args.steps, callback=metrics_cb, progress_bar=True)
    train_time = (datetime.now() - start).total_seconds()

    print(f"\n[DONE] Training: {train_time:.1f}s")

    df = metrics_cb.get_metrics_df()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(SAVE_DIR, f"metrics_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Saved] Metrics: {csv_path}")

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  Total Episodes:     {df['episode'].iloc[-1]}")
    print(f"  Final Reward (avg): {df['reward'].tail(50).mean():.2f}")
    print(f"  Final Length (avg): {df['length'].tail(50).mean():.1f}")

    plot_learning_curves(df, SAVE_DIR)

    model_path = os.path.join(SAVE_DIR, f"ppo_cartpole_{ts}")
    model.save(model_path)
    print(f"[Saved] Model: {model_path}.zip")

    # 录制视频
    print("\n[Video] Recording success case...")
    frames, meta = sample_and_record(model)
    if frames:
        video_path = os.path.join(SAVE_DIR, f"video_success_{ts}.mp4")
        save_video(frames, video_path)
        print(f"  Success case: {meta}")

    print("\n[DONE] All outputs saved to:", SAVE_DIR)


if __name__ == "__main__":
    main()
