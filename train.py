"""
CartPole PPO - 手写版
统一训练脚本：包含手写 PPO + Metrics + 视频录制
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from datetime import datetime
import os
import imageio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────
# Actor-Critic 网络
# ─────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.net(x)
        return self.actor(features), self.critic(features)

    def get_action(self, obs):
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1), action_logits

    def get_action_and_logprob(self, obs, action):
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        return dist.log_prob(action), dist.entropy(), value.squeeze(-1), action_logits


# ─────────────────────────────────────────────
# Rollout 收集
# ─────────────────────────────────────────────
def collect_rollout(env, model, n_steps):
    obs_buf, act_buf, rew_buf = [], [], []
    done_buf, logp_buf, val_buf = [], [], []
    ent_buf = []

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

    for _ in range(n_steps):
        with torch.no_grad():
            action, log_prob, entropy, value, _ = model.get_action(obs)

        obs_np, reward, done, truncated, _ = env.step(action.item())

        if done or truncated:
            obs_new, _ = env.reset()
            obs_new = torch.tensor(obs_new, dtype=torch.float32, device=DEVICE)
        else:
            obs_new = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)

        obs_buf.append(obs)
        act_buf.append(action)
        rew_buf.append(reward)
        done_buf.append(done or truncated)
        logp_buf.append(log_prob)
        val_buf.append(value)
        ent_buf.append(entropy)

        obs = obs_new

    return (torch.stack(obs_buf), torch.stack(act_buf),
            torch.tensor(rew_buf, device=DEVICE),
            torch.tensor(done_buf, device=DEVICE),
            torch.stack(logp_buf), torch.stack(val_buf),
            torch.stack(ent_buf))


# ─────────────────────────────────────────────
# GAE
# ─────────────────────────────────────────────
def compute_returns_and_advantages(val_buf, rew_buf, done_buf, gamma, lam):
    n = len(rew_buf)
    advantages = torch.zeros(n, device=DEVICE)

    next_value = val_buf[-1].item()
    next_gae = 0.0

    for t in reversed(range(n)):
        mask = 1.0 - done_buf[t].float()
        delta = rew_buf[t] + gamma * next_value * mask - val_buf[t]
        next_gae = delta + gamma * lam * mask * next_gae
        advantages[t] = next_gae
        next_value = val_buf[t].item()

    raw_advantages = advantages.clone()
    advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)
    returns = raw_advantages + val_buf
    return returns, advantages


# ─────────────────────────────────────────────
# PPO Loss
# ─────────────────────────────────────────────
def ppo_loss(model, obs_b, act_b, old_logp_b, returns_b, advantages_b,
             logp_new_b, ent_b, action_logits_b, old_logits_b,
             clip_range, value_coef, ent_coef):
    # Policy loss
    ratio = torch.exp(logp_new_b - old_logp_b)
    surr1 = ratio * advantages_b
    ratio_clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    surr2 = ratio_clipped * advantages_b
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    values_new = model.critic(model.net(obs_b)).squeeze(-1)
    value_loss = nn.functional.mse_loss(values_new, returns_b)

    # Entropy loss
    entropy_loss = -ent_b.mean()

    # KL divergence (approximate, from action logits)
    kl = (old_logits_b - action_logits_b).mean(dim=-1)
    kl_loss = kl.mean()

    return (policy_loss + value_coef * value_loss + ent_coef * entropy_loss,
            policy_loss, value_loss, entropy_loss, kl_loss)


# ─────────────────────────────────────────────
# Minibatch 更新
# ─────────────────────────────────────────────
metrics_log = {"update": [], "policy_loss": [], "value_loss": [],
               "entropy": [], "kl": [], "eval_reward": [], "eval_length": []}


def ppo_update(model, optimizer, obs_b, act_b, old_logp_b, returns_b, advantages_b,
               clip_range, value_coef, ent_coef, batch_size, n_epochs):
    n = obs_b.shape[0]

    # 缓存 old logits 用于 KL 计算
    with torch.no_grad():
        _, _, _, old_logits = model.get_action_and_logprob(obs_b, act_b)
        old_logits_detach = old_logits.detach()

    for _ in range(n_epochs):
        idx = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch_size):
            mb = idx[start:start + batch_size]
            logp_new, ent, val_new, logits_new = model.get_action_and_logprob(obs_b[mb], act_b[mb])
            loss, p_loss, v_loss, ent_loss, kl_loss = ppo_loss(
                model, obs_b[mb], act_b[mb], old_logp_b[mb],
                returns_b[mb], advantages_b[mb],
                logp_new, ent, logits_new, old_logits_detach[mb],
                clip_range, value_coef, ent_coef)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    return p_loss.item(), v_loss.item(), ent.item(), kl_loss.item()


# ─────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────
def evaluate(env, model, n_episodes=20):
    model.eval()
    rewards, lengths = [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        total_r = 0
        done = truncated = False

        while not (done or truncated):
            with torch.no_grad():
                action, _, _, _, _ = model.get_action(obs_t)
            obs, reward, done, truncated, _ = env.step(action.item())
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            total_r += reward

        rewards.append(total_r)
        lengths.append(len(rewards))

    model.train()
    return np.mean(rewards), np.mean(lengths)


# ─────────────────────────────────────────────
# 视频录制
# ─────────────────────────────────────────────
def record_success_video(env, model, max_attempts=100):
    """录制一个成功案例（500 steps）"""
    frames = []
    meta = None

    for ep in range(max_attempts):
        obs, _ = env.reset()
        done = truncated = False
        ep_reward = 0

        while not (done or truncated):
            frame = env.render()
            frames.append(frame)
            with torch.no_grad():
                action, _, _, _, _ = model.get_action(obs)
            obs, reward, done, truncated, _ = env.step(action.item())
            ep_reward += reward

        if ep_reward >= 500:
            meta = {"ep": ep+1, "reward": ep_reward, "length": len(frames)}
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
    return None


# ─────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────
def plot_metrics(metrics_log, save_dir):
    updates = metrics_log["update"]
    n = len(updates)
    if n == 0:
        return

    rows, cols = 3, 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 9))
    fig.suptitle("PPO Training Metrics - CartPole", fontsize=14, fontweight="bold")

    def subplot(ax, y, title, ylabel, color, hline=None):
        ax.plot(updates, y, color=color, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Update")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if hline:
            ax.axhline(y=hline, color="green", linestyle="--", alpha=0.7)

    subplot(axes[0, 0], metrics_log["eval_reward"], "Avg Episode Reward", "Reward", "blue", 500)
    subplot(axes[0, 1], metrics_log["eval_length"], "Avg Episode Length", "Length", "orange", 500)
    subplot(axes[1, 0], metrics_log["policy_loss"], "Policy Loss", "Loss", "red")
    subplot(axes[1, 1], metrics_log["value_loss"], "Value Loss", "Loss", "purple")
    subplot(axes[2, 0], metrics_log["entropy"], "Entropy", "Entropy", "green")
    subplot(axes[2, 1], metrics_log["kl"], "KL Divergence", "KL", "brown")

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"metrics_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] {path}")
    plt.close()


# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────
def train(env_id, total_steps=100_000, n_steps=2048, batch_size=64,
          n_epochs=10, lr=3e-4, gamma=0.99, lam=0.95,
          clip_range=0.2, value_coef=0.5, ent_coef=0.01,
          eval_every=10, log_every=5):
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim, hidden=128).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("=" * 60)
    print("PPO - CartPole (手写版)")
    print(f"  env={env_id} | obs={obs_dim} | act={act_dim}")
    print(f"  n_steps={n_steps} | batch={batch_size} | epochs={n_epochs}")
    print(f"  lr={lr} | gamma={gamma} | lam={lam} | clip={clip_range}")
    print(f"  device={DEVICE}")
    print("=" * 60)

    total_updates = total_steps // n_steps
    global metrics_log
    metrics_log = {"update": [], "policy_loss": [], "value_loss": [],
                   "entropy": [], "kl": [], "eval_reward": [], "eval_length": []}

    for it in range(1, total_updates + 1):
        # ── Rollout ──
        obs_b, act_b, rew_b, done_b, logp_b, val_b, ent_b = collect_rollout(env, model, n_steps)

        # ── GAE ──
        returns_b, advantages_b = compute_returns_and_advantages(val_b, rew_b, done_b, gamma, lam)

        # ── Train Metrics ──
        ep_rewards, ep_lengths = [], []
        ep_r, ep_l = 0.0, 0
        for i in range(n_steps):
            ep_r += rew_b[i].item()
            ep_l += 1
            if done_b[i]:
                ep_rewards.append(ep_r)
                ep_lengths.append(ep_l)
                ep_r, ep_l = 0.0, 0

        avg_train_r = np.mean(ep_rewards) if ep_rewards else 0.0
        avg_train_l = np.mean(ep_lengths) if ep_lengths else 0.0

        # ── Update ──
        p_loss, v_loss, entropy, kl = ppo_update(
            model, optimizer, obs_b, act_b, logp_b, returns_b, advantages_b,
            clip_range, value_coef, ent_coef, batch_size, n_epochs)

        # ── Log ──
        if it % log_every == 0:
            print(f"[U{it:3d}/{total_updates}] "
                  f"train_R={avg_train_r:7.2f} | train_L={avg_train_l:6.1f} | "
                  f"p_loss={p_loss:.4f} | v_loss={v_loss:.4f} | "
                  f"ent={entropy:.4f} | kl={kl:.4f}")

        # ── Eval ──
        if it % eval_every == 0 or it == total_updates:
            eval_r, eval_l = evaluate(env, model, n_episodes=20)
            metrics_log["update"].append(it)
            metrics_log["policy_loss"].append(p_loss)
            metrics_log["value_loss"].append(v_loss)
            metrics_log["entropy"].append(entropy)
            metrics_log["kl"].append(kl)
            metrics_log["eval_reward"].append(eval_r)
            metrics_log["eval_length"].append(eval_l)
            print(f"  >> [Eval] R={eval_r:7.2f} | L={eval_l:5.1f}")

    return model, metrics_log


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    args = parser.parse_args()

    print("=" * 60)
    print("CartPole PPO Training")
    print("=" * 60)

    model, metrics_log = train(
        env_id="CartPole-v1",
        total_steps=args.steps,
        n_steps=2048, batch_size=64, n_epochs=10,
        lr=3e-4, gamma=0.99, lam=0.95,
        clip_range=0.2, value_coef=0.5, ent_coef=0.01,
        eval_every=10, log_every=5
    )

    # ── 最终评估 ──
    env = gym.make("CartPole-v1")
    final_r, final_l = evaluate(env, model, n_episodes=50)
    print("\n" + "=" * 60)
    print("Final Evaluation (50 episodes)")
    print(f"  Avg Reward: {final_r:.2f} | Avg Length: {final_l:.1f}")
    print("=" * 60)

    # ── 可视化 ──
    plot_metrics(metrics_log, SAVE_DIR)

    # ── 保存模型 ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(SAVE_DIR, f"ppo_cartpole_{ts}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n[Saved] {model_path}")

    # ── 录制视频 ──
    print("\n[Video] Recording success case...")
    env_video = gym.make("CartPole-v1", render_mode="rgb_array")
    meta = record_success_video(env_video, model)
    if meta:
        print(f"  Success case: {meta}")
    else:
        print("  [WARN] No 500-step success case found")

    print("\n[DONE] All outputs saved to:", SAVE_DIR)
