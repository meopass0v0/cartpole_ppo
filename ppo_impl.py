"""
PPO Core Implementation - CartPole 版本
基于 acrobot_ppo 的手写版 PPO
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    def get_action(self, obs, deterministic=False):
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)

    def get_action_and_logprob(self, obs, action):
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        return dist.log_prob(action), dist.entropy(), value.squeeze(-1)


def collect_rollout(env, model, n_steps):
    """收集 n_steps 步数据"""
    obs_buffer, act_buffer, rew_buffer = [], [], []
    done_buffer, logp_buffer, val_buffer = [], [], []
    ent_buffer = []

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

    for _ in range(n_steps):
        with torch.no_grad():
            action, log_prob, entropy, value = model.get_action(obs)

        obs_np, reward, done, truncated, _ = env.step(action.item())

        if done or truncated:
            obs_new, _ = env.reset()
            obs_new = torch.tensor(obs_new, dtype=torch.float32, device=DEVICE)
        else:
            obs_new = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)

        obs_buffer.append(obs)
        act_buffer.append(action)
        rew_buffer.append(reward)
        done_buffer.append(done or truncated)
        logp_buffer.append(log_prob)
        val_buffer.append(value)
        ent_buffer.append(entropy)

        obs = obs_new

    return (torch.stack(obs_buffer), torch.stack(act_buffer),
            torch.tensor(rew_buffer, device=DEVICE),
            torch.tensor(done_buffer, device=DEVICE),
            torch.stack(logp_buffer), torch.stack(val_buffer),
            torch.stack(ent_buffer))


def compute_returns_and_advantages(val_buf, rew_buf, done_buf, gamma, lam):
    """GAE (Generalized Advantage Estimation)"""
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


def ppo_loss(model, obs_b, act_b, old_logp_b, returns_b, advantages_b,
             logp_new_b, ent_b, clip_range, value_coef, ent_coef):
    ratio = torch.exp(logp_new_b - old_logp_b)
    surr1 = ratio * advantages_b
    ratio_clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    surr2 = ratio_clipped * advantages_b
    policy_loss = -torch.min(surr1, surr2).mean()

    values_new = model.critic(model.net(obs_b)).squeeze(-1)
    value_loss = nn.functional.mse_loss(values_new, returns_b)

    entropy_loss = -ent_b.mean()

    return (policy_loss + value_coef * value_loss + ent_coef * entropy_loss,
            policy_loss, value_loss, entropy_loss)


def ppo_update(model, optimizer, obs_b, act_b, old_logp_b, returns_b, advantages_b,
               clip_range, value_coef, ent_coef, batch_size, n_epochs):
    n = obs_b.shape[0]
    for _ in range(n_epochs):
        idx = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch_size):
            mb = idx[start:start + batch_size]
            logp_new, ent, _ = model.get_action_and_logprob(obs_b[mb], act_b[mb])
            loss, _, _, _ = ppo_loss(model, obs_b[mb], act_b[mb], old_logp_b[mb],
                                      returns_b[mb], advantages_b[mb],
                                      logp_new, ent, clip_range, value_coef, ent_coef)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


def evaluate(env, model, n_episodes=20):
    """评估模型：返回平均 reward 和 episode length"""
    model.eval()
    rewards, lengths = [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        total_r = 0
        done = truncated = False

        while not (done or truncated):
            with torch.no_grad():
                action, _, _, _ = model.get_action(obs_t)
            obs, reward, done, truncated, _ = env.step(action.item())
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            total_r += reward

        rewards.append(total_r)
        lengths.append(len(rewards))  # steps = episode length

    return np.mean(rewards), np.mean(lengths)


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
    print("PPO - CartPole")
    print(f"  env={env_id} | obs={obs_dim} | act={act_dim}")
    print(f"  n_steps={n_steps} | batch={batch_size} | epochs={n_epochs}")
    print(f"  lr={lr} | gamma={gamma} | lam={lam} | clip={clip_range}")
    print(f"  device={DEVICE}")
    print("=" * 60)

    total_updates = total_steps // n_steps
    history = {"update": [], "train_reward": [], "train_length": [],
               "eval_reward": [], "eval_length": []}

    for it in range(1, total_updates + 1):
        obs_b, act_b, rew_b, done_b, logp_b, val_b, ent_b = collect_rollout(env, model, n_steps)
        returns_b, advantages_b = compute_returns_and_advantages(val_b, rew_b, done_b, gamma, lam)

        # 统计 train metrics
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

        ppo_update(model, optimizer, obs_b, act_b, logp_b, returns_b, advantages_b,
                   clip_range, value_coef, ent_coef, batch_size, n_epochs)

        if it % log_every == 0:
            print(f"[U{it:3d}/{total_updates}] "
                  f"train_R={avg_train_r:7.2f} | train_L={avg_train_l:6.1f}")

        if it % eval_every == 0 or it == total_updates:
            eval_r, eval_l = evaluate(env, model, n_episodes=20)
            history["update"].append(it)
            history["eval_reward"].append(eval_r)
            history["eval_length"].append(eval_l)
            print(f"  >> [Eval] R={eval_r:7.2f} | L={eval_l:5.1f}")

    return model, history


def plot(history, save_dir):
    updates = history["update"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("PPO Training - CartPole", fontsize=14, fontweight="bold")

    axes[0].plot(updates, history["eval_reward"], "b-", linewidth=2)
    axes[0].set_xlabel("Update"); axes[0].set_ylabel("Avg Reward")
    axes[0].set_title("Average Episode Reward (Higher = Better)")
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=500, color="green", linestyle="--", label="Max (500)")
    axes[0].legend()

    axes[1].plot(updates, history["eval_length"], "orange", linewidth=2)
    axes[1].set_xlabel("Update"); axes[1].set_ylabel("Avg Length")
    axes[1].set_title("Average Episode Length (Higher = Better)")
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=500, color="green", linestyle="--", label="Max (500)")
    axes[1].legend()

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"training_curves_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] {path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    args = parser.parse_args()

    save_dir = os.path.dirname(os.path.abspath(__file__))
    model, history = train(
        env_id="CartPole-v1",
        total_steps=args.steps,
        n_steps=2048, batch_size=64, n_epochs=10,
        lr=3e-4, gamma=0.99, lam=0.95,
        clip_range=0.2, value_coef=0.5, ent_coef=0.01,
        eval_every=10, log_every=5
    )

    env = gym.make("CartPole-v1")
    final_r, final_l = evaluate(env, model, n_episodes=50)
    print("\n" + "=" * 60)
    print("Final Evaluation (50 episodes)")
    print(f"  Avg Reward: {final_r:.2f} | Avg Length: {final_l:.1f}")
    print("=" * 60)

    plot(history, save_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), os.path.join(save_dir, f"ppo_cartpole_{ts}.pt"))
    print(f"\n[Saved] ppo_cartpole_{ts}.pt")
