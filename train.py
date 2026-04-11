"""
CartPole PPO - 手写版 + AsyncVectorEnv (8 envs)
包含: Value Clipping + KL Early Stopping + LR/Clip Decay
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
from gymnasium.vector import AsyncVectorEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────
# Env Factory (给 AsyncVectorEnv 用)
# ─────────────────────────────────────────────
def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env
    return _thunk


# ─────────────────────────────────────────────
# Actor-Critic 网络
# ─────────────────────────────────────────────
class ActorCriticSep(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128, critic_hidden=256):
        super().__init__()
        # Actor: smaller backbone, fast adaptation
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden, act_dim)

        # Critic: larger backbone, stable value estimation
        self.critic_net = nn.Sequential(
            nn.Linear(obs_dim, critic_hidden),
            nn.Tanh(),
            nn.Linear(critic_hidden, critic_hidden),
            nn.Tanh(),
            nn.Linear(critic_hidden, critic_hidden // 2),
            nn.Tanh(),
        )
        self.critic_head = nn.Linear(critic_hidden // 2, 1)

    def forward(self, x):
        actor_features = self.actor_net(x)
        logits = self.actor_head(actor_features)
        critic_features = self.critic_net(x)
        value = self.critic_head(critic_features).squeeze(-1)
        return logits, value

    def get_action(self, obs, deterministic=False):
        logits, val = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), val

    def get_action_and_logprob(self, obs, action):
        logits, val = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), val


# ─────────────────────────────────────────────
# Rollout 收集 (AsyncVectorEnv) — 无 ent_buf
# ─────────────────────────────────────────────
def collect_rollout_vec(vec_env, model, obs_batch, n_steps_per_env):
    """
    收集 n_steps_per_env × n_envs 个 step。
    Buffer shape: [T, N, ...]，T=n_steps_per_env, N=n_envs

    obs_batch: 传入的初始 obs [N, obs_dim]（上一次 rollout 结束的状态）
    返回: buffers + 最后一步的 obs_batch（供下一轮继续）

    注意: 不存 ent_buf，entropy 在 update 时重新算
    """
    n_envs = vec_env.num_envs

    T, N = n_steps_per_env, n_envs
    obs_dim = obs_batch.shape[1]

    obs_buf  = torch.zeros(T, N, obs_dim)           # CPU
    act_buf  = torch.zeros(T, N, dtype=torch.long)    # CPU
    rew_buf  = torch.zeros(T, N)                      # CPU
    done_buf = torch.zeros(T, N, dtype=torch.bool)    # CPU
    logp_buf = torch.zeros(T, N)                      # CPU
    val_buf  = torch.zeros(T, N)                      # CPU

    obs_current = obs_batch  # GPU tensor，外部传入

    for t in range(T):
        # GPU forward
        with torch.no_grad():
            action, log_prob, _, value = model.get_action(obs_current)

        # Env step（CPU）
        next_obs_np, reward_np, terminated_np, truncated_np, infos = vec_env.step(action.cpu().numpy())
        done_np = np.logical_or(terminated_np, truncated_np)

        # 转 CPU tensor（单次 GPU→CPU copy）
        obs_cpu = obs_current.cpu()
        act_cpu = action.cpu()
        val_cpu = value.to("cpu").clone()
        logp_cpu = log_prob.cpu()
        reward_t = torch.as_tensor(reward_np, dtype=torch.float32)
        done_t   = torch.as_tensor(done_np, dtype=torch.bool)

        obs_buf[t]   = obs_cpu
        act_buf[t]   = act_cpu
        rew_buf[t]   = reward_t
        done_buf[t]  = done_t
        logp_buf[t]  = logp_cpu
        val_buf[t]   = val_cpu

        # 下一 step：numpy → CPU → GPU
        obs_current = torch.as_tensor(next_obs_np, dtype=torch.float32, device=DEVICE)

    # obs_current 此时是 GPU tensor，供下一轮 rollout 用
    return obs_buf, act_buf, rew_buf, done_buf, logp_buf, val_buf, obs_current


# ─────────────────────────────────────────────
# GAE (Vectorized — 整块 [T, N] 一起算 + 全局 normalize)
# ─────────────────────────────────────────────
def compute_returns_and_advantages_vec(val_buf, rew_buf, done_buf, last_values, gamma, lam):
    """
    val_buf:     [T, N]
    rew_buf:     [T, N]
    done_buf:    [T, N]
    last_values: [N] — rollout 结束后最后一个状态的 value
    return:      returns_flat, advantages_flat 均为 [T*N]
    """
    T, N = rew_buf.shape
    advantages = torch.zeros(T, N)  # CPU
    next_gae   = torch.zeros(N)  # CPU

    for t in reversed(range(T)):
        next_value = last_values if t == T - 1 else val_buf[t + 1]
        mask = 1.0 - done_buf[t].float()
        delta = rew_buf[t] + gamma * next_value * mask - val_buf[t]
        next_gae = delta + gamma * lam * mask * next_gae
        advantages[t] = next_gae

    returns = advantages + val_buf

    # 全局 normalize（不是 per-env）
    adv_flat = advantages.reshape(T * N)
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    ret_flat = returns.reshape(T * N)

    # GAE 结果在 update 前搬到 GPU
    return ret_flat.to(DEVICE), adv_flat.to(DEVICE)


# ─────────────────────────────────────────────
# Flatten buffer helper — 无 ent_buf
# ─────────────────────────────────────────────
def flatten_buffer(obs_buf, act_buf, logp_buf, val_buf):
    """[T, N, obs_dim] → [T*N, obs_dim]，正确 reshape"""
    T, N = obs_buf.shape[:2]
    obs_dim = obs_buf.shape[2]
    return (
        obs_buf.reshape(T * N, obs_dim),
        act_buf.reshape(T * N),
        logp_buf.reshape(T * N),
        val_buf.reshape(T * N),
    )


# ─────────────────────────────────────────────
# PPO Loss (with Value Clipping) — values_new_b 从外部传入
# ─────────────────────────────────────────────
def ppo_loss(old_logp_b, old_v_b, returns_b, advantages_b,
             logp_new_b, values_new_b, ent_new_b, clip_range, value_coef, ent_coef):
    # Policy loss (PPO clipped objective)
    ratio = torch.exp(logp_new_b - old_logp_b)
    surr1 = ratio * advantages_b
    ratio_clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    surr2 = ratio_clipped * advantages_b
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss (with value clipping) — values_new_b 直接从 get_action_and_logprob 传来
    v_clipped = old_v_b + torch.clamp(values_new_b - old_v_b, -clip_range, clip_range)
    value_loss = torch.max(
        (values_new_b - returns_b) ** 2,
        (v_clipped - returns_b) ** 2
    ).mean()

    # Entropy loss — 用新的 entropy
    entropy_loss = -ent_new_b.mean()

    return (policy_loss + value_coef * value_loss + ent_coef * entropy_loss,
            policy_loss, value_loss, entropy_loss)


# ─────────────────────────────────────────────
# Minibatch 更新 (with KL Early Stopping + 修正 KL)
# ─────────────────────────────────────────────
def ppo_update(model, optimizer, obs_b, act_b, old_logp_b, old_v_b,
               returns_b, advantages_b, clip_range, value_coef, ent_coef,
               batch_size, n_epochs, target_kl=None):
    n = obs_b.shape[0]

    for epoch in range(n_epochs):
        idx = torch.randperm(n, device=DEVICE)
        kl_sum = 0.0
        num_batches = 0

        for start in range(0, n, batch_size):
            mb = idx[start:start + batch_size]
            logp_new, ent_new, values_new = model.get_action_and_logprob(obs_b[mb], act_b[mb])

            loss, p_loss, v_loss, ent_loss = ppo_loss(
                old_logp_b[mb], old_v_b[mb],
                returns_b[mb], advantages_b[mb],
                logp_new, values_new, ent_new,
                clip_range, value_coef, ent_coef)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # 稳定的 KL 估计: ((ratio-1) - log_ratio).mean()
            log_ratio = logp_new - old_logp_b[mb]
            ratio_kl = torch.exp(log_ratio)
            approx_kl = ((ratio_kl - 1) - log_ratio).mean()
            kl_sum += approx_kl.item()
            num_batches += 1

        avg_kl = kl_sum / num_batches if num_batches > 0 else 0.0
        if target_kl is not None and avg_kl > target_kl:
            return p_loss.item(), v_loss.item(), ent_loss.item(), avg_kl, epoch + 1

    return p_loss.item(), v_loss.item(), ent_loss.item(), avg_kl, n_epochs


# ─────────────────────────────────────────────
# 线性衰减
# ─────────────────────────────────────────────
def linear_decay(initial, final, progress):
    return initial - (initial - final) * progress


# ─────────────────────────────────────────────
# 评估（用单个 env）
# ─────────────────────────────────────────────
def evaluate(env, model, n_episodes=20):
    model.eval()
    rewards, lengths, successes = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        total_r = 0
        done = truncated = False
        steps = 0

        while not (done or truncated):
            with torch.no_grad():
                action, _, _, _ = model.get_action(obs_t, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action.item())
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
            total_r += reward
            steps += 1

        rewards.append(total_r)
        lengths.append(steps)
        successes.append(steps >= 500)

    model.train()
    return np.mean(rewards), np.mean(lengths), np.mean(successes) * 100


# ─────────────────────────────────────────────
# 视频录制
# ─────────────────────────────────────────────
def record_success_video(env, model, max_attempts=100):
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
                action, _, _, _ = model.get_action(
                    torch.as_tensor(obs, dtype=torch.float32, device=DEVICE), deterministic=True)
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
    fig.suptitle("PPO Training Metrics - CartPole (AsyncVectorEnv x8)", fontsize=14, fontweight="bold")

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
def train(env_id, total_steps=100_000, n_steps_per_env=512, n_envs=8,
          batch_size=512, n_epochs=3,
          lr=1e-4, gamma=0.99, lam=0.85,
          clip_range_init=0.15, clip_range_final=0.08,
          value_coef=0.5, ent_coef=0.01,
          target_kl=0.015,
          eval_every=10, log_every=5,
          load_path=None):
    """
    n_steps_per_env: 每个 env 收集的步数 (T=512)
    n_envs: 并行 env 数量 (N=8)
    总步数 per rollout = n_steps_per_env * n_envs = 4096
    """
    # ── AsyncVectorEnv（真并行）──
    env_fns = [make_env(env_id, 42, i) for i in range(n_envs)]
    vec_env = AsyncVectorEnv(env_fns)

    n_total_steps = n_steps_per_env * n_envs

    obs_dim = vec_env.single_observation_space.shape[0]
    act_dim = vec_env.single_action_space.n

    model = ActorCriticSep(obs_dim, act_dim, hidden=256, critic_hidden=512).to(DEVICE)

    # Load checkpoint if provided
    if load_path:
        ckpt = torch.load(load_path, map_location=DEVICE)
        model.load_state_dict(ckpt)
        print("[INFO] Loaded checkpoint from " + load_path)
    # 分离 optimizer: actor 保守，critic 积极
    actor_lr = 1e-4
    critic_lr = 2e-4
    optimizer = optim.Adam([
        {"params": model.actor_net.parameters(), "lr": actor_lr},
        {"params": model.actor_head.parameters(), "lr": actor_lr},
        {"params": model.critic_net.parameters(), "lr": critic_lr},
        {"params": model.critic_head.parameters(), "lr": critic_lr},
    ])

    print("=" * 60)
    print("PPO - CartPole (手写版 + AsyncVectorEnv x8)")
    print(f"  env={env_id} | obs={obs_dim} | act={act_dim}")
    print(f"  n_envs={n_envs} | n_steps_per_env={n_steps_per_env} | total={n_total_steps}")
    print(f"  batch={batch_size} | epochs={n_epochs}")
    print(f"  lr={lr} | gamma={gamma} | lam={lam}")
    print(f"  clip: {clip_range_init} -> {clip_range_final}")
    print(f"  value_clip + kl_early_stop + lr/clip_decay")
    print(f"  device={DEVICE}")
    print("=" * 60)

    total_updates = total_steps // n_total_steps
    metrics_log = {"update": [], "policy_loss": [], "value_loss": [],
                   "entropy": [], "kl": [], "lr": [], "clip": [],
                   "eval_reward": [], "eval_length": [], "eval_sr": []}

    # ── 只在开头 reset 一次 ──
    obs_np, _ = vec_env.reset(seed=42)
    obs_batch = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE)  # GPU

    # ── 跨 rollout 的 episode 状态 ──
    running_ep_r = np.zeros(n_envs, dtype=np.float32)
    running_ep_l = np.zeros(n_envs, dtype=np.int32)

    for it in range(1, total_updates + 1):
        progress = (it - 1) / max(total_updates - 1, 1)

        # ── Decay ──
        current_lr   = linear_decay(lr, lr * 0.1, progress)
        current_clip = linear_decay(clip_range_init, clip_range_final, progress)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # ── Rollout ──
        obs_b, act_b, rew_b, done_b, logp_b, val_b, obs_batch = \
            collect_rollout_vec(vec_env, model, obs_batch, n_steps_per_env)

        # ── 计算 last_values ──
        with torch.no_grad():
            _, last_values = model.forward(obs_batch)
            last_values = last_values.squeeze(-1).to("cpu").clone()  # GAE 在 CPU 算

        # ── GAE（整块 [T,N] + 全局 normalize）──
        returns_b, advantages_b = compute_returns_and_advantages_vec(
            val_b, rew_b, done_b, last_values, gamma, lam)

        # ── Flatten buffers → 一次性搬到 GPU ──
        obs_flat, act_flat, logp_flat, val_flat = flatten_buffer(obs_b, act_b, logp_b, val_b)
        obs_flat = obs_flat.to(DEVICE)
        act_flat = act_flat.to(DEVICE)
        logp_flat = logp_flat.to(DEVICE)
        val_flat = val_flat.to(DEVICE)

        # ── Train Metrics（per-env 追踪，running 跨 rollout）──
        finished_rewards, finished_lengths = [], []

        for t in range(n_steps_per_env):
            rew_t = rew_b[t].cpu().numpy()
            done_t = done_b[t].cpu().numpy()
            running_ep_r += rew_t
            running_ep_l += 1
            for e in range(n_envs):
                if done_t[e]:
                    finished_rewards.append(running_ep_r[e])
                    finished_lengths.append(running_ep_l[e])
                    running_ep_r[e] = 0.0
                    running_ep_l[e] = 0

        avg_train_r = np.mean(finished_rewards) if finished_rewards else 0.0
        avg_train_l = np.mean(finished_lengths) if finished_lengths else 0.0

        # ── Update ──
        p_loss, v_loss, entropy, kl, _ = ppo_update(
            model, optimizer, obs_flat, act_flat, logp_flat, val_flat,
            returns_b, advantages_b,
            current_clip, value_coef, ent_coef,
            batch_size, n_epochs, target_kl)

        # ── Log ──
        if it % log_every == 0:
            print(f"[U{it:3d}/{total_updates}] "
                  f"train_R={avg_train_r:7.2f} | train_L={avg_train_l:6.1f} | "
                  f"p_loss={p_loss:.4f} | v_loss={v_loss:.4f} | "
                  f"ent={entropy:.4f} | kl={kl:.4f} | "
                  f"lr={current_lr:.6f} | clip={current_clip:.4f}")

        # ── Eval ──
        if it % eval_every == 0 or it == total_updates:
            eval_env = gym.make(env_id)
            eval_r, eval_l, eval_sr = evaluate(eval_env, model, n_episodes=20)
            eval_env.close()
            metrics_log["update"].append(it)
            metrics_log["policy_loss"].append(p_loss)
            metrics_log["value_loss"].append(v_loss)
            metrics_log["entropy"].append(entropy)
            metrics_log["kl"].append(kl)
            metrics_log["lr"].append(current_lr)
            metrics_log["clip"].append(current_clip)
            metrics_log["eval_reward"].append(eval_r)
            metrics_log["eval_length"].append(eval_l)
            metrics_log["eval_sr"].append(eval_sr)
            print(f"  >> [Eval] R={eval_r:7.2f} | L={eval_l:5.1f} | SR={eval_sr:5.1f}%")

    vec_env.close()
    return model, metrics_log


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000000)
    parser.add_argument("--load_latest", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("CartPole PPO Training (AsyncVectorEnv x8)")
    print("=" * 60)

    # Load latest checkpoint if requested
    if args.load_latest:
        import glob as glob_module
        pt_files = sorted(glob_module.glob(os.path.join(SAVE_DIR, 'ppo_cartpole_*.pt')))
        if pt_files:
            latest = pt_files[-1]
            print(f'[INFO] Loading latest checkpoint: {latest}')
            load_path = latest
        else:
            print('[WARN] No checkpoint found, starting from scratch')
            load_path = None
    else:
        load_path = None

    model, metrics_log = train(
        env_id="CartPole-v1",
        total_steps=args.steps,
        load_path=load_path if "load_path" in dir() else None,
        n_steps_per_env=512, n_envs=16,
        batch_size=512, n_epochs=2,
        lr=2e-4, gamma=0.99, lam=0.9,
        clip_range_init=0.2, clip_range_final=0.08,
        value_coef=0.5, ent_coef=0.02,
        target_kl=0.015,
        eval_every=10, log_every=5
    )

    # ── 最终评估 ──
    env = gym.make("CartPole-v1")
    final_r, final_l, final_sr = evaluate(env, model, n_episodes=50)
    print("\n" + "=" * 60)
    print("Final Evaluation (50 episodes)")
    print(f"  Avg Reward: {final_r:.2f} | Avg Length: {final_l:.1f} | SR: {final_sr:.1f}%")
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
