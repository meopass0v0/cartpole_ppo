# CartPole PPO 项目设计文档

> 本文档记录 CartPole PPO 实现的技术细节，供团队成员 review。

**最后更新：** 2026-04-11
**维护者：** 凌云（TLM）

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                             │
│  for update in range(total_updates):                        │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Rollout     │    │    GAE       │    │   PPO Update │  │
│  │  (CPU-bound) │ →  │  (CPU)       │ →  │  (GPU-bound) │  │
│  │  AsyncVecEnv │    │  returns/adv │    │  minibatch   │  │
│  │  x8 parallel │    │  flatten     │    │  + ValueClip│  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                    │            │
│    obs_batch ──────────────→ obs_batch (GPU)      │            │
└─────────────────────────────────────────────────────────────┘
```

**核心设计原则：**
- Rollout 和 Update 解耦，各自使用合适的 device
- Buffer 全程留在 CPU，避免 GPU/CPU 每步互相拷贝
- 一次性批量搬运，减少碎片化开销

---

## 二、AsyncVectorEnv 并行采样

### 2.1 Env Factory 模式

```python
def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env
    return _thunk

env_fns = [make_env(env_id, 42, i) for i in range(n_envs)]
vec_env = AsyncVectorEnv(env_fns)
```

**关键点：**
- 每个 env 有独立 seed，保证可复现性
- `AsyncVectorEnv` 使用多进程并行，不同于 `DummyVecEnv` 的顺序执行
- 8 envs 并行，样本独立性更高，advantage 估计方差更低

### 2.2 Reset 只做一次

```python
# train() 开头只 reset 一次
obs_np, _ = vec_env.reset(seed=42)
obs_batch = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE)

for it in range(1, total_updates + 1):
    # 传入 obs_batch，rollout 结束后返回新的 obs_batch
    obs_b, act_b, ..., obs_batch = collect_rollout_vec(
        vec_env, model, obs_batch, n_steps_per_env)
```

**原因：** 每轮 reset 会截断跨 rollout 的 episode，导致统计偏差。

---

## 三、Rollout Buffer 设计

### 3.1 Buffer Shape

```
Buffer shape: [T, N, ...]
- T = n_steps_per_env = 512
- N = n_envs = 8
- 每列是一个 env 的时序数据
```

### 3.2 CPU/GPU 分离策略（关键性能优化）

```python
# Buffer 分配在 CPU
obs_buf  = torch.zeros(T, N, obs_dim)    # CPU
act_buf  = torch.zeros(T, N, dtype=torch.long)  # CPU
rew_buf  = torch.zeros(T, N)              # CPU
done_buf = torch.zeros(T, N, dtype=torch.bool)  # CPU
logp_buf = torch.zeros(T, N)              # CPU
val_buf  = torch.zeros(T, N)              # CPU

# 每步 GPU forward，结果立即 .to("cpu").clone() 回 CPU
obs_current = torch.as_tensor(next_obs_np, dtype=torch.float32, device=DEVICE)  # GPU
with torch.no_grad():
    action, log_prob, _, value = model.get_action(obs_current)  # GPU forward

# ⚠️ 必须用 .to("cpu").clone()，不能用 .cpu()
val_cpu = value.to("cpu").clone()  # 真正拷贝到 CPU
# .cpu() 只改 device flag，不拷贝数据，会导致后续计算混用 GPU/CPU tensor
```

### 3.3 Rollout 结束时批量搬运

```python
# flatten 后一次性搬到 GPU
obs_flat, act_flat, logp_flat, val_flat = flatten_buffer(obs_b, act_b, logp_b, val_b)
obs_flat = obs_flat.to(DEVICE)
act_flat = act_flat.to(DEVICE)
logp_flat = logp_flat.to(DEVICE)
val_flat = val_flat.to(DEVICE)
```

---

## 四、GAE（Generalized Advantage Estimation）

### 4.1 必须使用真实的 last_values

```python
# rollout 结束后，对最后的 obs_batch 预测 value
with torch.no_grad():
    _, last_values = model.forward(obs_batch)  # [N]
    last_values = last_values.squeeze(-1).to("cpu").clone()

# GAE 倒推时，用 last_values 作为 T 时刻的 bootstrap
next_value = last_values if t == T - 1 else val_buf[t + 1]
```

**❌ 错误做法：** 用 `val_buf[-1]` 作为 bootstrap — 这把最后一个采样状态自己的 value 当成 "rollout 结束后" 的 value，是错误的。

### 4.2 全局 Normalize

```python
# ❌ 错误：每个 env 单独 normalize，抹平不同 env 的真实贡献差异
advantages_norm = (raw - mean) / std  # per-env

# ✅ 正确：全部 flatten 后全局 normalize 一次
adv_flat = advantages.reshape(T * N)
adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
```

### 4.3 整块 [T, N] 计算

```python
advantages = torch.zeros(T, N)  # 全 CPU
next_gae = torch.zeros(N)

for t in reversed(range(T)):
    next_value = last_values if t == T - 1 else val_buf[t + 1]
    mask = 1.0 - done_buf[t].float()  # CPU
    delta = rew_buf[t] + gamma * next_value * mask - val_buf[t]
    next_gae = delta + gamma * lam * mask * next_gae
    advantages[t] = next_gae
```

---

## 五、PPO 更新

### 5.1 Value Clipping（与 Policy Clip 对称）

```python
def ppo_loss(old_logp_b, old_v_b, returns_b, advantages_b,
             logp_new_b, values_new_b, ent_new_b,
             clip_range, value_coef, ent_coef):

    # Policy loss: PPO clipped objective
    ratio = torch.exp(logp_new_b - old_logp_b)
    surr1 = ratio * advantages_b
    ratio_clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    surr2 = ratio_clipped * advantages_b
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss with clipping（对称于 policy clip）
    v_clipped = old_v_b + torch.clamp(values_new_b - old_v_b, -clip_range, clip_range)
    value_loss = torch.max(
        (values_new_b - returns_b) ** 2,
        (v_clipped - returns_b) ** 2
    ).mean()

    entropy_loss = -ent_new_b.mean()
    return policy_loss + value_coef * value_loss + ent_coef * entropy_loss, ...
```

**关键：`values_new_b` 从 `get_action_and_logprob()` 传入，不在 `ppo_loss` 内重复 forward。**

### 5.2 避免重复 Forward

```python
# ppo_update 中：
logp_new, ent_new, values_new = model.get_action_and_logprob(obs_b[mb], act_b[mb])
# ✅ 一次 forward，logp/entropy/value 全拿到

# ❌ 之前错误地在 ppo_loss 里又做了：
values_new = model.critic(model.net(obs_b))  # 重复 forward
```

### 5.3 稳定的 KL 估计

```python
# ❌ 错误：非对称，可能为负
kl = (logp_new - old_logp_b).mean()

# ✅ 正确：始终 >= 0，对称稳定
log_ratio = logp_new - old_logp_b
ratio_kl = torch.exp(log_ratio)
approx_kl = ((ratio_kl - 1) - log_ratio).mean()
```

### 5.4 KL Early Stopping

```python
for epoch in range(n_epochs):
    ...
    if target_kl is not None and avg_kl > target_kl:
        return p_loss, v_loss, ent_loss, avg_kl, epoch + 1
```

### 5.5 Entropy 不存 Buffer

```python
# ✅ 不存 ent_buf，update 时重新算
logp_new, ent_new, values_new = model.get_action_and_logprob(obs_b[mb], act_b[mb])

# ❌ 之前存了 ent_buf，浪费显存 + 无意义搬运
```

---

## 六、训练 Metrics

### 6.1 跨 Rollout 的 Episode 追踪

```python
# 初始化在 for it 循环外部，跨 rollout 保持状态
running_ep_r = np.zeros(n_envs, dtype=np.float32)
running_ep_l = np.zeros(n_envs, dtype=np.int32)

for it in range(1, total_updates + 1):
    finished_rewards, finished_lengths = [], []  # 每轮只重置 finished

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
```

**原因：** episode 可能横跨两个 rollout，必须跨 rollout 追踪。

### 6.2 per-env 独立统计

```python
# ❌ 错误：flatten 后串起来算，混淆不同 env 的 episode
for i in range(n_total_steps):
    ep_r += rew_flat[i].item()

# ✅ 正确：每个 env 独立追踪
for e in range(n_envs):
    if done_t[e]:
        finished_rewards.append(running_ep_r[e])
```

---

## 七、关键 Bug 记录

| Bug | 后果 | 修复 |
|-----|------|------|
| `values_new` typo → `values_new_b` | value_loss 完全失效，critic 学歪 | 统一变量名 |
| `.cpu()` 不拷贝只改 flag | GAE 混 GPU/CPU tensor 崩溃 | 改用 `.to("cpu").clone()` |
| per-env advantage normalize | 抹平不同 env 真实贡献差异 | 全局 normalize |
| `next_value = val_buf[-1]` | GAE bootstrap 错误 | 用真实 last_values |
| `flatten_buffer` 写反 shape | `[T,N,obs]→[T,N*obs]` 而非 `[T*N,obs]` | 明确 reshape 参数 |
| `ent_buf` 全程存储 | 浪费显存 + 无意义搬运 | 删除，update 时重算 |
| `ppo_loss` 内重复 forward | 同一 minibatch 做两次前向 | 从外部传入 values_new |

---

## 八、超参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_envs` | 8 | 并行 env 数（可试 16/32） |
| `n_steps_per_env` | 512 | 每 env 步数（总步数=4096） |
| `batch_size` | 256 | minibatch 大小 |
| `n_epochs` | 5 | 从 10 降至 5，防止过拟合 |
| `lr` | 3e-4 → 3e-5 | 线性衰减 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE lambda |
| `clip_range` | 0.2 → 0.1 | PPO clip 衰减 |
| `ent_coef` | 0.01 | entropy 系数 |
| `value_coef` | 0.5 | value loss 系数 |
| `target_kl` | 0.015 | KL early stopping 阈值 |

---

## 九、性能优化路线

### 当前架构瓶颈分析

| 阶段 | 瓶颈 | 原因 |
|------|------|------|
| Rollout | CPU | env step 虽轻但 8 进程通信 + tensor 拷贝 |
| Update | GPU | 矩阵运算 |

### 优化方向

1. **CPU 瓶颈**：增加 `n_envs`（16/32），减少 GPU/CPU 同步频率
2. **GPU 瓶颈**：减小 `batch_size`，增加 update 次数
3. **Wall-clock vs 样本效率**：测试 n_envs=8 vs n_envs=16 的 wall-clock 时间

---

## 十、文件结构

```
cartpole_ppo/
├── train.py              # 主训练脚本（单文件）
├── docs/
│   └── ppo_async_design.md  # 本文档
├── requirements.txt
└── README.md
```

---

_文档版本：v1.0_
_编写：凌云（TLM）_
_审核：待团队成员_
