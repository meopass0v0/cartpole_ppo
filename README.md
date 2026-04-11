# CartPole-v1 PPO 项目

## 📋 项目概述

### 任务描述

CartPole 是强化学习经典控制问题：
- **目标**：通过左右推动小车，保持杆子不倒下
- **环境**：`gymnasium.make("CartPole-v1")`
- **难度**：入门任务，收敛到 500 steps 需要正确的算法设计

### 环境规格

| 项目 | 值 |
|------|-----|
| Action Space | Discrete(2) — 左(0) / 右(1) |
| Observation Space | Box(4,) — 位置、速度、角度、角速度 |
| 终止条件 | 角度 > ±12° 或 位置 > ±2.4 |
| 最大步数 | 500 (v1) |
| 奖励 | 每步 +1 |

---

## 🎯 技术目标

| 里程碑 | 目标 | 验收标准 |
|--------|------|----------|
| M1 | 基础收敛 | 200+ steps |
| M2 | 强 Baseline | 400+ steps |
| M3 | 接近最优 | 450+ steps |

---

## 🏗️ 技术架构

### AsyncVectorEnv 并行采样

```
8 并行 env（AsyncVectorEnv）
- Rollout: 512 steps/env × 8 envs = 4096 步/rollout
- Buffer 全 CPU，update 时一次性搬到 GPU
- 避免每步 GPU/CPU 互相拷贝
```

详见：[docs/ppo_async_design.md](docs/ppo_async_design.md)

### 手写 PPO 实现

- Actor-Critic 共享网络（MLP 4→128→128→2）
- GAE (λ=0.95) + PPO Clipped Objective
- Value Clipping（对称于 Policy Clip）
- KL Early Stopping + LR/Clip Decay
- 稳定的 KL 估计：`((ratio-1) - log_ratio).mean()`

### 核心参数

| 参数 | 值 |
|------|-----|
| `n_envs` | 8 |
| `n_steps_per_env` | 512 |
| `batch_size` | 256 |
| `n_epochs` | 5 |
| `lr` | 3e-4 → 3e-5 (decay) |
| `gamma` | 0.99 |
| `lam` | 0.95 |
| `clip_range` | 0.2 → 0.1 |
| `target_kl` | 0.015 |

---

## 📁 项目结构

```
cartpole_ppo/
├── train.py                  # 统一训练脚本（PPO + Metrics + 视频）
├── docs/
│   └── ppo_async_design.md  # 架构设计文档（团队 review 用）
├── requirements.txt
└── README.md
```

---

## 🚀 运行方式

```bash
pip install -r requirements.txt
python train.py --steps 4000000
```

训练完成后自动生成：
- `metrics_YYYYMMDD_HHMMSS.png` — 训练曲线
- `ppo_cartpole_YYYYMMDD_HHMMSS.pt` — 模型权重
- `video_success_YYYYMMDD_HHMMSS.mp4` — 成功案例视频

---

## 📊 训练 Metrics

| 指标 | 说明 |
|------|------|
| Episode Reward | 每 episode 累计 reward |
| Episode Length | 持续步数（核心评估指标） |
| Policy Loss | PPO 策略损失 |
| Value Loss | Critic 损失（含 Value Clipping） |
| Entropy | 策略熵（探索程度） |
| KL Divergence | 策略更新幅度（early stopping 用） |

---

## 🔑 关键设计决策

1. **Buffer 全 CPU**：减少 GPU/CPU 拷贝开销，rollout 结束时一次性搬运
2. **跨 rollout episode 追踪**：running_ep_r/l 放在 update 循环外部
3. **GAE 全局 normalize**：不是 per-env，保持不同 env 的真实贡献差异
4. **真实 last_values**：rollout 结束后用 model 预测，不是用最后一个采样 value
5. **Entropy 不存 buffer**：update 时重新算，省显存

---

## ⚠️ 已知 Bug（已修复）

详见 [docs/ppo_async_design.md](docs/ppo_async_design.md#七关键-bug-记录)

| Bug | 后果 |
|-----|------|
| `.cpu()` 不拷贝只改 flag | GAE 混 GPU/CPU 崩溃 |
| `values_new` typo | value_loss 失效 |
| per-env advantage normalize | 抹平不同 env 贡献差异 |

---

## ✅ 任务清单

- [x] AsyncVectorEnv 并行采样
- [x] 手写 PPO 实现
- [x] Value Clipping
- [x] KL Early Stopping
- [x] LR/Clip Decay
- [x] CPU/GPU 分离策略
- [x] 架构设计文档
- [ ] M1 验收：200+ steps
- [ ] M2 验收：400+ steps
- [ ] M3 验收：450+ steps

---

_项目创建日期：2026-04-08_
_最后更新：2026-04-11（AsyncVectorEnv 重构）_
