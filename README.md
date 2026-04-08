# CartPole-v1 强化学习项目

## 📋 项目概述

### 任务描述

CartPole 是强化学习经典的控制问题：

- **目标**：通过左右推动小车，保持垂直于小车上的杆子不倒下
- **环境**：`gymnasium.make("CartPole-v1")`
- **难度**：经典入门任务，完全收敛到 500 steps 需要正确的算法设计

### 环境规格

| 项目 | 值 |
|------|-----|
| Action Space | Discrete(2) — 左(0) / 右(1) |
| Observation Space | Box(4,) — 位置、速度、角度、角速度 |
| 终止条件 | 角度 > ±12° 或 位置 > ±2.4 |
| 最大步数 | 500 (v1) |
| 奖励 | 每步 +1 |

### 理论极限

- **随机策略**：约 10-20 steps
- **强 Baseline**：500 steps（环境上限）
- **理论最优**：500 steps

---

## 🎯 项目目标

| 里程碑 | 目标 | 验收标准 |
|--------|------|----------|
| M1 | 基础收敛 | 200+ steps |
| M2 | 强 Baseline | 400+ steps |
| M3 | 接近最优 | 450+ steps |

---

## 🧠 技术方案

### 手写 PPO

基于 acrobot_ppo 框架，手写实现 Actor-Critic + GAE + PPO Clipped Objective：

```
网络结构：MLP (输入4 → 128 → 128 → 输出)
- Actor head：2个动作的 log 概率
- Critic head：状态价值 V(s)

训练参数：
- n_steps=2048 | batch_size=64 | epochs=10
- lr=3e-4 | gamma=0.99 | lam=0.95 | clip=0.2
```

### Metrics 指标

训练过程中记录：
| 指标 | 说明 |
|------|------|
| Episode Reward | 每 episode 累计 reward |
| Episode Length | 持续步数（核心） |
| Policy Loss | PPO 策略损失 |
| Value Loss | Critic 损失 |
| Entropy | 策略熵（探索程度） |
| KL Divergence | 策略更新幅度 |

---

## 📁 项目结构

```
cartpole_ppo\
├── train.py            # 统一训练脚本（PPO + Metrics + 视频）
├── requirements.txt    # 依赖
└── README.md
```

---

## 🚀 运行方式

```bash
pip install -r requirements.txt
python train.py --steps 100000
```

训练完成后自动生成：
- `metrics_YYYYMMDD_HHMMSS.png` — 训练曲线
- `ppo_cartpole_YYYYMMDD_HHMMSS.pt` — 模型权重
- `video_success_YYYYMMDD_HHMSS.mp4` — 成功案例视频

---

## 📝 参考文献

1. Barto, Sutton, Anderson - "Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem" (1983)
2. Schulman et al. - "Proximal Policy Optimization Algorithms" (2017)

---

## ✅ 任务清单

- [x] 手写 PPO 实现
- [x] Metrics 指标（Loss、Entropy、KL、Value Loss）
- [x] 中间状态展示
- [x] 视频录制（1 case）
- [ ] M1 验收：200+ steps
- [ ] M2 验收：400+ steps
- [ ] M3 验收：450+ steps

---

_项目创建日期：2026-04-08_
