# CartPole-v1 强化学习项目

## 📋 项目概述

### 任务描述

CartPole 是强化学习经典的控制问题：

- **目标**：通过左右推动小车，保持垂直于小车上的杆子不倒下
- **环境**：`gymnasium.make("CartPole-v1")`
- **难度**：经典入门任务，但完全收敛到 500 steps 需要正确的算法设计

### 环境规格

| 项目 | 值 |
|------|-----|
| Action Space | Discrete(2) — 左(0) / 右(1) |
| Observation Space | Box(4,) — 位置、速度、角度、角速度 |
| 终止条件 | 角度 > ±12° 或 位置 > ±2.4 |
| 最大步数 | 500 (v1) / 200 (v0) |
| 奖励 | 每步 +1 |

### 理论极限

- **随机策略**：约 10-20 steps
- **强 Baseline**：500 steps（环境上限）
- **理论最优**：500 steps

---

## 🎯 项目目标

| 里程碑 | 目标 | 验收标准 |
|--------|------|----------|
| M1 | 基础收敛 | 平均 200+ steps，3 seeds 稳定 |
| M2 | 强 Baseline | 平均 400+ steps，3 seeds 稳定 |
| M3 | 接近最优 | 平均 450+ steps，3 seeds 稳定 |

---

## 🧠 技术方案

### 候选方案

| 方案 | 优点 | 缺点 |
|------|------|------|
| **PPO** | 稳定、通用、参考多 | 超参数敏感 |
| **DQN** | 简单、效果好 | 离散动作专用 |
| **A2C/A3C** | 并行、样本效率高 | 实现复杂 |

### 初选方案：PPO

原因：
1. 通用性强，适合作为后续其他任务的基线
2. 稳定性和样本效率平衡好
3. 工程实现成熟，参考资料多

---

## 📁 项目结构

```
C:\gym-learning\cartpole_ppo\
├── README.md            # 本文档
├── train.py            # PPO 训练代码
├── eval.py             # 评估代码
├── ppo_impl.py         # PPO 核心实现
├── eval_framework.py   # 评估框架（从 acrobot_ppo 复用）
├── requirements.txt    # 依赖
├── docs/               # 详细技术文档
└── logs/               # 训练日志
```

---

## 📊 评估指标

训练过程中记录：
- **Episode Length** — 每 episode 的持续步数（核心指标）
- **Training Loss** — PPO 损失变化
- **Policy Entropy** — 探索程度
- **KL Divergence** — 策略更新幅度

最终评估：3 seeds × 50 episodes，取平均

---

## 🚀 运行方式（待实现）

```bash
# 训练
python train.py --seeds 3 --steps 500000

# 评估
python eval.py --checkpoint ppo_final.pt --episodes 50
```

---

## 📝 参考文献

1. Barto, Sutton, Anderson - "Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem" (1983)
2. Schulman et al. - "Proximal Policy Optimization Algorithms" (2017)

---

## ✅ 任务清单

- [ ] 项目结构搭建
- [ ] PPO 实现
- [ ] 训练脚本
- [ ] 评估脚本
- [ ] M1 验收：200+ steps
- [ ] M2 验收：400+ steps
- [ ] M3 验收：450+ steps

---

_项目创建日期：2026-04-08_
