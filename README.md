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

### 初选方案：PPO

原因：
1. 通用性强，适合作为后续其他任务的基线
2. 稳定性和样本效率平衡好
3. 工程实现成熟，参考资料多

### 网络结构

```
Actor-Critic 共享骨干网络：MLP (输入4 → 128 → 64 → 输出)
- Actor head：2个动作的 log概率
- Critic head：状态价值 V(s)
```

---

## 📁 项目结构

```
cartpole_ppo\
├── README.md            # 本文档
├── train.py            # PPO 训练脚本
├── eval.py             # 评估脚本
├── ppo_impl.py         # PPO 核心实现
├── eval_framework.py   # 评估框架（从 acrobot_ppo 复用）
├── video.py            # 视频录制脚本
├── requirements.txt    # 依赖
└── logs/               # 训练日志
```

---

## 📊 评估指标

| 指标 | 说明 |
|------|------|
| Episode Length | 每 episode 的持续步数（核心） |
| Training Loss | PPO 损失变化 |
| Policy Entropy | 探索程度 |
| KL Divergence | 策略更新幅度 |
| Value Loss | Critic 损失 |

---

## 🎬 视频采样

训练完成后录制 1 个成功案例视频，验证学习效果。

---

## 🚀 运行方式

```bash
# 训练
python train.py --steps 500000

# 评估
python eval.py --checkpoint ppo_final.pt --episodes 50

# 录制视频
python video.py --checkpoint ppo_final.pt
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
- [ ] 视频录制
- [ ] M1 验收：200+ steps
- [ ] M2 验收：400+ steps
- [ ] M3 验收：450+ steps

---

_项目创建日期：2026-04-08_
