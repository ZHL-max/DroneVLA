# VLA发展历史与脉络
## 从CLIPort到UAV-Flow：VLA的演化之路

---

## 发展时间线

```
2020 ──── 2021 ──── 2022 ──── 2023 ──── 2024 ──── 2025 ──── 2026
  │         │         │         │         │         │         │
  │      CLIPort    SayCan     RT-2    OpenVLA    UAV-Flow   未来
  │      (语言接地)  (可行性)   (动作token) (开源VLA)  (无人机VLA)
  │         │         │         │         │         │
  └─────────┴─────────┴─────────┴─────────┴─────────┘
                    VLA发展主线
```

---

## 第一阶段：语言接地的机器人操作（2020-2022）

### 核心问题：如何让机器人理解自然语言？

在VLA之前，机器人只能执行预编程的任务。这一阶段的目标是让机器人理解"把红色方块放到蓝色方块上"这样的自然语言指令。

### 代表模型

#### 1. CLIPort (2021) - 语言+空间

**论文**：CLIPort: What and Where Pathways for Robotic Manipulation

**核心思想**：
- **What通路**：用CLIP理解"红色方块"是什么（语义）
- **Where通路**：用TransportNet定位"红色方块在哪里"（空间）

```
输入：
- 图像：桌面上有红块和蓝块
- 指令："把红块放到蓝块上"

处理：
CLIP → "红块"的语义特征（what）
TransportNet → "红块"的空间位置（where）
两者融合 → 知道抓哪里、放哪里

输出：
- Pick位置：(x1, y1)
- Place位置：(x2, y2)
```

**关键创新**：首次将CLIP的语义理解能力用于机器人操作

**代码思路**：
```python
# CLIPort的核心思路
import clip

# 编码图像和文本
image_features = clip.encode_image(image)      # what
text_features = clip.encode_text("red block")   # what

# TransportNet提取空间特征
spatial_features = transport_net(image)          # where

# 融合what和where
pick_location = fusion(what_features, where_features)
place_location = fusion(what_features, where_features)
```

**局限**：只能做pick-and-place，不能处理复杂任务

---

#### 2. SayCan (2022) - LLM+机器人能力

**论文**：Do As I Can, Not As I Say: Grounding Language in Robotic Affordances

**核心思想**：
- LLM知道"怎么做"（语义知识）
- 但不知道"能不能做"（物理约束）
- SayCan让LLM和机器人协作

```
用户说："我渴了"

LLM生成候选动作：
1. 倒水（语义得分：0.9）
2. 拿饮料（语义得分：0.7）
3. 去冰箱（语义得分：0.5）

机器人评估可行性：
1. 倒水：手被占用 → 可行性：0.1
2. 拿饮料：旁边有杯子 → 可行性：0.8
3. 去冰箱：距离太远 → 可行性：0.2

最终得分 = 语义得分 × 可行性：
1. 倒水：0.9 × 0.1 = 0.09
2. 拿饮料：0.7 × 0.8 = 0.56  ← 选择这个！
3. 去冰箱：0.5 × 0.2 = 0.10
```

**关键创新**：将LLM的语义知识与机器人的物理能力结合

**局限**：需要预定义技能库，不能端到端学习

---

## 第二阶段：基础模型驱动的VLA（2023-2024）

### 核心突破：把动作变成"文字"

这一阶段的关键洞察是：如果能把机器人动作表示为token，就可以用训练大语言模型的方法训练机器人。

### 代表模型

#### 3. RT-2 (2023) - Google DeepMind

**论文**：RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control

**核心创新**：将动作编码为文本token

```
传统方法：
动作 = [0.5, -0.3, 0.2, 0.1, 0.0, 0.0, 1.0]  → 7个浮点数

RT-2方法：
动作 = "0.127 0.034 0.156 0.000 0.000 0.000 0.543"  → 文本token
```

**为什么这样做？**

因为这样就可以用训练ChatGPT的方法训练机器人！

```
训练数据来源：
1. 网页上的图文对（数万亿token）→ 学习"什么是锤子"
2. 机器人轨迹数据（数十万条）→ 学习"怎么用锤子"

共同训练：
输入：[图像] + "这个图像中应该执行什么动作？"
输出："0.127 0.034 0.156 ..."（动作token）
```

**涌现能力**：

RT-2表现出令人惊讶的泛化能力：
```python
# 测试1：从未见过的物体
指令："把石头放到锤子图标上"
RT-2从未在训练中见过"锤子图标"，但它从网页数据中知道锤子是什么

# 测试2：推理能力
指令："把能量饮料给累了的人"
RT-2推理：累了 → 需要能量 → 选择能量饮料

# 测试3：多步推理
指令："把石头当作锤子用"
RT-2推理：锤子 → 敲击 → 用石头模拟敲击动作
```

**关键贡献**：证明了大模型的语义知识可以迁移到机器人控制

---

#### 4. OpenVLA (2024) - 开源VLA

**论文**：OpenVLA: An Open-Source Vision-Language-Action Model

**核心价值**：让所有人都能用VLA

**架构详解**：

```
输入：
- 图像 (224×224)
- 指令："pick up the red block"

处理：
┌─────────────────────────────────────────────────┐
│                 OpenVLA (7B参数)                 │
│                                                 │
│  ┌──────────┐  ┌──────────┐                    │
│  │ DINOv2   │  │  SigLIP  │  ← 两个视觉编码器  │
│  │ (空间)   │  │  (语义)  │                    │
│  └────┬─────┘  └────┬─────┘                    │
│       └──────┬──────┘                          │
│              │                                  │
│       ┌──────┴──────┐                          │
│       │   Llama-2   │  ← 大语言模型            │
│       │   (7B)      │                          │
│       └──────┬──────┘                          │
│              │                                  │
│       ┌──────┴──────┐                          │
│       │  动作头     │  ← 输出7个数字           │
│       └─────────────┘                          │
└─────────────────────────────────────────────────┘

输出：
action = [x, y, z, roll, pitch, yaw, gripper]
```

**为什么用两个视觉编码器？**

```
DINOv2：擅长"在哪里"
- 自监督学习，不需要标签
- 能精确识别物体边界
- 适合空间定位

SigLIP：擅长"是什么"
- CLIP的改进版
- 能理解语义概念
- 适合任务理解

两者结合：既知道"是什么"，又知道"在哪里"
```

**关键贡献**：
- 开源7B参数的VLA模型
- 提供完整的训练代码
- 建立了VLA的标准benchmark

---

## 第三阶段：世界模型+VLA融合（2025-2026）

### 核心思想：先"想象"再"行动"

这一阶段引入世界模型，让机器人在行动前先预测后果。

### 代表模型

#### 5. Diffusion Policy (2023-2024)

**论文**：Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

**核心创新**：用扩散模型生成动作序列

```
传统方法（确定性）：
状态 → 策略网络 → 一个动作

扩散方法（概率性）：
状态 → 扩散模型 → 多个可能的动作序列
                    ↓
              选择最优的一个
```

**为什么用扩散模型？**

```
问题：机器人任务通常有多种解法
例如："把杯子放到桌上"可以从左边放，也可以从右边放

确定性模型：只输出一个动作（可能不是最优的）
扩散模型：输出多个可能的动作，选择最好的

类比：
确定性模型 = 只会一种解法的学生
扩散模型 = 会多种解法的学霸
```

**扩散过程**：
```python
# 前向过程：给动作加噪声
def add_noise(action, t):
    noise = torch.randn_like(action)
    noisy_action = sqrt(alpha[t]) * action + sqrt(1-alpha[t]) * noise
    return noisy_action

# 反向过程：从噪声中恢复动作
def denoise(noisy_action, t, condition):
    predicted_noise = model(noisy_action, condition, t)
    action = (noisy_action - sqrt(1-alpha[t]) * predicted_noise) / sqrt(alpha[t])
    return action

# 生成动作
def generate_action(condition):
    # 从纯噪声开始
    x = torch.randn(action_shape)
    # 逐步去噪
    for t in reversed(range(T)):
        x = denoise(x, t, condition)
    return x  # 去噪后的动作
```

---

#### 6. World Models (2025-2026)

**核心思想**：在"想象"中训练策略

```
DayDreamer算法：
1. 收集少量真实数据
2. 学习世界模型（预测下一状态和奖励）
3. 在世界模型中"想象"大量轨迹
4. 用想象的数据训练策略
5. 在真实环境中测试
6. 重复

类比：
人类学骑车：
1. 看别人骑（真实数据）
2. 在脑海中想象怎么骑（世界模型）
3. 在想象中练习（想象训练）
4. 实际骑一下（真实测试）
5. 根据结果调整想象（迭代优化）
```

**世界模型的三种角色**：

| 角色 | 功能 | 类比 |
|------|------|------|
| **策略** | 直接输出动作 | 直觉 |
| **模拟器** | 生成想象数据 | 想象力 |
| **生成器** | 预测未来状态 | 预测能力 |

---

#### 7. UAV-Flow (2025) - 北航出品

**论文**：UAV-Flow: Instruction-Conditioned UAV Control

**核心贡献**：
- 首个面向无人机的大规模VLA数据集
- 基于OpenVLA微调的无人机VLA模型
- UnrealZoo仿真评估环境

```
数据集：
- 真实无人机轨迹（HuggingFace）
- 仿真无人机轨迹（HuggingFace）
- 多种飞行场景和指令

模型：
OpenVLA-UAV = OpenVLA 7B + 无人机数据微调

评估：
nDTW（归一化动态时间规整）衡量轨迹质量
```

**对北航学生的意义**：
- 你们学校出品的研究
- 有现成的数据集和代码
- 可以直接复现和改进

---

## 模型对比总结

| 模型 | 年份 | 核心思想 | 视觉编码 | 语言编码 | 动作输出 | 是否开源 |
|------|------|----------|----------|----------|----------|----------|
| CLIPort | 2021 | 语义+空间 | CLIP | CLIP | Pick/Place位置 | ✓ |
| SayCan | 2022 | LLM+可行性 | - | LLM | 技能选择 | ✓ |
| RT-2 | 2023 | 动作token化 | ViT | PaLM | 文本token | ✗ |
| OpenVLA | 2024 | 开源VLA | DINOv2+SigLIP | Llama-2 | 7维向量 | ✓ |
| Diffusion | 2024 | 扩散策略 | CNN/ViT | - | 动作序列 | ✓ |
| UAV-Flow | 2025 | 无人机VLA | OpenVLA | OpenVLA | 速度指令 | ✓ |

---

## 发展趋势

```
趋势1：从专用到通用
CLIPort(只能pick-place) → RT-2(多种任务) → 通用VLA

趋势2：从大到小
RT-2(540B) → OpenVLA(7B) → 轻量级VLA(100M)

趋势3：从仿真到真实
仿真训练 → Sim-to-Real → 真实世界部署

趋势4：从机械臂到无人机
桌面操作 → 移动操作 → 无人机飞行

趋势5：世界模型融合
直接策略 → 预测-执行 → 想象训练
```

---

## 学习建议

1. **先理解CLIPort**：最简单的VLA，理解what和where通路
2. **再学RT-2**：理解动作token化的思想
3. **然后学OpenVLA**：理解完整的VLA架构
4. **最后学UAV-Flow**：理解无人机VLA的特殊性

---

*下一节：[核心概念速查](03_Key_Concepts.md)*
