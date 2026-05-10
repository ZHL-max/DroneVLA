# VLA (Vision-Language-Action) 学习笔记
## 从基础概念到前沿发展 - 面向无人机应用

> 本文档基于对 [World Model for Robot Learning Survey](https://arxiv.org/abs/2605.00080) 和 [Awesome-World-Model-for-Robotics-Policy](https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy) 的深度调研整理而成。

---

## 目录

1. [基础概念与发展历程](#1-基础概念与发展历程)
2. [VLA核心架构](#2-vla核心架构)
3. [世界模型与VLA的融合](#3-世界模型与vla的融合)
4. [关键技术范式](#4-关键技术范式)
5. [无人机VLA的特殊考量](#5-无人机vla的特殊考量)
6. [前沿发展与未来方向](#6-前沿发展与未来方向)
7. [学习路线图](#7-学习路线图)
8. [参考文献与资源](#8-参考文献与资源)

---

## 1. 基础概念与发展历程

### 1.1 什么是VLA？

**Vision-Language-Action (VLA)** 是一类将视觉感知、语言理解和动作执行统一到单一模型架构中的机器人学习方法。其核心思想是：

```
输入：视觉观测 (Vision) + 语言指令 (Language)
输出：机器人动作 (Action)
```

VLA模型的本质是**多模态条件策略**：给定当前视觉场景和自然语言任务描述，直接输出机器人应该执行的动作序列。

### 1.2 发展历程

#### 第一阶段：语言接地的机器人操作（2020-2022）

| 里程碑 | 核心思想 | 代表工作 |
|--------|----------|----------|
| **CLIPort** | CLIP语义 + TransporterNet空间精度 | cliport/cliport |
| **SayCan** | LLM规划 + 机器人技能接地 | Google, 2022 |
| **RT-1** | 大规模演示数据的Transformer策略 | Google, 2022 |

**关键洞察**：
- CLIPort: "结合CLIP的广泛语义理解(what)与TransporterNets的空间精度(where)"
- SayCan: "LLM提供高层语义知识，机器人技能提供物理环境的接地"

#### 第二阶段：基础模型驱动的VLA（2023-2024）

| 里程碑 | 核心创新 | 关键突破 |
|--------|----------|----------|
| **RT-2** | 动作表示为文本token | 涌现能力：语义推理、泛化 |
| **OpenVLA** | 开源VLA，7B参数 | 970K轨迹训练，LoRA微调 |
| **Octo** | 通用机器人策略 | 800K轨迹，多 embodiment |

**RT-2的核心贡献**：
> "将自然语言响应和机器人动作拟合到相同的格式中"——通过co-fine-tuning同时训练视觉-语言任务和机器人轨迹数据。

#### 第三阶段：世界模型+VLA融合（2025-2026）

| 范式 | 核心思想 | 代表工作 |
|------|----------|----------|
| **IDM策略** | 先预测未来视觉，再恢复动作 | UniPi, GR-1, VPP |
| **单骨干策略** | 联合建模视频和动作 | UVA, UD-VLA |
| **MoE策略** | 视频和动作专家分离交互 | GE-Act, Motus |
| **统一VLA** | 世界建模作为训练目标 | GR-2, DreamVLA |
| **隐空间建模** | JEPA风格的表征预测 | FLARE, VLA-JEPA |

---

## 2. VLA核心架构

### 2.1 基础架构组件

一个典型的VLA模型包含以下组件：

```
┌─────────────────────────────────────────────────┐
│                   VLA Model                      │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Vision   │  │ Language  │  │   Action     │  │
│  │  Encoder  │  │  Encoder  │  │   Decoder    │  │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  │
│       │              │               │          │
│       └──────┬───────┘               │          │
│              │                       │          │
│       ┌──────┴───────┐               │          │
│       │  Multimodal   │──────────────┘          │
│       │  Fusion       │                         │
│       └──────────────┘                          │
│                                                  │
└─────────────────────────────────────────────────┘
```

### 2.2 视觉编码器选择

| 编码器 | 特点 | 适用场景 |
|--------|------|----------|
| **CLIP** | 语义理解强 | 任务描述理解 |
| **SigLIP** | 更好的视觉-语言对齐 | 精确的空间定位 |
| **DINOv2** | 自监督，空间特征好 | 物体识别和定位 |
| **DINOv2+SigLIP** | 融合优势 | OpenVLA默认选择 |

### 2.3 语言模型选择

| 模型 | 参数量 | 特点 |
|------|--------|------|
| **Llama-2** | 7B-70B | OpenVLA使用，社区许可 |
| **Vicuna** | 7B-13B | 早期VLA模型使用 |
| **PaLM** | 540B | SayCan使用，强大推理 |
| **Gemini** | 多规模 | Google最新VLA研究 |

### 2.4 动作输出方式

**方式1：离散化token（RT-2风格）**
```python
# 将连续动作离散化为256个bin
action_bins = np.linspace(-1, 1, 256)
discretized_action = np.digitize continuous_action, action_bins)
# 作为文本token输出
action_tokens = [str(bin) for bin in discretized_action]
```

**方式2：连续回归（OpenVLA风格）**
```python
# 直接回归7-DoF动作
action = model.predict_action(observation, instruction)
# action = [x, y, z, roll, pitch, yaw, gripper]
```

**方式3：扩散生成（Diffusion Policy风格）**
```python
# 通过迭代去噪生成动作序列
noise = torch.randn(batch_size, action_horizon, action_dim)
for t in reversed(range(diffusion_steps)):
    action = denoise_model(noise, observation, t)
```

---

## 3. 世界模型与VLA的融合

### 3.1 世界模型的核心定义

> "世界模型是关于环境如何在动作下演化的预测性表征" —— 综述论文

世界模型的三大功能角色：
1. **做策略 (Policy)**：直接作为或增强机器人策略
2. **做模拟器 (Simulator)**：作为学习环境进行策略训练
3. **做生成 (Generation)**：生成未来视觉预测

### 3.2 世界模型作为策略

#### IDM（逆动力学模型）策略

**核心流程**：先想象未来 → 再恢复动作

```
语言指令 → [视频生成模型] → 未来视觉轨迹 → [逆动力学模型] → 机器人动作
```

**代表工作**：
- **UniPi** (NeurIPS'23): 文本引导视频生成的通用策略
- **GR-1** (ICLR'24): 大规模视频生成预训练
- **VPP** (ICML'25): 视频预测策略

#### 单骨干策略

**核心思想**：一个模型同时生成视频和动作

```
观测 + 指令 → [统一骨干网络] → 视频预测 + 动作预测
```

**代表工作**：
- **UVA** (RSS'25): 统一视频动作模型
- **UD-VLA** (ICLR'26): 统一扩散VLA

#### MoE/MoT策略

**核心思想**：视频专家和动作专家分离，通过注意力交互

```
观测 → [视频专家] ←交叉注意力→ [动作专家] → 动作
```

**代表工作**：
- **GE-Act** (ICLR'26): 并行流匹配动作专家
- **Motus** (arXiv'25.12): 统一隐空间动作世界模型

### 3.3 世界模型作为模拟器

**核心价值**：在想象的环境中训练策略，减少真实交互

```
┌─────────────────────────────────────────┐
│          World Model Simulator          │
│                                         │
│  真实观测 → [世界模型] → 预测的未来状态  │
│              ↑                          │
│          候选动作                        │
│                                         │
│  策略在预测的未来中训练和改进            │
└─────────────────────────────────────────┘
```

**代表工作**：
- **DayDreamer** (CoRL'23): 物理机器人的世界模型学习
- **WMPO** (ICLR'26): 基于世界模型的策略优化
- **RehearseVLA** (CVPR'26): 物理一致的模拟后训练

### 3.4 世界模型用于视频生成

**核心应用**：生成机器人训练数据和未来预测

**代表工作**：
- **UniSim** (ICLR'24): 学习交互式真实世界模拟器
- **Cosmos Predict 2.5** (2025): 基础视频世界模型
- **GigaWorld-0** (arXiv'25.11): 大规模世界模型

---

## 4. 关键技术范式

### 4.1 从RT-2到OpenVLA：VLA的演化

```
RT-2 (2023)                    OpenVLA (2024)
├── 专有模型                    ├── 开源模型
├── PaLM骨干                    ├── Llama-2骨干
├── 动作离散化为token            ├── 连续动作回归
├── 540B参数                    ├── 7B参数
└── Google内部                   └── 社区可复现
```

### 4.2 扩散策略 (Diffusion Policy)

**核心创新**：将动作生成建模为条件去噪过程

```python
# 训练：前向扩散
noise = torch.randn_like(action)
noisy_action = add_noise(action, noise, timestep)
predicted_noise = model(noisy_action, observation, timestep)
loss = MSE(noise, predicted_noise)

# 推理：反向去噪
action = torch.randn(action_shape)
for t in reversed(range(T)):
    action = denoise_step(model, action, observation, t)
```

**优势**：
- 捕获多模态动作分布
- 平滑的动作序列生成
- 适合复杂操作任务

### 4.3 视频预测策略 (Video Prediction Policy)

**VPP的核心流程**：
```
当前观测 → [视频预测模型] → 预测未来帧 → [视觉表征提取] → [动作解码器] → 动作
```

**关键洞察**：视频预测提供丰富的视觉表征，可以作为动作预测的先验。

### 4.4 JEPA风格的隐空间建模

**核心思想**：在表征空间而非像素空间进行预测

```
观测 → [编码器] → 隐表征 → [预测器] → 未来隐表征 → [解码器] → 动作
```

**代表工作**：
- **FLARE** (CoRL'25): 隐式世界建模的机器人学习
- **VLA-JEPA** (arXiv'26.02): 增强VLA的隐空间世界模型

---

## 5. 无人机VLA的特殊考量

### 5.1 与桌面操作的关键差异

| 维度 | 桌面操作 | 无人机操作 |
|------|----------|------------|
| **动作空间** | 6-7 DoF (位置+旋转+夹爪) | 4-6 DoF (速度/加速度指令) |
| **控制频率** | 5-20 Hz | 50-200 Hz |
| **观测空间** | 固定相机 | 机载相机(第一人称) |
| **动力学** | 准静态 | 高度动态 |
| **安全约束** | 桌面边界 | 3D空间+碰撞避免 |
| **延迟容忍** | 较高 | 极低 |

### 5.2 无人机动作空间设计

**方案1：低级控制指令**
```python
# 直接输出电机推力
action = [thrust_1, thrust_2, thrust_3, thrust_4]  # 4-DoF
```

**方案2：速度指令**
```python
# 输出机体坐标系速度
action = [vx, vy, vz, yaw_rate]  # 4-DoF
```

**方案3：位置/航点指令**
```python
# 输出目标位置
action = [x, y, z, yaw]  # 4-DoF
```

**方案4：加速度指令**
```python
# 输出期望加速度(用于底层控制器)
action = [ax, ay, az, yaw_rate]  # 4-DoF
```

### 5.3 无人机视觉处理

**挑战**：
- 运动模糊
- 光照变化大
- 遮挡和视角变化
- 实时性要求高

**解决方案**：
```python
class DroneVisualEncoder(nn.Module):
    def __init__(self):
        # 使用轻量级视觉编码器
        self.backbone = EfficientNet-B0  # 或 MobileNetV3
        self.temporal = TemporalAttention()  # 时序融合
        
    def forward(self, images):
        # 多帧输入处理
        features = self.backbone(images)  # [B, T, C, H, W]
        temporal_features = self.temporal(features)
        return temporal_features
```

### 5.4 无人机世界模型

**特殊需求**：
- 需要预测3D空间中的运动
- 考虑空气动力学效应
- 处理GPS/IMU融合
- 实时性要求

**架构设计**：
```python
class DroneWorldModel(nn.Module):
    def __init__(self):
        self.visual_encoder = DroneVisualEncoder()
        self.state_encoder = StateEncoder()  # 编码IMU/GPS
        self.dynamics_predictor = DynamicsPredictor()
        self.video_predictor = VideoPredictor()
        
    def predict_future(self, obs, action):
        visual_features = self.visual_encoder(obs.images)
        state_features = self.state_encoder(obs.state)
        
        # 预测未来状态
        future_state = self.dynamics_predictor(
            visual_features, state_features, action
        )
        
        # 预测未来视觉
        future_video = self.video_predictor(
            visual_features, action
        )
        
        return future_state, future_video
```

---

## 6. 前沿发展与未来方向

### 6.1 2025-2026年关键趋势

1. **统一架构**：越来越多的工作将视频生成、动作预测、世界建模统一到单一模型
2. **大规模预训练**：从百万级轨迹数据中学习通用机器人知识
3. **隐空间效率**：JEPA风格的方法避免像素级预测的计算开销
4. **闭环学习**：世界模型和策略的迭代改进
5. **多embodiment**：单一模型适应不同机器人形态

### 6.2 开放挑战

| 挑战 | 描述 | 当前进展 |
|------|------|----------|
| **实时性** | VLA模型推理速度慢 | OFT: 25-50x加速 |
| **泛化性** | 跨场景/物体泛化 | 大规模预训练改善 |
| **安全性** | 动态环境中的安全保障 | 世界模型预测+安全约束 |
| **数据效率** | 需要大量演示数据 | 少样本微调、合成数据 |
| **Sim2Real** | 仿真到真实的迁移 | 域随机化、世界模型 |

### 6.3 无人机VLA的未来方向

1. **航拍视频理解**：从大规模航拍视频中学习飞行策略
2. **多无人机协作**：VLA驱动的编队飞行
3. **动态环境适应**：风扰、障碍物的实时适应
4. **长航程规划**：结合世界模型的长距离任务规划
5. **人机协作**：自然语言指令的无人机控制

---

## 7. 学习路线图

### 阶段1：基础（2-4周）

**目标**：理解VLA的核心概念

**学习内容**：
- [ ] CLIP模型原理和使用
- [ ] 基础机器人操作环境（Gymnasium）
- [ ] 简单的模仿学习策略
- [ ] PyTorch基础和Transformer架构

**实践项目**：
- `demos/01_simple_vla/` - 最简单的VLA示例

### 阶段2：核心VLA（4-6周）

**目标**：掌握主流VLA架构

**学习内容**：
- [ ] RT-2论文精读和实现
- [ ] OpenVLA架构和微调
- [ ] 扩散策略原理和实现
- [ ] CLIPort和TransportNet

**实践项目**：
- `demos/02_clip_transport/` - CLIPort复现
- `demos/03_diffusion_policy/` - 扩散策略实现

### 阶段3：世界模型（4-6周）

**目标**：理解世界模型与VLA的融合

**学习内容**：
- [ ] 视频预测模型（VideoGPT, Cosmos）
- [ ] IDM策略流程
- [ ] 隐空间世界建模（JEPA）
- [ ] 世界模型作为模拟器

**实践项目**：
- `demos/04_world_model/` - 世界模型演示

### 阶段4：无人机专项（6-8周）

**目标**：掌握无人机VLA的特殊技术

**学习内容**：
- [ ] 无人机动力学和控制
- [ ] Gym-PyBullet-Drones环境
- [ ] PX4/ArduPilot接口
- [ ] 机载视觉处理
- [ ] 实时控制策略

**实践项目**：
- 主项目 `DroneVLA` 的完整实现

---

## 8. 参考文献与资源

### 核心论文

1. **RT-2**: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control" (arXiv:2307.15818)
2. **OpenVLA**: "OpenVLA: An Open-Source Vision-Language-Action Model" (GitHub: openvla/openvla)
3. **Diffusion Policy**: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023)
4. **World Model Survey**: "World Model for Robot Learning: A Comprehensive Survey" (arXiv:2605.00080)

### GitHub仓库

- [Awesome-World-Model-for-Robotics-Policy](https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy) - 论文汇总
- [OpenVLA](https://github.com/openvla/openvla) - 开源VLA模型
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) - 扩散策略
- [Octo](https://github.com/octo-models/octo) - 通用机器人策略
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) - 机器人学习基准
- [Gym-PyBullet-Drones](https://github.com/utiasDSL/gym-pybullet-drones) - 无人机仿真

### 学习资源

- [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms) - VLM框架
- [CLIPort](https://github.com/cliport/cliport) - 语言接地操作
- [AirSim](https://github.com/microsoft/AirSim) - 无人机仿真（已归档）

---

*最后更新：2026-05-10*
*作者：DroneVLA Project*
