# 无人机VLA项目指南
## Drone Vision-Language-Action Project Guide

---

## 项目概述

本项目旨在构建一个面向无人机的Vision-Language-Action (VLA) 系统，使无人机能够：
- 理解自然语言指令
- 通过视觉感知环境
- 自主执行飞行动作

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    DroneVLA System                           │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │  Camera   │    │   IMU    │    │   GPS    │             │
│  │  Input    │    │  Sensor  │    │  Sensor  │             │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘             │
│       │               │               │                    │
│       └───────────────┼───────────────┘                    │
│                       │                                    │
│              ┌────────┴────────┐                           │
│              │  State Encoder  │                           │
│              └────────┬────────┘                           │
│                       │                                    │
│  ┌────────────────────┼────────────────────┐              │
│  │                    │                    │              │
│  ▼                    ▼                    ▼              │
│ ┌────────┐    ┌──────────────┐    ┌──────────────┐       │
│ │Visual  │    │   Language   │    │   World      │       │
│ │Encoder │    │   Encoder    │    │   Model      │       │
│ └───┬────┘    └──────┬───────┘    └──────┬───────┘       │
│     │                │                    │              │
│     └────────────────┼────────────────────┘              │
│                      │                                   │
│              ┌───────┴───────┐                           │
│              │  Multimodal   │                           │
│              │    Fusion     │                           │
│              └───────┬───────┘                           │
│                      │                                   │
│              ┌───────┴───────┐                           │
│              │    Action     │                           │
│              │   Decoder     │                           │
│              └───────┬───────┘                           │
│                      │                                   │
│              ┌───────┴───────┐                           │
│              │   Flight      │                           │
│              │  Controller   │                           │
│              └───────────────┘                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 模块说明

### 1. 视觉编码器 (Visual Encoder)

**输入**：机载相机图像流
**输出**：视觉特征向量

```python
class DroneVisualEncoder(nn.Module):
    """
    无人机视觉编码器
    
    处理机载相机图像，提取视觉特征
    支持多帧时序融合
    """
    def __init__(self, backbone='efficientnet_b0', temporal=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True)
        self.temporal = TemporalAttention() if temporal else None
        
    def forward(self, images):
        """
        Args:
            images: [B, T, C, H, W] 多帧图像
        Returns:
            features: [B, D] 视觉特征
        """
        B, T, C, H, W = images.shape
        # 提取每帧特征
        features = []
        for t in range(T):
            feat = self.backbone(images[:, t])  # [B, D]
            features.append(feat)
        features = torch.stack(features, dim=1)  # [B, T, D]
        
        # 时序融合
        if self.temporal:
            features = self.temporal(features)
            
        return features
```

### 2. 语言编码器 (Language Encoder)

**输入**：自然语言指令
**输出**：语言特征向量

```python
class LanguageEncoder(nn.Module):
    """
    语言编码器
    
    使用预训练语言模型编码自然语言指令
    """
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def forward(self, instructions):
        """
        Args:
            instructions: List[str] 自然语言指令
        Returns:
            features: [B, D] 语言特征
        """
        inputs = self.tokenizer(
            instructions, 
            return_tensors='pt', 
            padding=True, 
            truncation=True
        )
        outputs = self.model(**inputs)
        # 使用[CLS] token作为句子表示
        features = outputs.last_hidden_state[:, 0]
        return features
```

### 3. 世界模型 (World Model)

**输入**：当前状态 + 动作
**输出**：预测的未来状态和视觉

```python
class DroneWorldModel(nn.Module):
    """
    无人机世界模型
    
    预测给定动作下的未来状态和视觉
    用于：
    1. 策略训练（在想象中训练）
    2. 策略评估（预测行为后果）
    3. 规划（选择最优动作序列）
    """
    def __init__(self):
        super().__init__()
        self.state_predictor = nn.LSTM(input_dim, hidden_dim, num_layers=2)
        self.video_predictor = VideoPredictionNetwork()
        
    def predict_future(self, state, action, horizon=10):
        """
        预测未来horizon步的状态和视觉
        
        Args:
            state: 当前状态
            action: 执行的动作
            horizon: 预测步数
        Returns:
            future_states: 预测的未来状态序列
            future_videos: 预测的未来视觉序列
        """
        future_states = []
        future_videos = []
        
        current_state = state
        for _ in range(horizon):
            # 预测下一状态
            next_state = self.state_predictor(current_state, action)
            # 预测下一视觉
            next_video = self.video_predictor(current_state, action)
            
            future_states.append(next_state)
            future_videos.append(next_video)
            
            current_state = next_state
            
        return future_states, future_videos
```

### 4. 动作解码器 (Action Decoder)

**输入**：多模态融合特征
**输出**：无人机控制指令

```python
class ActionDecoder(nn.Module):
    """
    动作解码器
    
    将多模态特征解码为无人机控制指令
    
    动作空间：
    - vx, vy, vz: 机体坐标系速度 (m/s)
    - yaw_rate: 偏航角速度 (rad/s)
    """
    def __init__(self, input_dim, action_dim=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # 输出归一化到[-1, 1]
        )
        
    def forward(self, features):
        """
        Args:
            features: [B, D] 多模态融合特征
        Returns:
            action: [B, 4] 无人机控制指令
        """
        action = self.mlp(features)
        # 缩放到实际范围
        action = action * torch.tensor([2.0, 2.0, 1.0, 1.0])  # vx,vy: ±2m/s, vz: ±1m/s, yaw: ±1rad/s
        return action
```

---

## 环境配置

### 依赖项

```bash
# 基础依赖
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
timm>=0.9.0

# 无人机仿真
gymnasium>=0.29.0
pybullet>=3.2.5
gym-pybullet-drones>=2.0.0

# 视觉处理
opencv-python>=4.8.0
Pillow>=10.0.0

# 强化学习
stable-baselines3>=2.0.0
tensorboard>=2.14.0

# 工具
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

### 安装步骤

```bash
# 1. 创建虚拟环境
conda create -n dronevla python=3.10 -y
conda activate dronevla

# 2. 安装PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. 安装项目
cd DroneVLA
pip install -e .

# 4. 安装无人机仿真
pip install gym-pybullet-drones

# 5. 验证安装
python scripts/test_installation.py
```

---

## 训练流程

### 阶段1：数据收集

```bash
# 使用仿真环境收集演示数据
python scripts/collect_demos.py \
    --env hover \
    --num_episodes 1000 \
    --save_dir data/demos/
```

### 阶段2：预训练

```bash
# 预训练视觉编码器
python scripts/pretrain_visual.py \
    --data_dir data/demos/ \
    --epochs 50 \
    --output models/visual_encoder/

# 预训练语言编码器
python scripts/pretrain_language.py \
    --data_dir data/demos/ \
    --epochs 30 \
    --output models/language_encoder/
```

### 阶段3：VLA训练

```bash
# 训练完整VLA模型
python scripts/train_vla.py \
    --config configs/vla_default.yaml \
    --data_dir data/demos/ \
    --epochs 100 \
    --output models/vla/
```

### 阶段4：世界模型训练

```bash
# 训练世界模型
python scripts/train_world_model.py \
    --config configs/world_model.yaml \
    --data_dir data/demos/ \
    --epochs 50 \
    --output models/world_model/
```

### 阶段5：联合微调

```bash
# 使用世界模型微调VLA策略
python scripts/finetune_with_wm.py \
    --vla_model models/vla/ \
    --wm_model models/world_model/ \
    --epochs 20 \
    --output models/vla_finetuned/
```

---

## 评估流程

### 仿真评估

```bash
# 在仿真环境中评估
python scripts/evaluate.py \
    --model models/vla_finetuned/ \
    --env hover \
    --num_episodes 100 \
    --render True
```

### 指标说明

| 指标 | 描述 | 目标值 |
|------|------|--------|
| **Success Rate** | 任务完成率 | >80% |
| **Average Reward** | 平均累积奖励 | >0.8 |
| **Collision Rate** | 碰撞率 | <5% |
| **Inference Time** | 推理时间 | <20ms |

---

## 部署到真实无人机

### 硬件要求

- 飞控：PX4或ArduPilot兼容
- 计算板：NVIDIA Jetson Orin Nano或更高
- 相机：RGB相机（推荐RealSense D435）
- 通信：MAVLink支持

### 部署步骤

```bash
# 1. 导出模型为ONNX
python scripts/export_onnx.py \
    --model models/vla_finetuned/ \
    --output models/dronevla.onnx

# 2. 部署到Jetson
python scripts/deploy_jetson.py \
    --model models/dronevla.onnx \
    --connection /dev/ttyUSB0
```

---

## 安全注意事项

1. **始终在仿真中验证后再部署到真实无人机**
2. **设置安全边界和紧急停止机制**
3. **监控电池电量和信号强度**
4. **在开阔场地进行首次飞行测试**
5. **保持手动遥控器在手，随时准备接管**

---

*最后更新：2026-05-10*
*DroneVLA Project*
