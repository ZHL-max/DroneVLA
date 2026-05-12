# UAV-Flow (2025)
## 北航出品的无人机VLA系统

---

## 论文信息

- **标题**：UAV-Flow: Instruction-Conditioned UAV Control
- **作者**：北京航空航天大学 CoLA Lab
- **会议**：NeurIPS 2025
- **代码**：https://github.com/buaa-colalab/UAV-Flow
- **数据**：HuggingFace (UAV-Flow, UAV-Flow-Sim)

---

## 核心贡献

UAV-Flow是首个面向无人机的大规模VLA系统，包含：

1. **数据集**：真实+仿真的无人机飞行轨迹
2. **模型**：基于OpenVLA微调的OpenVLA-UAV
3. **评估环境**：基于UnrealZoo的仿真评估

```
┌─────────────────────────────────────────────────┐
│                  UAV-Flow                        │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ UAV-Flow │  │OpenVLA-UAV│  │UAV-Flow  │     │
│  │ Dataset  │→│  Model   │→│  Eval    │     │
│  └──────────┘  └──────────┘  └──────────┘     │
│                                                 │
│  真实轨迹      OpenVLA微调     UnrealZoo仿真    │
│  + 仿真轨迹    + 无人机数据    + nDTW评估       │
└─────────────────────────────────────────────────┘
```

---

## 模型架构：OpenVLA-UAV

### 基础架构

OpenVLA-UAV基于OpenVLA 7B模型进行微调：

```
输入：
- 无人机相机图像 (224×224)
- 自然语言指令 ("fly to the red building")

架构：
┌─────────────────────────────────────────────────┐
│              OpenVLA-UAV (7B参数)               │
│                                                 │
│  ┌──────────┐  ┌──────────┐                    │
│  │ DINOv2   │  │  SigLIP  │  ← 视觉编码       │
│  │ (空间)   │  │  (语义)  │                    │
│  └────┬─────┘  └────┬─────┘                    │
│       └──────┬──────┘                          │
│              │                                  │
│       ┌──────┴──────┐                          │
│       │   Llama-2   │  ← 语言模型              │
│       │   (7B)      │                          │
│       └──────┬──────┘                          │
│              │                                  │
│       ┌──────┴──────┐                          │
│       │  动作头     │  ← 输出4维速度指令       │
│       └─────────────┘                          │
└─────────────────────────────────────────────────┘

输出：
action = [vx, vy, vz, yaw_rate]
```

### 微调策略

```python
# 微调配置
finetune_config = {
    'base_model': 'openvla-7b',           # 预训练模型
    'dataset': 'UAV-Flow',                # 无人机数据
    'learning_rate': 2e-5,                # 学习率
    'batch_size': 16,                     # 批大小
    'epochs': 10,                         # 训练轮数
    'lora_rank': 64,                      # LoRA秩
    'precision': 'bf16',                  # 混合精度
}

# LoRA微调（参数高效）
# 只训练约1%的参数，大幅减少计算需求
```

---

## 数据集

### UAV-Flow（真实数据）

```python
# 数据格式
{
    'instruction': 'fly to the red building',
    'trajectory': [
        {'image': img1, 'state': [x,y,z,vx,vy,vz], 'action': [vx,vy,vz,yaw]},
        {'image': img2, 'state': [x,y,z,vx,vy,vz], 'action': [vx,vy,vz,yaw]},
        ...
    ],
    'metadata': {
        'environment': 'outdoor',
        'weather': 'sunny',
        'duration': 30.0
    }
}
```

### UAV-Flow-Sim（仿真数据）

```python
# 使用UnrealZoo生成
# 优势：
# - 可以大规模生成
# - 环境可控
# - 安全无风险

# 仿真环境特点：
# - 高保真视觉渲染
# - 物理引擎模拟
# - 多种场景（城市、森林、室内）
```

---

## 评估方法：nDTW

### 什么是nDTW？

**nDTW (normalized Dynamic Time Warping)** 是衡量两条轨迹相似度的指标。

```
参考轨迹：  A → B → C → D
预测轨迹：  A → B' → C' → D

nDTW计算：
1. 找到两条轨迹的最佳对齐
2. 计算对齐后的平均距离
3. 归一化到[0, 1]范围

nDTW = 1.0：完美匹配
nDTW = 0.0：完全不匹配
```

### 评估流程

```python
def evaluate_trajectory(predicted, reference):
    """
    评估预测轨迹与参考轨迹的相似度

    Args:
        predicted: 预测轨迹 [(x1,y1,z1), (x2,y2,z2), ...]
        reference: 参考轨迹 [(x1,y1,z1), (x2,y2,z2), ...]

    Returns:
        ndtw: 归一化DTW分数
    """
    # 计算DTW距离
    dtw_distance = compute_dtw(predicted, reference)

    # 归一化
    trajectory_length = len(reference)
    ndtw = np.exp(-dtw_distance / trajectory_length)

    return ndtw
```

---

## 代码结构

```
UAV-Flow/
├── OpenVLA-UAV/              # 模型训练和推理
│   ├── finetune_uav.sh       # 微调脚本
│   ├── inference_server.py   # Flask推理服务器
│   └── configs/              # 配置文件
├── UAV-Flow-Eval/            # 评估环境
│   ├── unrealzoo_gym/        # UnrealZoo Gym接口
│   └── evaluation/           # 评估脚本
├── dataset_tools/            # 数据工具
│   ├── preprocess.py         # 数据预处理
│   └── visualization.py      # 数据可视化
└── README.md
```

---

## 训练流程

### 1. 环境配置

```bash
# Python 3.10 for OpenVLA-UAV
conda create -n openvla-uav python=3.10
conda activate openvla-uav

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.5.5
pip install -r requirements.txt
```

### 2. 下载数据

```python
from huggingface_hub import snapshot_download

# 下载UAV-Flow数据集
snapshot_download(
    repo_id="buaa-colalab/UAV-Flow",
    local_dir="./data/UAV-Flow"
)

# 下载预训练模型
snapshot_download(
    repo_id="buaa-colalab/OpenVLA-UAV",
    local_dir="./models/OpenVLA-UAV"
)
```

### 3. 微调模型

```bash
# 运行微调脚本
bash finetune_uav.sh \
    --pretrained_model ./models/OpenVLA-UAV \
    --dataset ./data/UAV-Flow \
    --output_dir ./checkpoints \
    --wandb_project uav-flow
```

### 4. 评估

```bash
# 在UnrealZoo中评估
python UAV-Flow-Eval/evaluation/evaluate.py \
    --model ./checkpoints/best_model \
    --dataset ./data/UAV-Flow/test \
    --metrics ndtw
```

---

## 与DroneVLA的对比

| 维度 | UAV-Flow | DroneVLA |
|------|----------|----------|
| **模型规模** | 7B参数 | 115M参数 |
| **基础模型** | OpenVLA | 自定义架构 |
| **训练方式** | LoRA微调 | 从头训练 |
| **数据规模** | 大规模 | 小规模合成 |
| **评估环境** | UnrealZoo | PyBullet |
| **适用场景** | 研究前沿 | 教学入门 |
| **计算需求** | 高（需要GPU集群） | 低（单GPU可训练） |

---

## 学习建议

### 作为北航学生

1. **直接复现**：UAV-Flow是你们学校的研究，可以直接联系作者
2. **数据集**：使用HuggingFace上的公开数据集
3. **改进方向**：
   - 更轻量的模型架构
   - 更高效的数据利用
   - 更好的sim-to-real迁移

### 学习路径

```
阶段1：理解UAV-Flow
├── 阅读论文
├── 理解OpenVLA架构
└── 运行示例代码

阶段2：复现实验
├── 下载数据集
├── 微调模型
└── 评估结果

阶段3：改进创新
├── 尝试不同的微调策略
├── 设计更好的数据增强
└── 探索新的应用场景
```

---

## 相关资源

- **论文**：NeurIPS 2025 proceedings
- **代码**：https://github.com/buaa-colalab/UAV-Flow
- **数据**：HuggingFace (buaa-colalab)
- **模型**：HuggingFace (OpenVLA-UAV)
- **评估**：UnrealZoo Gym

---

*上一章：[π0](07_Pi0_2025.md) | 下一章：[代码实现教学](../03_Code_Tutorials/)*
