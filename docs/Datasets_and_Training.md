# 数据集与训练指南
## DroneVLA 数据准备、训练与评估

---

## 目录

1. [数据集概述](#1-数据集概述)
2. [合成数据生成](#2-合成数据生成)
3. [公开数据集](#3-公开数据集)
4. [训练流程](#4-训练流程)
5. [评估与可视化](#5-评估与可视化)
6. [自定义数据集](#6-自定义数据集)

---

## 1. 数据集概述

### 1.1 VLA训练需要什么数据？

```
一条训练样本包含：
┌─────────────────────────────────────────────────┐
│ 图像序列：[img_t-3, img_t-2, img_t-1, img_t]    │
│ 状态向量：[x, y, z, vx, vy, vz, roll, pitch,    │
│           yaw, wx, wy, wz]                      │
│ 动作序列：[action_t, action_t+1, ..., action_t+7]│
│ 语言指令："fly to the red building"              │
└─────────────────────────────────────────────────┘
```

### 1.2 数据格式

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| images | [T, H, W, 3] | float32 | RGB图像序列 |
| states | [12] | float32 | 无人机状态 |
| actions | [8, 4] | float32 | 速度控制动作 |
| instruction | str | string | 自然语言指令 |

### 1.3 动作空间定义

```python
action = [vx, vy, vz, yaw_rate]
# vx: 前后速度 (m/s), 范围 [-2, 2]
# vy: 左右速度 (m/s), 范围 [-2, 2]
# vz: 垂直速度 (m/s), 范围 [-2, 2]
# yaw_rate: 偏航角速度 (rad/s), 范围 [-1, 1]
```

---

## 2. 合成数据生成

### 2.1 快速生成

```bash
# 生成500个训练样本（默认6种任务）
python scripts/generate_dataset.py --num_episodes 500 --output data/train

# 生成指定任务
python scripts/generate_dataset.py --num_episodes 1000 --output data/train --tasks navigate avoid hover

# 生成大量数据用于正式训练
python scripts/generate_dataset.py --num_episodes 5000 --output data/train_full
```

### 2.2 支持的任务类型

| 任务 | 说明 | 难度 |
|------|------|------|
| hover | 保持悬停 | 简单 |
| navigate | 导航到目标 | 中等 |
| follow | 跟踪移动目标 | 中等 |
| avoid | 避障导航 | 困难 |
| land | 降落 | 简单 |
| takeoff | 起飞 | 简单 |

### 2.3 生成的数据结构

```
data/train/
├── demonstrations.npz    # 所有episode数据
└── metadata.json         # 数据集元信息
```

metadata.json 示例：
```json
{
  "num_episodes": 500,
  "tasks": ["hover", "navigate", "follow", "avoid", "land", "takeoff"],
  "image_size": 64,
  "state_dim": 12,
  "action_dim": 4,
  "tasks_info": {
    "hover": {"count": 83, "description": "保持悬停"},
    "navigate": {"count": 83, "description": "导航到目标"},
    ...
  }
}
```

---

## 3. 公开数据集

### 3.1 Open X-Embodiment

**简介**：Google DeepMind 发布的大规模机器人学习数据集

**规模**：
- 22种不同机器人
- 21个研究机构
- 527种技能
- 超过100万条轨迹

**下载方式**：
```bash
# 使用TensorFlow Datasets
pip install tensorflow-datasets
python -c "import tfds; tfds.load('fractal_episode_stats')"

# 或使用gsutil直接下载
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/{dataset_name} ~/tensorflow_datasets/
```

**参考**：
- 论文：https://arxiv.org/abs/2310.08864
- GitHub：https://github.com/google-deepmind/open_x_embodiment

### 3.2 其他相关数据集

| 数据集 | 任务类型 | 规模 | 链接 |
|--------|----------|------|------|
| **Bridge V2** | 桌面操作 | 60K轨迹 | github.com/rail-berkeley/bridge_data_v2 |
| **DROID** | 机器人操作 | 350K轨迹 | github.com/droid-dataset/droid |
| **RoboSet** | 多任务操作 | 160K轨迹 | github.com/roboset/roboset |
| **RH20TH** | 家庭任务 | 200小时 | github.com/rh20th |

### 3.3 无人机专用数据集

| 数据集 | 任务类型 | 说明 |
|--------|----------|------|
| **AirSim** | 仿真飞行 | Microsoft 仿真器 |
| **FlightMare** | 仿真飞行 | ETH Zurich |
| **MAV dataset** | 真实飞行 | 多种飞行场景 |

---

## 4. 训练流程

### 4.1 完整训练流程

```bash
# 步骤1：生成数据集
python scripts/generate_dataset.py --num_episodes 1000 --output data/train

# 步骤2：训练模型
python scripts/train.py \
    --data data/train \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --save_dir logs \
    --use_world_model

# 步骤3：评估模型
python scripts/evaluate.py \
    --model logs/best_model.pt \
    --episodes 50 \
    --tasks navigate avoid
```

### 4.2 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --epochs | 100 | 训练轮数 |
| --batch_size | 32 | 批大小 |
| --lr | 1e-4 | 学习率 |
| --num_frames | 4 | 输入帧数 |
| --action_horizon | 8 | 动作序列长度 |
| --use_world_model | False | 启用世界模型 |
| --device | auto | 计算设备 |

### 4.3 训练监控

训练过程中会输出：
```
Epoch 10/100 | Train Loss: 0.0572 | Val Loss: 0.0613 | LR: 0.000095
Epoch 20/100 | Train Loss: 0.0016 | Val Loss: 0.0021 | LR: 0.000081
...
```

保存的文件：
```
logs/
├── best_model.pt      # 最佳模型
├── final_model.pt     # 最终模型
└── config.json        # 训练配置
```

---

## 5. 评估与可视化

### 5.1 评估指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **成功率** | 到达目标的episode比例 | >80% |
| **平均步数** | 完成任务的平均步数 | <50 |
| **平均奖励** | 累积奖励 | >-5 |

### 5.2 评估命令

```bash
# 评估所有任务
python scripts/evaluate.py --model logs/best_model.pt --episodes 100

# 评估特定任务
python scripts/evaluate.py --model logs/best_model.pt --episodes 50 --tasks navigate

# 使用GPU评估
python scripts/evaluate.py --model logs/best_model.pt --device cuda
```

### 5.3 结果可视化

评估结果保存在 `logs/evaluation_results.json`：
```json
{
  "navigate": {
    "success_rate": 65.0,
    "avg_steps": 32.5,
    "avg_reward": -2.34
  },
  "avoid": {
    "success_rate": 45.0,
    "avg_steps": 48.2,
    "avg_reward": -3.56
  }
}
```

---

## 6. 自定义数据集

### 6.1 数据收集接口

```python
from src.training.trainer import DemonstrationDataset

# 定义数据收集器
class MyDataCollector:
    def collect_episode(self):
        """收集一个episode的数据"""
        episode = {
            "images": [],      # 图像序列
            "states": [],      # 状态序列
            "actions": [],     # 动作序列
            "instruction": ""  # 语言指令
        }

        # ... 收集逻辑 ...

        return episode
```

### 6.2 数据格式转换

```python
import numpy as np

# 将自定义数据转换为标准格式
def convert_to_standard_format(data):
    episodes = []
    for episode in data:
        episodes.append({
            "images": np.array(episode["images"], dtype=np.float32),
            "states": np.array(episode["states"], dtype=np.float32),
            "actions": np.array(episode["actions"], dtype=np.float32),
            "instruction": episode["instruction"]
        })

    np.savez_compressed("data/custom/demonstrations.npz", episodes=episodes)
```

### 6.3 使用真实飞行数据

```python
# 从ROS bag文件提取数据
import rosbag

def extract_from_rosbag(bag_file):
    bag = rosbag.Bag(bag_file)
    episodes = []

    for topic, msg, t in bag.read_messages():
        if topic == "/camera/image_raw":
            # 提取图像
            image = msg_to_numpy(msg)
        elif topic == "/mavros/state":
            # 提取状态
            state = extract_state(msg)
        elif topic == "/mavros/setpoint_velocity":
            # 提取动作
            action = extract_action(msg)

    return episodes
```

---

## 常见问题

### Q: 训练loss不下降？
A: 检查学习率是否合适，尝试 `--lr 1e-3` 或 `--lr 1e-5`

### Q: 内存不足？
A: 减小批大小 `--batch_size 8` 或 `--batch_size 4`

### Q: 如何使用GPU训练？
A: 确保安装了CUDA版本的PyTorch，训练会自动使用GPU

### Q: 数据集太大怎么办？
A: 使用 `--num_episodes` 控制生成数量，或使用数据子集训练

---

*最后更新：2026-05-11*
