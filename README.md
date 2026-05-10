# DroneVLA

<div align="center">

**Vision-Language-Action Model for Drone Control**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 项目简介

DroneVLA是一个面向无人机的**视觉-语言-动作 (Vision-Language-Action)** 系统，使无人机能够：

- **理解自然语言指令**：如"飞到红色建筑物旁边"、"悬停在3米高度"
- **通过视觉感知环境**：处理机载相机图像流
- **自主执行飞行动作**：输出速度、航点等控制指令

本项目基于对 [World Model for Robot Learning Survey](https://arxiv.org/abs/2605.00080) 的深度调研，融合了VLA领域的最新进展。

---

## 核心特性

| 特性 | 描述 |
|------|------|
| **多模态融合** | 整合视觉、语言、状态信息 |
| **世界模型** | 预测未来状态，支持想象训练 |
| **扩散策略** | 生成平滑的动作序列 |
| **多任务支持** | 悬停、导航、跟踪、避障 |
| **仿真环境** | 基于PyBullet的物理仿真 |
| **模块化设计** | 易于扩展和定制 |

---

## 项目结构

```
DroneVLA/
├── README.md                    # 项目文档
├── LICENSE                      # MIT许可证
├── requirements.txt             # 依赖项
├── setup.py                     # 安装脚本
├── main.py                      # 主程序入口
│
├── docs/                        # 学习笔记和文档
│   ├── VLA_Learning_Notes.md    # VLA学习笔记（由浅入深）
│   ├── World_Model_Survey.md    # 世界模型综述总结
│   ├── Drone_VLA_Guide.md       # 无人机VLA指南
│   ├── VLA_Terminology_and_Principles.md  # VLA术语与原理解析
│   ├── Multi_Platform_Installation.md     # 多平台安装手册
│   ├── Hardware_Connection_Guide.md       # 硬件连接方案
│   └── Datasets_and_Training.md           # 数据集与训练指南
│
├── demos/                       # 经典小项目Demo
│   ├── 01_simple_vla/           # 最简单的VLA示例
│   ├── 02_clip_transport/       # CLIPort演示
│   ├── 03_diffusion_policy/     # 扩散策略演示
│   └── 04_world_model/          # 世界模型演示
│
├── src/                         # 核心源代码
│   ├── __init__.py
│   ├── models/                  # 模型定义
│   │   ├── __init__.py
│   │   └── drone_vla.py         # DroneVLA模型
│   ├── environments/            # 无人机环境
│   │   ├── __init__.py
│   │   └── drone_env.py         # PyBullet仿真环境
│   ├── camera/                  # 相机驱动
│   │   ├── __init__.py
│   │   ├── camera_base.py       # 相机基类
│   │   ├── realsense_camera.py  # Intel RealSense
│   │   ├── oakd_camera.py       # OAK-D相机
│   │   ├── usb_camera.py        # USB摄像头
│   │   └── picamera.py          # 树莓派相机
│   ├── training/                # 训练代码
│   │   ├── __init__.py
│   │   └── trainer.py           # 训练器
│   └── inference/               # 推理代码
│       └── __init__.py
│
├── configs/                     # 配置文件
│   └── default.yaml             # 默认配置
│
├── scripts/                     # 工具脚本
│   ├── generate_dataset.py      # 合成数据集生成
│   ├── train.py                 # 训练脚本
│   └── evaluate.py              # 评估脚本
│
├── tests/                       # 测试代码
│   └── run_tests.py             # 模型测试（全部通过）
│
├── data/                        # 数据目录
│   └── demos/                   # 演示数据
│
├── logs/                        # 训练日志
│
└── assets/                      # 资源文件
```

---

## 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/ZHL-max/DroneVLA.git
cd DroneVLA

# 创建虚拟环境
conda create -n dronevla python=3.10 -y
conda activate dronevla

# 安装依赖
pip install -e .
```

### 2. 收集演示数据

```bash
# 使用默认配置收集数据
python main.py --mode collect --task hover --num_episodes 100

# 使用自定义配置
python main.py --mode collect --config configs/default.yaml
```

### 3. 训练模型

```bash
# 训练DroneVLA模型
python main.py --mode train --config configs/default.yaml

# 指定数据目录
python main.py --mode train --data_dir data/demos
```

### 4. 评估模型

```bash
# 评估训练好的模型
python main.py --mode evaluate --model logs/best_model.pt
```

### 5. 运行演示

```bash
# 带渲染的演示
python main.py --mode demo --model logs/best_model.pt
```

---

## Demo项目

本项目包含4个由浅入深的Demo，帮助理解VLA的核心概念：

### Demo 01: 最简单的VLA
```bash
cd demos/01_simple_vla
python simple_vla.py
```
**学习要点**：视觉编码、语言编码、多模态融合、动作解码

### Demo 02: CLIPort
```bash
cd demos/02_clip_transport
python clip_transport.py
```
**学习要点**：CLIP语义理解、TransportNet空间精度、注意力融合

### Demo 03: 扩散策略
```bash
cd demos/03_diffusion_policy
python diffusion_policy.py
```
**学习要点**：扩散模型、条件去噪、动作序列生成

### Demo 04: 世界模型
```bash
cd demos/04_world_model
python world_model.py
```
**学习要点**：动态模型、奖励预测、想象训练

---

## 学习笔记

详细的学习笔记请查看 `docs/` 目录：

1. **[VLA学习笔记](docs/VLA_Learning_Notes.md)**：由浅入深的VLA知识体系
2. **[世界模型综述](docs/World_Model_Survey.md)**：World Model + VLA的全面总结
3. **[无人机VLA指南](docs/Drone_VLA_Guide.md)**：面向无人机的VLA实践指南
4. **[VLA术语与原理](docs/VLA_Terminology_and_Principles.md)**：所有VLA核心概念的通俗解释
5. **[多平台安装手册](docs/Multi_Platform_Installation.md)**：Windows/Linux/macOS/Jetson安装指南
6. **[硬件连接方案](docs/Hardware_Connection_Guide.md)**：相机、飞控、机载计算机选型与接线
7. **[数据集与训练](docs/Datasets_and_Training.md)**：数据生成、训练流程、评估方法

---

## 模型架构

```
┌─────────────────────────────────────────────────────────┐
│                      DroneVLA                            │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │  Visual   │  │ Language  │  │  State   │             │
│  │  Encoder  │  │  Encoder  │  │  Encoder │             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
│       │              │              │                   │
│       └──────────────┼──────────────┘                   │
│                      │                                  │
│              ┌───────┴───────┐                          │
│              │  Multimodal   │                          │
│              │    Fusion     │                          │
│              └───────┬───────┘                          │
│                      │                                  │
│       ┌──────────────┼──────────────┐                   │
│       │              │              │                   │
│       ▼              ▼              ▼                   │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Action  │  │  World   │  │  Value   │             │
│  │ Decoder │  │  Model   │  │  Head    │             │
│  └─────────┘  └──────────┘  └──────────┘             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 参考文献

### 核心论文

1. **RT-2**: Vision-Language-Action Models for Robotics (Google DeepMind)
2. **OpenVLA**: An Open-Source Vision-Language-Action Model
3. **Diffusion Policy**: Visuomotor Policy Learning via Action Diffusion
4. **World Model Survey**: World Model for Robot Learning (arXiv:2605.00080)

### 相关资源

- [Awesome-World-Model-for-Robotics-Policy](https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy) - 论文汇总
- [OpenVLA](https://github.com/openvla/openvla) - 开源VLA模型
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) - 扩散策略
- [Gym-PyBullet-Drones](https://github.com/utiasDSL/gym-pybullet-drones) - 无人机仿真

---

## 许可证

本项目采用 [MIT许可证](LICENSE)。

---

## 致谢

感谢以下项目和论文的启发：
- NTU、UC Berkeley、Stanford、Oxford的世界模型+VLA综述
- Google DeepMind的RT-2和OpenVLA项目
- Stanford的Diffusion Policy项目

---

<div align="center">

**如果这个项目对您有帮助，请给我们一个 Star！**

</div>
