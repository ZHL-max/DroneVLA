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
│   ├── train_gpu.py             # GPU训练脚本
│   ├── run_iterations.py        # 多轮迭代训练
│   ├── evaluate.py              # 评估脚本
│   ├── evaluate_gpu.py          # GPU评估脚本
│   └── plot_results.py          # 结果可视化
│
├── experiments/                 # 实验结果
│   ├── iteration_01/ ~ iteration_10/  # 10轮迭代优化
│   └── comparison.png           # 迭代对比图
│
├── knowledge_base/              # VLA学习知识库
│   ├── 01_VLA_Foundations/      # VLA基础概念
│   ├── 02_Model_Evolution/      # 模型发展详解
│   ├── 03_Code_Tutorials/       # 代码实现教学
│   ├── 04_UAV_Specific/         # 无人机VLA专题
│   ├── 05_Experiment_Guide/     # 实验指南
│   ├── 06_Paper_Reading/        # 论文阅读路线
│   ├── 07_Research_Topics/      # 研究选题建议
│   └── 08_References/           # 参考资料
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

## VLA学习知识库

专为北航无人机人工智能方向学生设计的系统学习资源，详见 `knowledge_base/` 目录：

### 学习路线

| 阶段 | 时间 | 内容 |
|------|------|------|
| **基础入门** | 1-2周 | VLA概念、模型发展历史、PyTorch基础 |
| **深入理解** | 2-3周 | OpenVLA、Diffusion Policy、UAV-Flow |
| **动手实验** | 2-3周 | 环境配置、训练自己的VLA模型 |
| **研究探索** | 持续 | 论文精读、研究选题、复现创新 |

### 知识库内容

1. **VLA基础**：什么是VLA、发展历史、核心概念
2. **模型发展**：CLIPort → SayCan → RT-2 → OpenVLA → Diffusion Policy → UAV-Flow
3. **代码教程**：环境配置、从零构建VLA、训练技巧
4. **无人机专题**：无人机vs机械臂、控制基础、传感器融合
5. **实验指南**：实验设计、数据准备、训练流程、评估方法
6. **论文阅读**：RT-2、OpenVLA等核心论文解读
7. **研究方向**：轻量化VLA、Sim-to-Real、多机协同
8. **参考资料**：论文、开源项目、学习资源

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
8. **[真实部署指南](docs/Real_World_Deployment.md)**：从仿真到实飞的完整流程
9. **[API参考手册](docs/API_Reference.md)**：核心模块与接口文档
10. **[故障排除](docs/Troubleshooting_FAQ.md)**：常见问题与解决方案
11. **[项目路线图](docs/Project_Roadmap.md)**：开发计划与里程碑

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

## 训练结果

### 10轮迭代优化结果

通过系统化的10轮迭代优化，DroneVLA模型在4个核心任务上取得了显著提升：

| 迭代 | 策略 | 整体成功率 | 导航 | 避障 | 悬停 | 降落 |
|------|------|-----------|------|------|------|------|
| 01 | 基线模型 | 47.5% | 0% | 0% | 100% | 90% |
| 02 | 增加数据量 | 58.3% | 23.3% | 10% | 100% | 100% |
| 03 | 数据增强 | 59.2% | 20% | 16.7% | 100% | 100% |
| 04 | 注意力机制 | 62.5% | 23.3% | 26.7% | 100% | 100% |
| 05 | 增大模型 | 65.8% | 23.3% | 40% | 100% | 100% |
| 06 | 学习率调优 | 59.2% | 23.3% | 13.3% | 100% | 100% |
| 07 | 聚焦弱任务 | 69.2% | 43.3% | 33.3% | 100% | 100% |
| 08 | 加深解码器 | 68.3% | 50% | 23.3% | 100% | 100% |
| 09 | 课程学习 | 69.2% | 40% | 36.7% | 100% | 100% |
| **10** | **最终优化** | **72.5%** | **43.3%** | **46.7%** | **100%** | **100%** |

**关键改进**：
- 悬停和降落任务：从初始就接近完美
- 导航任务：从0%提升到43.3%
- 避障任务：从0%提升到46.7%
- 整体成功率：从47.5%提升到72.5%（+52.6%）

运行训练：
```bash
# 运行所有10轮迭代
python scripts/run_iterations.py

# 运行指定迭代范围
python scripts/run_iterations.py --start 5 --end 10
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
