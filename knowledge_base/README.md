# 无人机VLA学习知识库
## 北航无人机人工智能方向 - 系统学习指南

<div align="center">

**从零开始，系统掌握 Vision-Language-Action 全栈知识**

适用对象：北京航空航天大学 无人机人工智能方向学生

</div>

---

## 知识库结构

```
knowledge_base/
├── README.md                        # 本文件 - 知识库导航
├── 01_VLA_Foundations/              # 第一章：VLA基础概念
│   ├── 01_What_is_VLA.md           # 什么是VLA
│   ├── 02_Development_History.md   # 发展历史与脉络
│   └── 03_Key_Concepts.md          # 核心概念速查
├── 02_Model_Evolution/              # 第二章：模型发展详解
│   ├── 01_CLIPort_2021.md          # CLIPort
│   ├── 02_SayCan_2022.md           # SayCan
│   ├── 03_RT2_2023.md              # RT-2
│   ├── 04_OpenVLA_2024.md          # OpenVLA
│   ├── 05_Diffusion_Policy.md      # Diffusion Policy
│   ├── 06_World_Models.md          # 世界模型
│   ├── 07_Pi0_2025.md              # π0
│   └── 08_UAV_Flow_2025.md         # UAV-Flow (北航)
├── 03_Code_Tutorials/               # 第三章：代码实现教学
│   ├── 01_Environment_Setup.md     # 环境配置教程
│   ├── 02_Build_VLA_From_Scratch.md # 从零构建VLA
│   ├── 03_Train_Your_First_Model.md # 训练第一个模型
│   └── 04_Advanced_Training.md     # 高级训练技巧
├── 04_UAV_Specific/                 # 第四章：无人机VLA专题
│   ├── 01_Drone_vs_Manipulator.md  # 无人机vs机械臂
│   ├── 02_Drone_Control_Basics.md  # 无人机控制基础
│   ├── 03_MAVLink_Protocol.md      # MAVLink协议
│   └── 04_Sim_to_Real.md           # 仿真到现实迁移
├── 05_Experiment_Guide/             # 第五章：实验指南
│   ├── 01_Experiment_Design.md     # 实验设计指南
│   ├── 02_Data_Preparation.md      # 数据准备
│   ├── 03_Training_Pipeline.md     # 训练流程
│   └── 04_Evaluation_Methods.md    # 评估方法
├── 06_Paper_Reading/                # 第六章：论文阅读路线
│   ├── 01_RT2_Paper.md             # RT-2论文解读
│   ├── 02_OpenVLA_Paper.md         # OpenVLA论文解读
│   └── Paper_Notes/                # 更多论文笔记
├── 07_Research_Topics/              # 第七章：研究选题建议
│   └── Future_Directions.md        # 未来方向
└── 08_References/                   # 第八章：参考资料
    └── Resource_List.md            # 资源清单
```

---

## 学习路线建议

### 阶段一：基础入门（1-2周）

| 天数 | 内容 | 文档 |
|------|------|------|
| Day 1-2 | VLA是什么？核心概念 | `01_VLA_Foundations/` |
| Day 3-4 | 模型发展历史 | `02_Model_Evolution/01-03` |
| Day 5-7 | PyTorch基础 + 动手写代码 | `03_Code_Tutorials/` |

### 阶段二：深入理解（2-3周）

| 天数 | 内容 | 文档 |
|------|------|------|
| Day 8-10 | OpenVLA、Diffusion Policy | `02_Model_Evolution/04-05` |
| Day 11-14 | 世界模型、π0 | `02_Model_Evolution/06-07` |
| Day 15-17 | UAV-Flow详解 | `02_Model_Evolution/08` |
| Day 18-21 | 无人机专题 | `04_UAV_Specific/` |

### 阶段三：动手实验（2-3周）

| 天数 | 内容 | 文档 |
|------|------|------|
| Day 22-24 | 环境配置、数据准备 | `05_Experiment_Guide/01-02` |
| Day 25-28 | 训练自己的VLA模型 | `05_Experiment_Guide/03` |
| Day 29-31 | 评估与优化 | `05_Experiment_Guide/04` |

### 阶段四：研究探索（持续）

| 内容 | 文档 |
|------|------|
| 论文精读 | `06_Paper_Reading/` |
| 研究选题 | `07_Research_Topics/` |
| 复现与创新 | 结合DroneVLA项目实践 |

---

## 快速开始

```bash
# 1. 进入项目目录
cd D:/BH/github/DroneVLA

# 2. 激活环境
conda activate dronevla

# 3. 开始学习
# 阅读第一章
cat knowledge_base/01_VLA_Foundations/01_What_is_VLA.md

# 4. 运行Demo验证理解
python demos/01_simple_vla/simple_vla.py
```

---

*知识库持续更新中 | 最后更新：2026-05-13*
