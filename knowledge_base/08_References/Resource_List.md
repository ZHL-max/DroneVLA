# 参考资源清单
## VLA学习与研究的完整资源

---

## 1. 核心论文

### 1.1 VLA基础论文

| 论文 | 年份 | 关键贡献 | 链接 |
|------|------|----------|------|
| CLIPort | 2021 | CLIP+Transporter | arxiv:2109.12098 |
| SayCan | 2022 | 语言+ affordance | arxiv:2204.01691 |
| RT-1 | 2022 | Transformer策略 | arxiv:2212.06817 |
| RT-2 | 2023 | VLM→VLA | arxiv:2307.15818 |
| OpenVLA | 2024 | 开源VLA | arxiv:2406.09246 |

### 1.2 扩散策略论文

| 论文 | 年份 | 关键贡献 | 链接 |
|------|------|----------|------|
| Diffusion Policy | 2023 | 扩散动作生成 | arxiv:2303.04137 |
| 3D Diffusion Policy | 2024 | 3D扩散策略 | arxiv:2403.03954 |

### 1.3 世界模型论文

| 论文 | 年份 | 关键贡献 | 链接 |
|------|------|----------|------|
| DreamerV3 | 2023 | 世界模型+RL | arxiv:2301.04104 |
| π0 | 2025 | 通用机器人基础模型 | Physical Intelligence |

### 1.4 无人机VLA论文

| 论文 | 年份 | 关键贡献 | 链接 |
|------|------|----------|------|
| UAV-Flow | 2025 | 无人机VLA | NeurIPS 2025 |

---

## 2. 开源项目

### 2.1 VLA模型

| 项目 | Stars | 特点 | 链接 |
|------|-------|------|------|
| OpenVLA | 2K+ | 开源7B VLA | github.com/openvla/openvla |
| Diffusion Policy | 1K+ | 扩散策略 | github.com/real-stanford/diffusion_policy |
| UAV-Flow | 100+ | 无人机VLA | github.com/buaa-colalab/UAV-Flow |
| DroneVLA | - | 本项目 | github.com/ZHL-max/DroneVLA |

### 2.2 仿真环境

| 环境 | 特点 | 链接 |
|------|------|------|
| PyBullet | 轻量级物理仿真 | pybullet.org |
| Isaac Gym | GPU加速仿真 | developer.nvidia.com |
| AirSim | 无人机仿真 | github.com/microsoft/AirSim |
| Flightmare | 无人机仿真 | github.com/uzh-rpg/flightmare |
| UnrealZoo | 高保真仿真 | github.com/UnrealZoo |

### 2.3 数据集

| 数据集 | 规模 | 任务 | 链接 |
|--------|------|------|------|
| Open X-Embodiment | 100万+ | 通用机器人 | robotics-transformer-x.github.io |
| UAV-Flow | 10万+ | 无人机控制 | HuggingFace |
| RLDS | - | 数据格式 | github.com/kakaobrain/rlds |

---

## 3. 学习资源

### 3.1 在线课程

| 课程 | 学校 | 内容 | 链接 |
|------|------|------|------|
| CS231n | Stanford | CNN视觉 | cs231n.stanford.edu |
| CS224n | Stanford | NLP | cs224n.stanford.edu |
| CS285 | UC Berkeley | 深度RL | rail.eecs.berkeley.edu |
| 16-831 | CMU | 机器人学习 | - |

### 3.2 书籍

| 书籍 | 作者 | 内容 |
|------|------|------|
| Deep Learning | Goodfellow | 深度学习基础 |
| Probabilistic Robotics | Thrun | 机器人学 |
| Reinforcement Learning | Sutton | 强化学习 |
| Robot Learning | Peters | 机器人学习 |

### 3.3 博客与教程

| 资源 | 内容 | 链接 |
|------|------|------|
| Lilian Weng博客 | VLA综述 | lilianweng.github.io |
| HuggingFace博客 | 模型教程 | huggingface.co/blog |
| PyTorch教程 | 框架教程 | pytorch.org/tutorials |

---

## 4. 工具与框架

### 4.1 深度学习框架

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| PyTorch | 灵活，研究友好 | 研究原型 |
| JAX | 函数式，高效 | 大规模训练 |
| TensorFlow | 生产部署 | 工业应用 |

### 4.2 机器人框架

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| ROS | 机器人操作系统 | 真实机器人 |
| PyBullet | 轻量仿真 | 快速原型 |
| Isaac Gym | GPU加速 | 大规模训练 |

### 4.3 实验管理

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| W&B | 实验跟踪 | 团队协作 |
| MLflow | 模型管理 | 生产部署 |
| TensorBoard | 可视化 | 训练监控 |

---

## 5. 北航相关资源

### 5.1 实验室与团队

| 团队 | 研究方向 | 联系方式 |
|------|----------|----------|
| CoLA Lab | 无人机VLA | UAV-Flow作者 |
| 其他实验室 | 机器人学习 | - |

### 5.2 课程与项目

| 课程 | 内容 | 建议 |
|------|------|------|
| 机器学习 | 基础理论 | 必修 |
| 计算机视觉 | 视觉处理 | 推荐 |
| 自然语言处理 | 语言理解 | 推荐 |
| 机器人学 | 控制基础 | 必修 |

### 5.3 竞赛与会议

| 活动 | 时间 | 建议 |
|------|------|------|
| ICRA | 每年5月 | 投稿 |
| IROS | 每年10月 | 投稿 |
| NeurIPS | 每年12月 | 投稿 |
| 无人机竞赛 | 不定期 | 参加 |

---

## 6. 常用代码片段

### 6.1 PyTorch基础

```python
# 模型定义
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 训练循环
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(100):
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 6.2 数据处理

```python
# 加载数据
dataset = MyDataset(data_dir)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 数据增强
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
```

### 6.3 模型部署

```python
# ONNX导出
torch.onnx.export(model, dummy_input, "model.onnx")

# ONNX推理
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input": input_data})
```

---

## 7. 常见问题解答

### Q1: 如何入门VLA？
A: 阅读本知识库第一章，运行Demo代码，理解基本概念。

### Q2: 需要什么硬件？
A: GPU推荐RTX 3060 12GB以上，CPU也可以跑轻量模型。

### Q3: 如何选择研究方向？
A: 阅读最新论文，了解领域趋势，结合自己的兴趣和能力。

### Q4: 如何获取数据？
A: 使用公开数据集（UAV-Flow、Open X-Embodiment）或自己生成仿真数据。

### Q5: 如何发表论文？
A: 做出创新工作，写好论文，投稿到相关会议（ICRA、IROS、NeurIPS）。

---

## 8. 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-05-13 | 初始版本，包含核心论文、开源项目、学习资源 |

---

*返回：[知识库首页](../README.md)*
