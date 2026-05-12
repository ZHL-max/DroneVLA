# UAV-Flow 深度研究报告

> NeurIPS 2025 论文 | 北京航空航天大学 (BUAA)
> 项目地址: https://github.com/buaa-colalab/UAV-Flow
> 论文标题: *UAV-Flow Colosseo: A Real-World Benchmark for Flying-on-a-Word UAV Imitation Learning*

---

## 一、项目概述

### 1.1 研究背景与动机

无人机 (UAV) 智能控制是当前机器人学习领域的热点问题。传统的无人机控制方法依赖于手工设计的控制器或基于规则的系统，难以处理复杂的自然语言指令。UAV-Flow 项目的核心思想是: **将视觉-语言-动作 (Vision-Language-Action, VLA) 模型引入无人机控制领域**，使无人机能够根据自然语言指令和当前视觉观察自主生成飞行轨迹。

### 1.2 核心贡献

UAV-Flow 项目提供了三大组件:

| 组件 | 说明 |
|------|------|
| **UAV-Flow 数据集** | 包含真实世界 (UAV-Flow) 和仿真环境 (UAV-Flow-Sim) 的无人机轨迹数据 |
| **UAV-Flow-Eval** | 基于 UnrealZoo Gym 的仿真评估环境，使用 nDTW 指标进行轨迹质量评估 |
| **OpenVLA-UAV** | 基于 OpenVLA 7B 模型微调的无人机控制基线模型 |

### 1.3 技术路线总结

```
自然语言指令 + 当前视觉图像 + 本体感知状态
        |
        v
   OpenVLA-UAV 模型 (7B 参数)
        |
        v
   离散化动作 Token 序列
        |
        v
   连续动作 (dx, dy, dz, dyaw) -- 机体坐标系下的相对位移
        |
        v
   无人机执行动作
```

---

## 二、核心模型架构详解

### 2.1 OpenVLA 基础架构

OpenVLA-UAV 基于 [OpenVLA](https://github.com/openvla/openvla) 7B 模型。OpenVLA 是一个 **视觉-语言-动作 (VLA)** 模型，其架构可以分解为三个核心模块:

#### 2.1.1 视觉骨干网络 (Vision Backbone)

```
输入图像 (224x224 RGB)
    |
    v
TIMM 视觉编码器 (SigLIP-400M / CLIP-400M)
    |
    v
图像 Patch 特征 (N_patches x embed_dim)
```

- 代码位置: `prismatic/extern/hf/modeling_prismatic.py` -> `PrismaticVisionBackbone`
- 使用 TIMM 库创建视觉特征提取器
- 支持**融合双骨干** (Fused Backbone): 可以将两个不同分辨率/架构的视觉编码器的特征拼接
- 默认取倒数第二层的 patch 特征作为输出
- 嵌入维度: 取决于所用的 ViT 模型 (如 SigLIP 为 1152)

#### 2.1.2 多模态投影器 (Multimodal Projector)

```
视觉特征 (vision_dim)
    |
    v
Linear -> GELU -> Linear (LLM 输入维度)
    |
    v
投影后的视觉 Token
```

- 代码位置: `prismatic/extern/hf/modeling_prismatic.py` -> `PrismaticProjector`
- 作用: 将视觉特征从视觉编码器的维度空间映射到 LLM 的嵌入空间
- 非融合骨干: 两层 MLP (fc1 -> GELU -> fc2)
- 融合骨干: 三层 MLP (fc1 -> GELU -> fc2 -> GELU -> fc3)

#### 2.1.3 语言模型骨干 (LLM Backbone)

```
[视觉 Token] + [文本 Token] -> LLaMA-2 7B -> 下一个 Token 预测
```

- 基础模型: LLaMA-2 7B (通过 HuggingFace `AutoModelForCausalLM` 加载)
- 文本输入格式: `"In: Current State: {proprio_str}, What action should the uav take to {instruction}?\nOut:"`
- 输出: 预测的**动作 Token 序列** (离散化的连续动作值)

### 2.2 动作离散化机制 (Action Tokenizer)

这是理解 UAV-Flow 动作预测的关键。OpenVLA 不直接回归连续动作值，而是将连续动作**离散化为 Token**:

- 代码位置: `prismatic/vla/action_tokenizer.py` -> `ActionTokenizer`
- 分辨率: **256 bins**，覆盖范围 [-1, 1]
- 映射方式: 将连续动作均匀分箱后，映射到词表中**最后 256 个 Token**

```
连续动作 [-1, 1]
    |
    v
np.digitize(action, 256 个均匀分箱)
    |
    v
离散化索引 [1, 256]
    |
    v
词表中的 Token ID (vocab_size - 索引)
    |
    v
LLM 输出这些 Token -> 解码回连续值
```

关键实现细节:
- `action_token_begin_idx = tokenizer.vocab_size - (256 + 1)`
- 解码时使用 **bin 中心值** (bin_centers) 作为连续动作的估计
- 最终输出 4 个维度: `[dx, dy, dz, dyaw]`

### 2.3 UAV 特有的设计

相比原始 OpenVLA (用于机器人操作)，UAV-Flow 进行了以下关键修改:

1. **动作空间**: 从 7-DoF 机器人手臂改为 4-DoF 无人机控制 `[dx, dy, dz, dyaw]`
2. **本体感知输入**: 使用无人机的当前位姿 `[x, y, z, yaw]` 作为 proprioceptive state
3. **动作表示**: 采用**机体坐标系下的相对位移** (local frame)，而非世界坐标系
4. **坐标变换**: 推理时需要将机体坐标系的预测动作转换回世界坐标系执行

---

## 三、数据集格式与来源

### 3.1 数据集概览

| 数据集 | 存储位置 | 类型 | 说明 |
|--------|----------|------|------|
| UAV-Flow | HuggingFace: `wangxiangyu0814/UAV-Flow` | 真实世界 | 真实无人机飞行轨迹 |
| UAV-Flow-Sim | HuggingFace: `wangxiangyu0814/UAV-Flow-Sim` | 仿真 | 仿真环境中的飞行轨迹 |

### 3.2 原始数据格式 (Parquet)

数据以 Parquet 文件存储，每一行包含:

```python
{
    "id": "trajectory_id",      # 轨迹标识符
    "frame_idx": int,            # 帧序号
    "log": json_string,          # 包含轨迹信息的 JSON 字符串
    "image": PIL.Image           # 对应帧的 RGB 图像
}
```

其中 `log` 字段解析后包含:

```python
{
    "raw_logs": [[x, y, z, roll, yaw, pitch], ...],          # 原始 6-DoF 位姿序列
    "preprocessed_logs": [[x, y, z, roll, yaw, pitch], ...], # 预处理后的位姿序列
    "instruction": "Fly forward 5 meters",                     # 自然语言指令
    "instruction_unified": "Move forward 5 meters"            # 统一格式的指令
}
```

### 3.3 训练数据格式 (文件夹)

通过 `dataset_tools/prepare_data.py` 转换后，每条轨迹成为一个独立文件夹:

```
<trajectory_id>/
    000000.jpg       # 第 0 帧图像
    000001.jpg       # 第 1 帧图像
    ...
    log.json         # 轨迹元数据
```

`log.json` 结构:

```python
{
    "id": "trajectory_id",
    "raw_logs": [[x,y,z,roll,yaw,pitch], ...],       # 原始世界坐标位姿
    "preprocessed_logs": [[x,y,z,roll,yaw,pitch], ...], # 相对于起始帧的位姿
    "instruction": "原始指令",
    "instruction_unified": "统一指令",
    "length": 20                                       # 帧数
}
```

### 3.4 测试任务格式 (Evaluation JSON)

评估用的每个任务 JSON 文件结构:

```python
{
    "instruction": "Turn to the direction of the person.",
    "instruction_unified": "Turn to the direction of the person.",
    "initial_pos": [-608.855, -1270.567, 128.141, 0.0, 61.39, 0.0],  # [x,y,z,roll,yaw,pitch]
    "end_pos": [...],                                   # 预期终止位姿
    "obj_id": 19,                                       # 场景中的目标对象 ID
    "use_obj": 1,                                       # 是否使用对象
    "target_pos": [-202.88, -1019.80, 128.141, 0, 0, 0], # 目标位置
    "reference_path_raw": [[x,y,z,roll,yaw,pitch], ...], # 原始参考轨迹 (GT)
    "reference_path_preprocessed": [[x,y,z,roll,yaw,pitch], ...]  # 预处理后的参考轨迹 (GT)
}
```

### 3.5 动作计算方式

在 `uav_dataset.py` 中，训练数据的**动作**定义为:

```python
# 当前帧到下一帧的机体坐标系相对位移
action[i] = transform_to_local_frame(pose[i], pose[i+1])
# 输出: [dx_local, dy_local, dz_local, dyaw_relative]
```

其中 `transform_to_local_frame` 的核心逻辑:

1. 计算世界坐标系下的相对位移: `relative_pos = next_pos - current_pos`
2. 用当前 yaw 角构建旋转矩阵 R
3. 旋转到机体坐标系: `local_pos = R^(-1) @ relative_pos`
4. 计算相对偏航角: `relative_yaw = next_yaw - current_yaw` (归一化到 [-pi, pi])

### 3.6 评估任务分类

评估任务按动作类型分为 **11 类** (见 `classified_instr.json`):

| 类别 | 说明 | 特殊处理 |
|------|------|----------|
| Turn | 原地转向 | 忽略位置，只看朝向 |
| Move | 直线移动 | 采样步长 2 |
| Shift | 侧向平移 | 默认采样步长 |
| Rotate | 原地旋转 | 忽略位置，只看朝向 |
| Surround | 环绕目标 | 默认采样步长 |
| Ascend/Descend | 上升/下降 | 默认采样步长 |
| Approach | 接近目标 | 默认采样步长 |
| Retreat | 远离目标 | 默认采样步长 |
| Pass | 经过目标 | 默认采样步长 |
| Land | 降落 | 默认采样步长 |

---

## 四、训练流程

### 4.1 环境配置

```bash
conda create -n openvla python=3.10 -y
conda activate openvla
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

关键依赖版本:
- PyTorch: 2.2.0
- Transformers: 4.40.1
- Tokenizers: 0.19.1
- TIMM: 0.9.10
- PEFT: 0.11.1

### 4.2 训练脚本

训练入口: `OpenVLA-UAV/vla-scripts/finetune_uav.sh`

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune_uav.py \
  --vla_path /path/to/pretrained_openvla_model \
  --data_root_dir /path/to/dataset \
  --run_root_dir /path/to/run_name \
  --adapter_tmp_dir /path/to/run_name/adapter-tmp \
  --lora_rank 32 \
  --batch_size 4 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project openvla \
  --wandb_entity your-wandb-username \
  --save_steps 20000
```

### 4.3 训练策略详解

#### 4.3.1 LoRA 微调

```python
lora_config = LoraConfig(
    r=32,                              # LoRA 秩
    lora_alpha=min(32, 16),            # alpha = 16
    lora_dropout=0.0,                  # 无 dropout
    target_modules="all-linear",       # 对所有线性层应用 LoRA
    init_lora_weights="gaussian",      # 高斯初始化
)
```

- 使用 LoRA 而非全参数微调，大幅降低显存需求
- `target_modules="all-linear"`: 对模型中所有线性层 (包括 attention 和 FFN) 都应用 LoRA
- 训练结束后会**合并 LoRA 权重**到基础模型中保存

#### 4.3.2 数据加载

- 使用 `IterableDataset` 模式流式加载
- 数据集初始化时**预计算所有动作的统计量** (mean, std, 1st/99th percentile)
- 动作归一化到 [-1, 1]: `normalized = 2 * (action - min) / (max - min) - 1`
- **首尾帧重复 5 次**: 对轨迹的第一帧和最后一帧进行过采样 (last_frame_repeat_count=5)
- 每次迭代前随机打乱样本顺序

#### 4.3.3 Prompt 格式

训练时的 prompt 格式:

```
In: Current State: {x},{y},{z},{yaw}, What action should the uav take to {instruction}?\nOut: {action_tokens}
```

- proprio (本体感知状态) 为 4 维: `[x, y, z, yaw]`
- 位置使用**相对于起始帧的坐标** (preprocessed_logs)
- yaw 使用**角度值**
- 只有动作 Token 部分计算 loss (labels 中 prompt 部分设为 IGNORE_INDEX=-100)

#### 4.3.4 训练监控指标

训练过程中监控三个指标:
1. **train_loss**: 标准交叉熵损失
2. **action_accuracy**: 动作 Token 预测准确率
3. **l1_loss**: 解码回连续动作后的 L1 距离

#### 4.3.5 Norm Stats 更新

训练时会将数据集的统计信息写入模型的 `norm_stats` 字段:

```python
norm_stats["sim"] = {
    "action": {
        "mean": [...],   # 4 维
        "std": [...],
        "min": [...],    # 1st percentile
        "max": [...],    # 99th percentile
    }
}
```

推理时使用 `unnorm_key="sim"` 来查找对应的反归一化参数。

---

## 五、评估方法

### 5.1 评估流程概览

```
1. 启动 OpenVLA-UAV 推理服务器 (Flask, 端口 5007)
2. 启动 UnrealZoo Gym 仿真环境 (DowntownWest 场景)
3. 对每个测试任务:
   a. 设置初始位姿和场景物体
   b. 循环执行:
      - 获取当前图像 (256x256 -> 224x224)
      - 获取当前 proprio state
      - 发送 HTTP POST 到推理服务器
      - 接收预测的相对位姿
      - 坐标变换到世界坐标系
      - 在仿真器中执行
      - 记录轨迹
   c. 绘制 2D/3D 轨迹图
4. 使用 nDTW 指标评估轨迹质量
```

### 5.2 推理服务器

文件: `OpenVLA-UAV/vla-scripts/openvla_act.py`

核心流程:

```python
# 1. 接收请求: 图像 + 本体感知状态 + 指令
# 2. 构建 prompt
prompt = f"In: Current State: {proprio_str}, What action should the uav take to {instruction}?\nOut:"

# 3. 模型推理
pred_action = model.predict_action(**inputs, unnorm_key="sim", do_sample=False)
# pred_action: [1, 4] -> [dx, dy, dz, dyaw] (机体坐标系)

# 4. 坐标变换: 机体坐标系 -> 世界坐标系
R = rotation_matrix(current_yaw)
pred_action[0, 0:3] = R @ pred_action[0, 0:3]        # 旋转位移到世界坐标系
pred_action[0, 0:3] = current_pos + pred_action[0, 0:3]  # 加上当前位置
pred_action[0, -1] = pred_action[0, -1] + current_yaw    # 加上当前偏航角
```

### 5.3 仿真控制循环

文件: `UAV-Flow-Eval/batch_run_act_all.py`

关键设计:
- **环境**: UnrealZoo Gym 的 `UnrealTrack-DowntownWest-ContinuousColor-v0`
- **分辨率**: 256x256 (发送前缩放到 224x224)
- **时间膨胀**: TimeDilation=10 (保持仿真 FPS 稳定)
- **早停机制**: 连续 10 步位移变化小于 3.0 且 yaw 变化小于 1.0 度时自动终止
- **最大步数**: 默认 100 步
- **动作执行**: 接收的 action 为**单步相对位姿**，变换到世界坐标后直接设置物体位置 (非物理仿真)

### 5.4 nDTW 指标

文件: `UAV-Flow-Eval/metric.py`

nDTW (normalized Dynamic Time Warping) 是评估轨迹相似度的核心指标:

#### 5.4.1 DTW 计算

```python
def dtw_distance(vecs1, vecs2):
    # 构建欧氏距离矩阵
    dist_matrix = cdist(vecs1, vecs2, metric='euclidean')
    # 动态规划求解最优对齐路径
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dist_matrix[i-1, j-1]
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i-1, j-1], dtw[i, j-1])
    return dtw[n, m]
```

#### 5.4.2 nDTW 归一化

```python
nDTW = exp(-DTW_distance / (eta * L_gt))
```

- `L_gt`: 参考轨迹的总长度
- `eta`: 归一化超参数 (默认 1)
- nDTW 取值范围: [0, 1]，越接近 1 表示轨迹越接近参考

#### 5.4.3 6D 状态表示

评估使用的 6D 状态向量: `[x/100, y/100, z/100, cos(roll), cos(yaw), cos(pitch)]`

- 位置除以 100 进行缩放
- 旋转角取余弦值 (避免角度不连续问题)
- 对于 Turn/Rotate 类任务，**位置设为零**，只评估朝向

#### 5.4.4 分类评估

评估按任务类别分别计算 nDTW，最终输出:
- 每个类别的 Mean nDTW
- 整体的 Overall Mean nDTW
- Turn/Rotate: step=2, zero_pos=True
- Move: step=2
- 其他类别: step=5

---

## 六、与 DroneVLA 的对比

| 维度 | UAV-Flow (OpenVLA-UAV) | DroneVLA |
|------|------------------------|----------|
| **基础模型** | OpenVLA 7B (LLaMA-2 7B + SigLIP) | 待分析 |
| **参数量** | 7B (LoRA 微调，仅训练少量参数) | 待分析 |
| **动作空间** | 4-DoF: [dx, dy, dz, dyaw]，相对位移 | 待分析 |
| **动作表示** | 离散化为 256 bins -> 词表 Token | 待分析 |
| **输入** | RGB 图像 (224x224) + proprio (x,y,z,yaw) | 待分析 |
| **坐标系** | 机体坐标系下的相对动作 | 待分析 |
| **训练方式** | LoRA 微调 (rank=32, all-linear) | 待分析 |
| **推理架构** | Flask HTTP 服务器 | 待分析 |
| **评估环境** | UnrealZoo Gym (DowntownWest) | 待分析 |
| **评估指标** | nDTW (normalized Dynamic Time Warping) | 待分析 |
| **数据来源** | 真实世界 + 仿真 | 待分析 |

---

## 七、学习建议 (面向 BUAA 学生)

### 7.1 入门路径

```
阶段 1: 理解基础概念
├── 学习 VLA (Vision-Language-Action) 的基本概念
├── 理解 OpenVLA 论文和代码架构
├── 掌握 LoRA 微调原理
└── 理解动作离散化 (Action Tokenization) 机制

阶段 2: 深入 UAV-Flow 代码
├── 阅读 uav_dataset.py 理解数据处理流程
├── 阅读 finetune_uav.py 理解训练循环
├── 阅读 openvla_act.py 理解推理流程
└── 阅读 metric.py 理解评估指标

阶段 3: 实践复现
├── 下载 UAV-Flow-Sim 数据集
├── 配置训练环境 (conda + PyTorch + CUDA)
├── 使用 prepare_data.py 转换数据格式
├── 尝试小规模微调实验
└── 配置 UnrealZoo 仿真环境进行评估
```

### 7.2 关键知识点

#### 7.2.1 坐标系变换 (最重要)

理解机体坐标系和世界坐标系的变换是理解整个系统的基础:

```python
# 机体坐标系 -> 世界坐标系
R = [[cos(yaw), -sin(yaw), 0],
     [sin(yaw),  cos(yaw), 0],
     [0,         0,        1]]
world_pos = R @ local_pos + current_pos
```

#### 7.2.2 动作离散化 vs 回归

OpenVLA 选择离散化动作而非直接回归:
- **优点**: 复用语言模型的 Token 预测能力，训练更稳定
- **缺点**: 256 bins 的分辨率可能限制控制精度
- **替代方案**: DDPM (扩散模型) 直接回归连续动作，如 Octo、Diffusion Policy

#### 7.2.3 LoRA 微调策略

- 对 7B 参数的模型，LoRA 是必要的 (显存限制)
- `target_modules="all-linear"` 意味着对所有线性层都加 LoRA，参数量仍然可观
- `lora_alpha = min(rank, 16)` 控制 LoRA 更新的缩放

#### 7.2.4 数据增强

UAV-Flow 当前未启用图像增强 (`image_aug=False`)。这可能是因为:
- 飞行场景中增强可能改变语义 (如翻转改变左右)
- 数据量可能已经足够

### 7.3 常见问题与调试

1. **推理速度**: 模型 7B 参数，单次推理约需数百毫秒，需注意控制频率
2. **坐标系混淆**: 最容易出错的地方，务必画图验证坐标变换
3. **yaw 角度范围**: 注意角度归一化 ([-180, 180] vs [0, 360])
4. **动作尺度**: 训练前务必检查动作统计量，确保归一化正确
5. **评估环境**: UnrealZoo 仅支持 Windows，需要较大显存

### 7.4 进阶研究方向

1. **实时性改进**: 当前模型推理延迟较高，可探索模型量化/蒸馏
2. **动作空间扩展**: 增加速度控制、加速度约束等
3. **闭环鲁棒性**: 当前评估为开环设置，可探索闭环扰动下的鲁棒性
4. **多模态融合**: 结合激光雷达、IMU 等额外传感器
5. **Sim-to-Real**: 如何将仿真训练的模型迁移到真实世界
6. **更强基线**: 对比 Pi-0-UAV (项目提到 Coming Soon)

### 7.5 必读论文列表

1. **OpenVLA**: *OpenVLA: An Open-Source Vision-Language-Action Model* (Kim et al., 2024)
2. **Prismatic VLMs**: *Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models* (Karamcheti et al., 2024)
3. **RT-2**: *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control* (Brohan et al., 2023)
4. **LoRA**: *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)
5. **DTW**: *Dynamic Time Warping* (经典算法)
6. **UnrealZoo**: *UnrealZoo: Enriching Photo-realistic Virtual Worlds for Embodied AI* (相关项目)

---

## 八、代码结构速查

```
UAV-Flow/
├── README.md                         # 项目总文档
├── LICENSE                           # Apache-2.0 许可
├── dataset_tools/
│   ├── prepare_data.py               # Parquet -> 文件夹格式转换
│   └── README.md
├── OpenVLA-UAV/                      # 模型训练/推理代码
│   ├── pyproject.toml                # 依赖配置
│   ├── vla-scripts/
│   │   ├── finetune_uav.sh           # 训练启动脚本
│   │   ├── finetune_uav.py           # 训练主逻辑
│   │   └── openvla_act.py            # 推理服务器
│   └── prismatic/                    # 核心模型代码
│       ├── vla/
│       │   ├── action_tokenizer.py   # 动作离散化
│       │   └── datasets/
│       │       └── uav_dataset.py    # UAV 数据集类
│       ├── extern/hf/
│       │   ├── modeling_prismatic.py # 模型架构定义
│       │   ├── configuration_prismatic.py
│       │   └── processing_prismatic.py
│       └── models/backbones/         # 视觉/语言骨干
└── UAV-Flow-Eval/                    # 评估环境
    ├── batch_run_act_all.py          # 批量评估主脚本
    ├── metric.py                     # nDTW 指标计算
    ├── classified_instr.json         # 任务分类定义
    ├── test_jsons/                   # 测试任务 JSON
    └── gym_unrealcv/                 # UnrealZoo Gym 封装
        └── envs/setting/Track/       # 环境配置
            └── DowntownWest.json     # 主测试场景配置
```

---

*本报告基于 UAV-Flow 项目代码的深度分析编写。如有疑问，建议结合源代码和论文原文进行学习。*
