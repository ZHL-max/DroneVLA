# VLA核心概念
## 理解VLA的关键技术要素

---

## 1. 多模态融合 (Multimodal Fusion)

VLA的核心是将视觉、语言、状态三种模态的信息融合在一起。

### 融合策略

```
策略1：早期融合 (Early Fusion)
┌──────┐  ┌──────┐  ┌──────┐
│ 视觉 │  │ 语言 │  │ 状态 │
└──┬───┘  └──┬───┘  └──┬───┘
   └─────────┼─────────┘
        ┌────┴────┐
        │ 拼接融合 │
        └────┬────┘
        ┌────┴────┐
        │ 动作输出 │
        └─────────┘

策略2：晚期融合 (Late Fusion)
┌──────┐  ┌──────┐  ┌──────┐
│ 视觉 │  │ 语言 │  │ 状态 │
└──┬───┘  └──┬───┘  └──┬───┘
   ↓         ↓         ↓
┌──────┐  ┌──────┐  ┌──────┐
│编码器│  │编码器│  │编码器│
└──┬───┘  └──┬───┘  └──┬───┘
   └─────────┼─────────┘
        ┌────┴────┐
        │ 注意力  │
        └────┬────┘
        ┌────┴────┐
        │ 动作输出 │
        └─────────┘
```

### 代码实现

```python
# 早期融合
class EarlyFusion(nn.Module):
    def forward(self, visual, language, state):
        combined = torch.cat([visual, language, state], dim=-1)
        return self.decoder(combined)

# 晚期融合（带注意力）
class LateFusion(nn.Module):
    def forward(self, visual, language, state):
        # 各自编码
        v = self.visual_encoder(visual)
        l = self.language_encoder(language)
        s = self.state_encoder(state)

        # 交叉注意力
        v_l = self.cross_attention(v, l)  # 视觉关注语言
        l_v = self.cross_attention(l, v)  # 语言关注视觉

        # 最终融合
        combined = v_l + l_v + s
        return self.decoder(combined)
```

---

## 2. 视觉编码 (Visual Encoding)

### 常用视觉编码器

| 编码器 | 特点 | 适用场景 |
|--------|------|----------|
| **ResNet** | 经典CNN，稳定可靠 | 通用视觉任务 |
| **EfficientNet** | 高效，参数少 | 资源受限场景 |
| **ViT** | Transformer，全局视野 | 复杂场景 |
| **DINOv2** | 自监督，特征丰富 | 机器人视觉 |
| **SigLIP** | 语言-视觉对齐 | VLA任务 |

### 代码示例

```python
# 简单CNN编码器
class SimpleVisualEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, output_dim)
        )

    def forward(self, image):
        return self.encoder(image)  # [B, output_dim]

# ViT编码器（使用预训练）
class ViTEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.proj = nn.Linear(768, output_dim)

    def forward(self, image):
        features = self.vit(image).last_hidden_state[:, 0]
        return self.proj(features)
```

---

## 3. 语言编码 (Language Encoding)

### 编码方式对比

```
方式1：词袋模型 (Bag of Words)
"fly to the red building"
→ [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, ...]
优点：简单快速
缺点：丢失语序信息

方式2：词嵌入 + 平均
"fly to the red building"
→ [emb_fly, emb_to, emb_the, emb_red, emb_building]
→ mean → [256维向量]
优点：保留语义
缺点：丢失语序

方式3：Transformer编码器（BERT等）
"fly to the red building"
→ [CLS] fly to the red building [SEP]
→ BERT → [768维向量]
优点：完整语义理解
缺点：计算量大
```

### 代码示例

```python
# 简单词嵌入编码器
class SimpleLanguageEncoder(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_ids):
        embeds = self.embedding(token_ids)  # [B, L, D]
        pooled = embeds.mean(dim=1)  # [B, D]
        return self.fc(pooled)

# BERT编码器
class BERTLanguageEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.proj = nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled = outputs.pooler_output  # [B, 768]
        return self.proj(pooled)
```

---

## 4. 状态编码 (State Encoding)

### 无人机状态表示

```python
# 12维状态向量
state = [
    x, y, z,           # 位置 (m)
    vx, vy, vz,        # 速度 (m/s)
    roll, pitch, yaw,  # 姿态 (rad)
    wx, wy, wz         # 角速度 (rad/s)
]

# 编码方式
class StateEncoder(nn.Module):
    def __init__(self, state_dim=12, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, state):
        return self.encoder(state)  # [B, output_dim]
```

### 为什么需要状态编码？

```
视觉只能看到"当前看到什么"
状态知道"自己在哪里、怎么动的"

例子：
- 视觉：看到前方有建筑
- 状态：知道自己以2m/s前进，高度5m

融合后：
- 知道"我正以2m/s接近前方建筑，当前高度5m"
- 可以做出更准确的决策
```

---

## 5. 动作空间 (Action Space)

### 无人机动作空间设计

```python
# 方案1：速度控制（最常用）
action = [vx, vy, vz, yaw_rate]
# 范围：[-1, 1]（归一化后）
# 映射：实际速度 = action * max_velocity

# 方案2：加速度控制
action = [ax, ay, az, yaw_rate]
# 更底层，需要积分得到速度

# 方案3：目标位置
action = [target_x, target_y, target_z, target_yaw]
# 需要控制器转换为速度指令

# 选择建议：
# - 速度控制：简单直接，适合学习
# - 加速度控制：更平滑，适合精密任务
# - 目标位置：需要额外控制器，但更直观
```

### 动作归一化

```python
# 为什么需要归一化？
# - 不同维度范围不同（位置0-20m，速度0-2m/s）
# - 神经网络输出范围有限（Tanh: [-1, 1]）

# 归一化方法
def normalize_action(action, action_range):
    return action / action_range  # 映射到[-1, 1]

def denormalize_action(normalized, action_range):
    return normalized * action_range  # 映射回实际范围

# 无人机示例
max_velocity = 2.0  # m/s
action_range = np.array([max_velocity, max_velocity, max_velocity, 1.0])

# 训练时：归一化
normalized_action = normalize_action(expert_action, action_range)

# 推理时：反归一化
actual_action = denormalize_action(model_output, action_range)
```

---

## 6. 损失函数 (Loss Functions)

### 常用损失函数

```python
# 1. MSE损失（回归任务）
loss = nn.MSELoss()
l = loss(predicted_action, expert_action)

# 2. L1损失（对异常值更鲁棒）
loss = nn.L1Loss()
l = loss(predicted_action, expert_action)

# 3. Huber损失（结合MSE和L1）
loss = nn.HuberLoss(delta=1.0)
l = loss(predicted_action, expert_action)

# 4. 动作分量加权损失
def weighted_mse_loss(pred, target, weights):
    return (weights * (pred - target) ** 2).mean()

# 权重设置：给重要维度更高权重
weights = torch.tensor([1.0, 1.0, 2.0, 0.5])  # z轴更重要
```

---

## 7. 训练策略 (Training Strategies)

### 学习率调度

```python
# 1. 余弦退火
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# 2. 步进衰减
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# 3. 预热+衰减
def warmup_schedule(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0
```

### 正则化

```python
# 1. Dropout
self.fc = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.1),  # 10%的神经元随机失活
    nn.Linear(128, action_dim)
)

# 2. 权重衰减（L2正则）
optimizer = AdamW(model.parameters(), weight_decay=1e-4)

# 3. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 8. 评估指标 (Evaluation Metrics)

### 无人机VLA评估指标

```python
# 1. 成功率
success_rate = num_success / total_episodes

# 2. 平均步数
avg_steps = total_steps / total_episodes

# 3. 轨迹误差
def trajectory_error(predicted, reference):
    return np.mean(np.linalg.norm(predicted - reference, axis=1))

# 4. nDTW（归一化动态时间规整）
def ndtw(predicted, reference):
    dtw_dist = compute_dtw(predicted, reference)
    return np.exp(-dtw_dist / len(reference))

# 5. 碰撞率
collision_rate = num_collisions / total_episodes

# 6. 任务完成时间
completion_time = steps * dt  # dt为控制周期
```

---

## 9. 关键超参数

| 超参数 | 典型值 | 影响 |
|--------|--------|------|
| **学习率** | 1e-4 ~ 1e-3 | 收敛速度和稳定性 |
| **批大小** | 16 ~ 64 | 训练稳定性和速度 |
| **嵌入维度** | 128 ~ 512 | 模型容量 |
| **帧数** | 1 ~ 8 | 时序信息 |
| **训练轮数** | 50 ~ 200 | 收敛程度 |
| **权重衰减** | 1e-5 ~ 1e-3 | 过拟合控制 |

---

## 10. 常见问题

### Q: 模型输出全为0或固定值？
A: 检查损失函数是否正确，数据是否归一化，学习率是否合适

### Q: 训练损失不下降？
A: 检查数据格式，降低学习率，增加模型容量

### Q: 过拟合严重？
A: 增加数据量，添加正则化，使用数据增强

### Q: 动作抖动？
A: 增加时序平滑约束，使用更大的动作平滑窗口

---

*上一节：[VLA发展历史](02_Development_History.md) | 下一节：[代码教程](../03_Code_Tutorials/)*
