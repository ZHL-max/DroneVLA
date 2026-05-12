# 从零构建VLA模型
## 手把手教你用PyTorch实现一个完整的VLA

---

## 目标

用PyTorch从零实现一个能工作的VLA模型，包含：
1. 视觉编码器（CNN）
2. 语言编码器（Embedding + RNN）
3. 多模态融合（Concatenation + MLP）
4. 动作解码器（MLP）
5. 完整训练流程

---

## Step 1：视觉编码器

```python
import torch
import torch.nn as nn

class VisualEncoder(nn.Module):
    """
    视觉编码器：将图像转换为特征向量

    输入：[B, 3, 64, 64] RGB图像
    输出：[B, 256] 特征向量
    """

    def __init__(self, embed_dim=256):
        super().__init__()

        # 卷积层提取空间特征
        self.conv_layers = nn.Sequential(
            # 第1层：3 → 32通道，保持空间尺寸
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 → 32x32

            # 第2层：32 → 64通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 → 16x16

            # 第3层：64 → 128通道
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 16x16 → 1x1
        )

        # 全连接层映射到嵌入维度
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        """
        x: [B, 3, 64, 64]
        return: [B, embed_dim]
        """
        features = self.conv_layers(x)  # [B, 128, 1, 1]
        features = features.flatten(1)   # [B, 128]
        return self.fc(features)         # [B, embed_dim]
```

**关键点**：
- `BatchNorm2d`：稳定训练
- `AdaptiveAvgPool2d(1)`：将任意尺寸的特征图压缩为1x1
- `flatten(1)`：将多维特征展平

---

## Step 2：语言编码器

```python
class LanguageEncoder(nn.Module):
    """
    语言编码器：将指令文本转换为特征向量

    输入：token序列 [B, seq_len]
    输出：[B, embed_dim] 特征向量
    """

    def __init__(self, vocab_size=1000, embed_dim=256, hidden_dim=256):
        super().__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # GRU处理序列
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True  # 双向GRU
        )

        # 映射到统一维度
        self.fc = nn.Linear(hidden_dim * 2, embed_dim)  # *2因为双向

    def forward(self, tokens):
        """
        tokens: [B, seq_len] token序列
        return: [B, embed_dim]
        """
        # 词嵌入
        embeds = self.embedding(tokens)  # [B, seq_len, embed_dim]

        # GRU编码
        outputs, hidden = self.rnn(embeds)  # outputs: [B, seq_len, hidden*2]

        # 取最后时刻的隐藏状态
        # hidden: [4, B, hidden] (2层*2方向)
        # 拼接前向和后向的最后隐藏状态
        forward_hidden = hidden[-2]   # 前向最后层
        backward_hidden = hidden[-1]  # 后向最后层
        combined = torch.cat([forward_hidden, backward_hidden], dim=-1)

        return self.fc(combined)  # [B, embed_dim]
```

**关键点**：
- `nn.Embedding`：将整数token转换为稠密向量
- `bidirectional=True`：双向GRU能更好地理解上下文
- 取最后时刻的隐藏状态作为整个句子的表示

---

## Step 3：状态编码器

```python
class StateEncoder(nn.Module):
    """
    状态编码器：将无人机状态转换为特征向量

    输入：[B, 12] 状态向量
    输出：[B, 128] 特征向量
    """

    def __init__(self, state_dim=12, embed_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, state):
        """
        state: [B, 12] 包含位置、速度、姿态
        return: [B, embed_dim]
        """
        return self.mlp(state)
```

---

## Step 4：多模态融合

```python
class MultimodalFusion(nn.Module):
    """
    多模态融合：将视觉、语言、状态特征融合

    输入：visual [B, 256], language [B, 256], state [B, 128]
    输出：[B, 256] 融合特征
    """

    def __init__(self, visual_dim=256, language_dim=256, state_dim=128, output_dim=256):
        super().__init__()

        input_dim = visual_dim + language_dim + state_dim

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, visual_feat, language_feat, state_feat):
        """
        visual_feat: [B, 256]
        language_feat: [B, 256]
        state_feat: [B, 128]
        return: [B, output_dim]
        """
        # 拼接所有特征
        combined = torch.cat([visual_feat, language_feat, state_feat], dim=-1)
        return self.fusion(combined)
```

---

## Step 5：动作解码器

```python
class ActionDecoder(nn.Module):
    """
    动作解码器：将融合特征转换为控制动作

    输入：[B, 256] 融合特征
    输出：[B, 4] 动作 [vx, vy, vz, yaw_rate]
    """

    def __init__(self, input_dim=256, action_dim=4):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # 输出归一化到[-1, 1]
        )

    def forward(self, fused_features):
        """
        fused_features: [B, 256]
        return: [B, 4] 归一化的动作
        """
        return self.decoder(fused_features)
```

---

## Step 6：组装完整模型

```python
class DroneVLA(nn.Module):
    """
    完整的无人机VLA模型

    架构：
    图像 → VisualEncoder ─┐
                          ├→ Fusion → ActionDecoder → 动作
    指令 → LanguageEncoder ┘
    状态 → StateEncoder ──┘
    """

    def __init__(
        self,
        vocab_size=1000,
        visual_dim=256,
        language_dim=256,
        state_dim=12,
        state_embed_dim=128,
        action_dim=4
    ):
        super().__init__()

        # 编码器
        self.visual_encoder = VisualEncoder(embed_dim=visual_dim)
        self.language_encoder = LanguageEncoder(vocab_size=vocab_size, embed_dim=language_dim)
        self.state_encoder = StateEncoder(state_dim=state_dim, embed_dim=state_embed_dim)

        # 融合层
        self.fusion = MultimodalFusion(
            visual_dim=visual_dim,
            language_dim=language_dim,
            state_dim=state_embed_dim,
            output_dim=256
        )

        # 动作解码器
        self.action_decoder = ActionDecoder(input_dim=256, action_dim=action_dim)

    def forward(self, image, instruction_tokens, state):
        """
        前向传播

        Args:
            image: [B, 3, 64, 64] RGB图像
            instruction_tokens: [B, seq_len] 指令token
            state: [B, 12] 无人机状态

        Returns:
            action: [B, 4] 控制动作
        """
        # 编码各模态
        visual_feat = self.visual_encoder(image)           # [B, 256]
        language_feat = self.language_encoder(instruction_tokens)  # [B, 256]
        state_feat = self.state_encoder(state)             # [B, 128]

        # 多模态融合
        fused = self.fusion(visual_feat, language_feat, state_feat)  # [B, 256]

        # 解码动作
        action = self.action_decoder(fused)  # [B, 4]

        return action
```

---

## Step 7：训练循环

```python
import torch.optim as optim
from torch.utils.data import DataLoader

def train_vla(model, train_loader, num_epochs=50, lr=1e-3):
    """训练VLA模型"""

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            # 解包数据
            images = batch['images']           # [B, 3, 64, 64]
            tokens = batch['tokens']           # [B, seq_len]
            states = batch['states']           # [B, 12]
            target_actions = batch['actions']  # [B, 4]

            # 前向传播
            predicted_actions = model(images, tokens, states)

            # 计算损失
            loss = criterion(predicted_actions, target_actions)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 打印进度
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model
```

---

## Step 8：推理使用

```python
def inference(model, image, instruction, state):
    """使用训练好的模型进行推理"""

    model.eval()
    with torch.no_grad():
        # 预处理输入
        image_tensor = preprocess_image(image)  # [1, 3, 64, 64]
        tokens = tokenize(instruction)          # [1, seq_len]
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # [1, 12]

        # 模型推理
        action = model(image_tensor, tokens, state_tensor)

        # 后处理
        action = action.numpy()[0]  # [4]
        vx, vy, vz, yaw_rate = action

        return {
            'vx': vx * 2.0,      # 缩放到实际速度范围
            'vy': vy * 2.0,
            'vz': vz * 2.0,
            'yaw_rate': yaw_rate * 1.0
        }

# 使用示例
action = inference(model, camera_image, "fly to the red building", drone_state)
print(f"动作: 前进{action['vx']:.2f}m/s, 右移{action['vy']:.2f}m/s")
```

---

## 完整代码

完整的可运行代码请参考：
- `demos/01_simple_vla/simple_vla.py` - 最简单的VLA示例
- `src/models/drone_vla.py` - 完整的DroneVLA实现

---

## 练习题

1. **修改视觉编码器**：将CNN替换为ResNet-18，观察训练效果变化
2. **添加注意力机制**：在融合层中加入交叉注意力
3. **实现时序VLA**：处理连续多帧图像输入
4. **尝试不同的动作空间**：将4维速度控制改为7维位置控制

---

*下一节：[训练你的第一个模型](03_Train_Your_First_Model.md)*
