"""
Demo 02: CLIPort - 语言接地的机器人操作

核心思想：
- 结合CLIP的语义理解(what)与TransporterNet的空间精度(where)
- 通过注意力融合实现语言条件的操作

参考论文：CLIPort: What and Where Pathways for Robotic Manipulation

作者：DroneVLA Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================
# 1. CLIP视觉编码器（简化版）
# ============================================================

class SimpleCLIPEncoder(nn.Module):
    """
    简化的CLIP编码器

    实际CLIP使用：
    - 视觉：ViT (Vision Transformer)
    - 文本：Transformer

    这里使用简化的CNN和RNN作为教学示例
    """

    def __init__(self, embed_dim=256):
        super().__init__()

        # 视觉编码器
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim)
        )

        # 文本编码器
        self.text_encoder = nn.GRU(
            input_size=64,
            hidden_size=embed_dim,
            num_layers=2,
            batch_first=True
        )
        self.text_embedding = nn.Embedding(1000, 64)  # 词汇表大小1000

    def encode_image(self, image):
        """编码图像"""
        return self.visual_encoder(image)

    def encode_text(self, tokens):
        """编码文本"""
        embeds = self.text_embedding(tokens)
        _, hidden = self.text_encoder(embeds)
        return hidden[-1]  # 取最后一层的隐藏状态


# ============================================================
# 2. TransportNet（简化版）
# ============================================================

class TransportNet(nn.Module):
    """
    TransportNet：空间精度网络

    核心思想：
    - 预测pick位置（抓取点）
    - 预测place位置（放置点）
    - 通过注意力机制实现
    """

    def __init__(self, in_channels=3, feature_dim=64):
        super().__init__()

        # 特征提取 (带下采样)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, feature_dim, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, image):
        """
        提取空间特征图

        Args:
            image: [B, C, H, W]
        Returns:
            features: [B, D, H, W] 空间特征图
        """
        return self.features(image)


# ============================================================
# 3. CLIPort模型
# ============================================================

class CLIPort(nn.Module):
    """
    CLIPort模型

    结合CLIP语义与TransportNet空间精度

    架构：
    1. CLIP编码视觉和语言
    2. TransportNet提取空间特征
    3. 注意力融合生成pick和place热图
    """

    def __init__(self, embed_dim=256, feature_dim=64):
        super().__init__()

        # CLIP编码器
        self.clip = SimpleCLIPEncoder(embed_dim)

        # TransportNet
        self.transport = TransportNet(feature_dim=feature_dim)

        # 注意力融合
        self.attention = nn.Sequential(
            nn.Linear(embed_dim + feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Pick和Place预测头
        self.pick_head = nn.Conv2d(feature_dim, 1, 1)
        self.place_head = nn.Conv2d(feature_dim, 1, 1)

    def forward(self, image, instruction_tokens):
        """
        前向传播

        Args:
            image: [B, 3, H, W] 输入图像
            instruction_tokens: [B, seq_len] 指令token
        Returns:
            pick_heatmap: [B, 1, H, W] 抓取位置热图
            place_heatmap: [B, 1, H, W] 放置位置热图
        """
        B, C, H, W = image.shape

        # 1. CLIP编码
        image_features = self.clip.encode_image(image)  # [B, embed_dim]
        text_features = self.clip.encode_text(instruction_tokens)  # [B, embed_dim]

        # 2. 语言条件的视觉特征
        # 扩展文本特征以匹配空间维度
        text_features_spatial = text_features.unsqueeze(-1).unsqueeze(-1)
        text_features_spatial = text_features_spatial.expand(-1, -1, H // 4, W // 4)

        # 3. TransportNet提取空间特征
        spatial_features = self.transport(image)  # [B, feature_dim, H/4, W/4]

        # 4. 注意力融合
        # 将空间特征和文本特征拼接
        combined = torch.cat([
            spatial_features,
            text_features_spatial
        ], dim=1)  # [B, feature_dim + embed_dim, H/4, W/4]

        # 计算注意力权重
        combined_flat = combined.permute(0, 2, 3, 1)  # [B, H/4, W/4, D]
        attention_weights = self.attention(combined_flat)  # [B, H/4, W/4, 1]
        attention_weights = F.softmax(attention_weights.view(B, -1), dim=-1)
        attention_weights = attention_weights.view(B, 1, H // 4, W // 4)

        # 加权的空间特征
        attended_features = spatial_features * attention_weights

        # 5. 预测pick和place位置
        pick_heatmap = self.pick_head(attended_features)
        place_heatmap = self.place_head(attended_features)

        # 上采样到原始分辨率
        pick_heatmap = F.interpolate(pick_heatmap, size=(H, W), mode='bilinear')
        place_heatmap = F.interpolate(place_heatmap, size=(H, W), mode='bilinear')

        return pick_heatmap, place_heatmap


# ============================================================
# 4. 简单的桌面操作环境
# ============================================================

class SimpleManipulationEnv:
    """
    简单的桌面操作环境

    - 桌面上有多个彩色方块
    - 任务：根据语言指令抓取和放置
    """

    def __init__(self, size=64):
        self.size = size
        self.objects = []
        self.robot_pos = None

    def reset(self):
        """重置环境"""
        # 随机生成2-4个物体
        num_objects = np.random.randint(2, 5)
        self.objects = []
        colors = ['red', 'blue', 'green', 'yellow']

        for i in range(num_objects):
            obj = {
                'color': colors[i],
                'pos': np.random.randint(5, self.size - 5, 2),
                'size': np.random.randint(4, 8)
            }
            self.objects.append(obj)

        self.robot_pos = np.array([self.size // 2, self.size // 2])

        return self._get_obs(), self._get_instruction()

    def step(self, pick_pos, place_pos):
        """
        执行pick-and-place操作

        Args:
            pick_pos: (x, y) 抓取位置
            place_pos: (x, y) 放置位置
        Returns:
            obs: 新的观测
            reward: 奖励
            done: 是否成功
        """
        # 找到最近的物体
        min_dist = float('inf')
        picked_obj = None

        for obj in self.objects:
            dist = np.linalg.norm(np.array(pick_pos) - obj['pos'])
            if dist < min_dist:
                min_dist = dist
                picked_obj = obj

        # 移动物体
        if picked_obj and min_dist < picked_obj['size']:
            picked_obj['pos'] = np.array(place_pos)
            reward = 1.0
            done = True
        else:
            reward = -0.1
            done = False

        return self._get_obs(), reward, done

    def _get_obs(self):
        """获取观测图像"""
        obs = np.ones((3, self.size, self.size), dtype=np.float32) * 0.9  # 白色背景

        # 绘制物体
        for obj in self.objects:
            x, y = obj['pos']
            s = obj['size']
            color = {'red': [1,0,0], 'blue': [0,0,1], 'green': [0,1,0], 'yellow': [1,1,0]}
            c = color[obj['color']]
            obs[0, x-s:x+s, y-s:y+s] = c[0]
            obs[1, x-s:x+s, y-s:y+s] = c[1]
            obs[2, x-s:x+s, y-s:y+s] = c[2]

        return obs

    def _get_instruction(self):
        """生成语言指令"""
        if not self.objects:
            return "do nothing"

        obj = np.random.choice(self.objects)
        target_pos = np.random.randint(10, self.size - 10, 2)

        return f"pick the {obj['color']} block and place it at {target_pos}"


# ============================================================
# 5. 训练和评估
# ============================================================

def train_clip_transport():
    """训练CLIPort模型"""
    print("=" * 60)
    print("Demo 02: CLIPort - 语言接地的机器人操作")
    print("=" * 60)

    # 创建模型和环境
    model = CLIPort()
    env = SimpleManipulationEnv()

    # 简单的分词器
    vocab = {'<pad>': 0, '<unk>': 1}
    word_idx = 2

    def tokenize(text):
        nonlocal word_idx
        tokens = []
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = word_idx
                word_idx += 1
            tokens.append(vocab[word])
        return tokens + [0] * (20 - len(tokens))  # 填充到20

    # 收集演示数据
    print("\n[1/3] 收集演示数据...")
    demos = []

    for _ in range(50):
        obs, instruction = env.reset()
        tokens = tokenize(instruction)

        # 简单的启发式：选择目标位置附近
        # 这里简化处理，实际应该有专家策略
        pick_pos = env.objects[0]['pos'] if env.objects else [32, 32]
        place_pos = [32, 32]

        demos.append({
            'obs': obs,
            'tokens': tokens,
            'pick': pick_pos,
            'place': place_pos
        })

    print(f"  收集了 {len(demos)} 个演示样本")

    # 训练模型
    print("\n[2/3] 训练模型...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        total_loss = 0

        for demo in demos:
            # 准备数据
            obs_tensor = torch.FloatTensor(demo['obs']).unsqueeze(0)
            tokens_tensor = torch.LongTensor(demo['tokens']).unsqueeze(0)

            # 创建目标热图
            pick_target = torch.zeros(1, 1, 64, 64)
            place_target = torch.zeros(1, 1, 64, 64)

            px, py = demo['pick']
            pick_target[0, 0, px-2:px+2, py-2:py+2] = 1.0

            ax, ay = demo['place']
            place_target[0, 0, ax-2:ax+2, ay-2:ay+2] = 1.0

            # 前向传播
            pick_pred, place_pred = model(obs_tensor, tokens_tensor)

            # 计算损失
            loss = F.mse_loss(pick_pred, pick_target) + F.mse_loss(place_pred, place_target)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30, Loss: {total_loss/len(demos):.4f}")

    # 评估
    print("\n[3/3] 评估模型...")
    print("  模型已训练完成，可以预测pick和place位置")
    print("  在实际应用中，需要更复杂的专家策略和更大的数据集")

    print("\n" + "=" * 60)
    print("Demo完成！")
    print("关键概念：")
    print("  1. CLIP语义理解 (what)")
    print("  2. TransportNet空间精度 (where)")
    print("  3. 注意力融合机制")
    print("=" * 60)

    return model


if __name__ == "__main__":
    model = train_clip_transport()
