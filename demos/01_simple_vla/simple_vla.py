"""
Demo 01: 最简单的VLA (Vision-Language-Action) 示例

核心思想：
- 输入：图像观测 + 自然语言指令
- 输出：机器人动作

这个Demo演示了VLA的基本原理：
1. 使用CLIP编码视觉和语言
2. 通过简单的MLP解码为动作
3. 在简单的网格环境中执行

作者：DroneVLA Project
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# 1. 简单的VLA模型
# ============================================================

class SimpleVLA(nn.Module):
    """
    最简单的VLA模型

    架构：
    - 视觉编码器：简单的CNN
    - 语言编码器：简单的嵌入层
    - 动作解码器：MLP

    这是一个教学示例，展示了VLA的核心思想
    """

    def __init__(self, vocab_size=100, embed_dim=64, action_dim=4):
        super().__init__()

        # 视觉编码器：简单的CNN
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, embed_dim)  # 假设输入64x64
        )

        # 语言编码器：嵌入层 + 平均池化
        self.language_encoder = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # 多模态融合
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 动作解码器
        self.action_decoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()  # 输出归一化到[-1, 1]
        )

    def forward(self, image, instruction_tokens):
        """
        前向传播

        Args:
            image: [B, 3, 64, 64] 图像张量
            instruction_tokens: [B, seq_len] 指令token序列
        Returns:
            action: [B, action_dim] 动作预测
        """
        # 编码视觉
        visual_features = self.visual_encoder(image)  # [B, embed_dim]

        # 编码语言
        lang_embed = self.language_encoder(instruction_tokens)  # [B, seq_len, embed_dim]
        lang_features = lang_embed.mean(dim=1)  # [B, embed_dim]

        # 多模态融合
        combined = torch.cat([visual_features, lang_features], dim=-1)  # [B, embed_dim*2]
        fused = self.fusion(combined)  # [B, 64]

        # 解码动作
        action = self.action_decoder(fused)  # [B, action_dim]

        return action


# ============================================================
# 2. 简单的网格环境
# ============================================================

class SimpleGridEnv:
    """
    简单的网格环境

    - 10x10网格
    - 机器人在网格中移动
    - 目标位置由语言指令指定
    """

    def __init__(self, size=10):
        self.size = size
        self.robot_pos = None
        self.goal_pos = None
        self.step_count = 0
        self.max_steps = 50

    def reset(self):
        """重置环境"""
        self.robot_pos = np.array([0, 0])
        self.goal_pos = np.array([self.size-1, self.size-1])
        self.step_count = 0
        return self._get_obs(), self._get_instruction()

    def step(self, action):
        """
        执行动作

        Args:
            action: [dx, dy] 归一化动作 [-1, 1]
        Returns:
            obs: 新的观测
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 将动作转换为网格移动
        dx = int(np.clip(action[0] * 2, -1, 1))
        dy = int(np.clip(action[1] * 2, -1, 1))

        # 更新位置
        new_pos = self.robot_pos + np.array([dx, dy])
        new_pos = np.clip(new_pos, 0, self.size - 1)
        self.robot_pos = new_pos

        self.step_count += 1

        # 计算奖励
        distance = np.linalg.norm(self.robot_pos - self.goal_pos)
        reward = -distance / self.size  # 距离越近奖励越高

        # 检查是否到达目标
        if np.array_equal(self.robot_pos, self.goal_pos):
            reward = 10.0
            done = True
        elif self.step_count >= self.max_steps:
            done = True
        else:
            done = False

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        """获取观测（简单的图像表示）"""
        obs = np.zeros((3, 64, 64), dtype=np.float32)

        # 绘制网格
        for i in range(self.size):
            for j in range(self.size):
                x = i * 64 // self.size
                y = j * 64 // self.size
                obs[:, x:x+4, y:y+4] = 0.1  # 网格线

        # 绘制目标（红色）
        gx, gy = self.goal_pos
        x, y = gx * 64 // self.size, gy * 64 // self.size
        obs[0, x+4:x+12, y+4:y+12] = 1.0  # 红色

        # 绘制机器人（蓝色）
        rx, ry = self.robot_pos
        x, y = rx * 64 // self.size, ry * 64 // self.size
        obs[2, x+4:x+12, y+4:y+12] = 1.0  # 蓝色

        return obs

    def _get_instruction(self):
        """获取语言指令"""
        instructions = [
            "move to the red goal",
            "go to the target position",
            "navigate to the goal",
            "reach the red square"
        ]
        return np.random.choice(instructions)


# ============================================================
# 3. 词汇表和编码器
# ============================================================

class SimpleTokenizer:
    """
    简单的分词器

    将自然语言指令转换为token序列
    """

    def __init__(self):
        self.word2idx = {
            '<pad>': 0, '<unk>': 1,
            'move': 2, 'to': 3, 'the': 4, 'red': 5,
            'goal': 6, 'go': 7, 'target': 8, 'position': 9,
            'navigate': 10, 'reach': 11, 'square': 12
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode(self, text, max_len=10):
        """将文本编码为token序列"""
        words = text.lower().split()
        tokens = [self.word2idx.get(w, 1) for w in words]  # 1 = <unk>

        # 填充到固定长度
        if len(tokens) < max_len:
            tokens = tokens + [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]

        return tokens


# ============================================================
# 4. 训练循环
# ============================================================

def collect_demonstrations(env, num_episodes=100):
    """
    收集专家演示数据

    使用简单的启发式策略：
    - 向目标方向移动
    """
    demos = []

    for _ in range(num_episodes):
        obs, instruction = env.reset()
        episode_data = []

        for _ in range(env.max_steps):
            # 启发式策略：向目标方向移动
            diff = env.goal_pos - env.robot_pos
            action = np.array([
                np.sign(diff[0]) * 0.5,
                np.sign(diff[1]) * 0.5
            ])

            next_obs, reward, done, _ = env.step(action)

            episode_data.append({
                'obs': obs,
                'instruction': instruction,
                'action': action
            })

            obs = next_obs

            if done:
                break

        demos.extend(episode_data)

    return demos


def train_simple_vla():
    """训练简单的VLA模型"""
    print("=" * 60)
    print("Demo 01: 最简单的VLA示例")
    print("=" * 60)

    # 创建环境和模型
    env = SimpleGridEnv()
    model = SimpleVLA(action_dim=2)  # 网格环境只需要2D动作 (dx, dy)
    tokenizer = SimpleTokenizer()

    # 收集演示数据
    print("\n[1/4] 收集专家演示数据...")
    demos = collect_demonstrations(env, num_episodes=50)
    print(f"  收集了 {len(demos)} 个演示样本")

    # 准备训练数据
    print("\n[2/4] 准备训练数据...")
    observations = torch.FloatTensor(np.array([d['obs'] for d in demos]))
    instructions = torch.LongTensor(np.array([tokenizer.encode(d['instruction']) for d in demos]))
    actions = torch.FloatTensor(np.array([d['action'] for d in demos]))

    # 训练模型
    print("\n[3/4] 训练VLA模型...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(50):
        # 前向传播
        pred_actions = model(observations, instructions)
        loss = criterion(pred_actions, actions)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

    # 评估模型
    print("\n[4/4] 评估模型...")
    success_count = 0
    total_episodes = 20

    for _ in range(total_episodes):
        obs, instruction = env.reset()

        for _ in range(env.max_steps):
            # 编码输入
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            inst_tensor = torch.LongTensor(tokenizer.encode(instruction)).unsqueeze(0)

            # 预测动作
            with torch.no_grad():
                action = model(obs_tensor, inst_tensor).numpy()[0]

            # 执行动作
            obs, reward, done, _ = env.step(action)

            if done:
                if np.array_equal(env.robot_pos, env.goal_pos):
                    success_count += 1
                break

    print(f"\n  成功率: {success_count}/{total_episodes} ({success_count/total_episodes*100:.1f}%)")

    # 可视化训练曲线
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VLA Training Loss')
    plt.grid(True)
    plt.savefig('demos/01_simple_vla/training_curve.png')
    plt.close()
    print("\n  训练曲线已保存到 demos/01_simple_vla/training_curve.png")

    print("\n" + "=" * 60)
    print("Demo完成！")
    print("=" * 60)

    return model


if __name__ == "__main__":
    model = train_simple_vla()
