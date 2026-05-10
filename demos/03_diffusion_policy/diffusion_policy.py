"""
Demo 03: 扩散策略 (Diffusion Policy)

核心思想：
- 将动作生成建模为条件去噪过程
- 从噪声中逐步去噪生成动作序列
- 能够捕获多模态动作分布

参考论文：Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

作者：DroneVLA Project
"""

import torch
import torch.nn as nn
import numpy as np

# ============================================================
# 1. 扩散过程核心
# ============================================================

class DiffusionScheduler:
    """
    扩散调度器

    管理前向扩散（加噪）和反向去噪过程

    前向过程：q(x_t | x_0) = N(x_t; √(ᾱ_t) x_0, (1-ᾱ_t)I)
    反向过程：p(x_{t-1} | x_t) 通过学习的去噪网络预测
    """

    def __init__(self, num_steps=100, beta_start=0.0001, beta_end=0.02):
        self.num_steps = num_steps

        # 线性噪声调度
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

    def add_noise(self, x0, noise, t):
        """
        前向扩散：给干净数据加噪

        Args:
            x0: [B, T, D] 干净的动作序列
            noise: [B, T, D] 随机噪声
            t: [B] 时间步
        Returns:
            x_t: [B, T, D] 加噪后的数据
        """
        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1)

        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    def step(self, model_output, t, x_t):
        """
        反向去噪步骤

        Args:
            model_output: [B, T, D] 模型预测的噪声
            t: [B] 当前时间步
            x_t: [B, T, D] 当前含噪数据
        Returns:
            x_{t-1}: [B, T, D] 去噪后的数据
        """
        beta = self.betas[t].view(-1, 1, 1)
        alpha = self.alphas[t].view(-1, 1, 1)
        alpha_cumprod = self.alpha_cumprod[t].view(-1, 1, 1)

        # 去噪公式
        x0_pred = (x_t - torch.sqrt(1 - alpha_cumprod) * model_output) / torch.sqrt(alpha_cumprod)
        x0_pred = torch.clamp(x0_pred, -1, 1)

        # 计算x_{t-1}
        mean = torch.sqrt(alpha) * (x_t - beta / torch.sqrt(1 - alpha_cumprod) * model_output)
        variance = beta

        if t.min() > 0:
            noise = torch.randn_like(x_t)
            x_prev = mean + torch.sqrt(variance) * noise
        else:
            x_prev = mean

        return x_prev


# ============================================================
# 2. 去噪网络（简化版UNet）
# ============================================================

class SimpleDenoiser(nn.Module):
    """
    简化的去噪网络

    实际Diffusion Policy使用：
    - ConditionalUnet1D: 1D U-Net
    - TransformerForDiffusion: Transformer架构

    这里使用简化的MLP作为教学示例
    """

    def __init__(self, obs_dim=64, action_dim=4, action_horizon=8, hidden_dim=256):
        super().__init__()

        self.action_horizon = action_horizon
        self.action_dim = action_dim

        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )

        # 条件编码（观测）
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # 去噪网络
        self.denoiser = nn.Sequential(
            nn.Linear(action_horizon * action_dim + 128 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_horizon * action_dim)
        )

    def forward(self, x_t, t, obs):
        """
        预测噪声

        Args:
            x_t: [B, T, D] 含噪动作序列
            t: [B] 时间步
            obs: [B, obs_dim] 观测条件
        Returns:
            noise_pred: [B, T, D] 预测的噪声
        """
        B = x_t.shape[0]

        # 编码时间步
        t_embed = self.time_embed(t.float().unsqueeze(-1))  # [B, 64]

        # 编码观测
        obs_embed = self.obs_encoder(obs)  # [B, 128]

        # 拼接所有条件
        x_flat = x_t.reshape(B, -1)  # [B, T*D]
        combined = torch.cat([x_flat, obs_embed, t_embed], dim=-1)

        # 预测噪声
        noise_pred = self.denoiser(combined)
        noise_pred = noise_pred.reshape(B, self.action_horizon, self.action_dim)

        return noise_pred


# ============================================================
# 3. 扩散策略
# ============================================================

class DiffusionPolicy(nn.Module):
    """
    扩散策略

    结合扩散模型和条件生成，实现动作序列生成
    """

    def __init__(self, obs_dim=64, action_dim=4, action_horizon=8, num_diffusion_steps=50):
        super().__init__()

        self.action_horizon = action_horizon
        self.action_dim = action_dim

        # 扩散调度器
        self.scheduler = DiffusionScheduler(num_steps=num_diffusion_steps)

        # 去噪网络
        self.denoiser = SimpleDenoiser(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_horizon=action_horizon
        )

    def forward(self, obs, actions=None):
        """
        训练时：计算去噪损失
        推理时：生成动作序列
        """
        if actions is not None:
            # 训练模式
            return self.compute_loss(obs, actions)
        else:
            # 推理模式
            return self.generate_actions(obs)

    def compute_loss(self, obs, actions):
        """
        计算去噪损失

        Args:
            obs: [B, obs_dim] 观测
            actions: [B, T, D] 真实动作序列
        Returns:
            loss: 去噪损失
        """
        B = obs.shape[0]

        # 随机选择时间步
        t = torch.randint(0, self.scheduler.num_steps, (B,))

        # 生成随机噪声
        noise = torch.randn_like(actions)

        # 前向扩散
        x_t = self.scheduler.add_noise(actions, noise, t)

        # 预测噪声
        noise_pred = self.denoiser(x_t, t, obs)

        # 计算损失
        loss = nn.functional.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def generate_actions(self, obs):
        """
        生成动作序列

        Args:
            obs: [B, obs_dim] 观测
        Returns:
            actions: [B, T, D] 生成的动作序列
        """
        B = obs.shape[0]

        # 从纯噪声开始
        x = torch.randn(B, self.action_horizon, self.action_dim)

        # 逐步去噪
        for t in reversed(range(self.scheduler.num_steps)):
            t_batch = torch.full((B,), t)
            noise_pred = self.denoiser(x, t_batch, obs)
            x = self.scheduler.step(noise_pred, t_batch, x)

        return x


# ============================================================
# 4. 简单的机器人环境
# ============================================================

class SimpleReachEnv:
    """
    简单的到达任务环境

    - 3D空间中的点机器人
    - 任务：到达目标位置
    - 观测：当前位置 + 目标位置
    - 动作：速度指令
    """

    def __init__(self):
        self.pos = None
        self.goal = None
        self.max_steps = 50
        self.step_count = 0

    def reset(self):
        """重置环境"""
        self.pos = np.random.randn(3) * 0.5
        self.goal = np.random.randn(3) * 0.5
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        """
        执行动作

        Args:
            action: [vx, vy, vz] 速度指令
        Returns:
            obs: 新的观测
            reward: 奖励
            done: 是否完成
        """
        # 更新位置
        self.pos = self.pos + action * 0.1
        self.step_count += 1

        # 计算奖励
        dist = np.linalg.norm(self.pos - self.goal)
        reward = -dist

        # 检查是否到达
        done = dist < 0.1 or self.step_count >= self.max_steps

        return self._get_obs(), reward, done

    def _get_obs(self):
        """获取观测"""
        return np.concatenate([self.pos, self.goal])


# ============================================================
# 5. 训练和评估
# ============================================================

def train_diffusion_policy():
    """训练扩散策略"""
    print("=" * 60)
    print("Demo 03: 扩散策略 (Diffusion Policy)")
    print("=" * 60)

    # 创建环境和模型
    env = SimpleReachEnv()
    policy = DiffusionPolicy(
        obs_dim=6,      # 3D位置 + 3D目标
        action_dim=3,   # 3D速度
        action_horizon=8,
        num_diffusion_steps=20  # 减少步数以加快训练
    )

    # 收集演示数据
    print("\n[1/3] 收集专家演示数据...")
    demos = []

    for _ in range(200):
        obs = env.reset()
        trajectory = [obs.copy()]

        for _ in range(env.max_steps):
            # 专家策略：向目标方向移动
            direction = env.goal - env.pos
            action = direction * 2.0  # 简单的比例控制
            action = np.clip(action, -1, 1)

            obs, reward, done = env.step(action)
            trajectory.append(obs.copy())

            if done:
                break

        demos.append(trajectory)

    print(f"  收集了 {len(demos)} 条轨迹")

    # 准备训练数据
    print("\n[2/3] 准备训练数据...")
    observations = []
    action_sequences = []

    for traj in demos:
        for i in range(len(traj) - 8):
            obs = traj[i][:6]  # 位置 + 目标
            actions = []
            for j in range(8):
                if i + j + 1 < len(traj):
                    action = traj[i + j + 1][:3] - traj[i + j][:3]  # 速度
                    actions.append(action)
                else:
                    actions.append([0, 0, 0])

            observations.append(obs)
            action_sequences.append(actions)

    obs_tensor = torch.FloatTensor(observations)
    action_tensor = torch.FloatTensor(action_sequences)

    # 训练模型
    print("\n[3/3] 训练扩散策略...")
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    for epoch in range(50):
        # 随机采样批次
        indices = np.random.choice(len(obs_tensor), 32)
        obs_batch = obs_tensor[indices]
        action_batch = action_tensor[indices]

        # 计算损失
        loss = policy(obs_batch, action_batch)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

    # 评估
    print("\n  评估模型...")
    success_count = 0
    total_episodes = 20

    for _ in range(total_episodes):
        obs = env.reset()

        for _ in range(env.max_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            # 生成动作序列
            with torch.no_grad():
                actions = policy.generate_actions(obs_tensor)

            # 执行第一个动作
            action = actions[0, 0].numpy()
            obs, reward, done = env.step(action)

            if done:
                if np.linalg.norm(env.pos - env.goal) < 0.1:
                    success_count += 1
                break

    print(f"\n  成功率: {success_count}/{total_episodes} ({success_count/total_episodes*100:.1f}%)")

    print("\n" + "=" * 60)
    print("Demo完成！")
    print("关键概念：")
    print("  1. 扩散过程：前向加噪 + 反向去噪")
    print("  2. 条件生成：以观测为条件生成动作")
    print("  3. 多模态分布：能捕获多种可能的动作")
    print("  4. 动作序列：预测未来多步动作")
    print("=" * 60)

    return policy


if __name__ == "__main__":
    policy = train_diffusion_policy()
