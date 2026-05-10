"""
Demo 04: 世界模型 (World Model)

核心思想：
- 学习环境的动态模型
- 在想象中预测未来状态
- 用于策略训练和规划

参考：
- DayDreamer: World Models for Physical Robot Learning
- TD-MPC2: Scalable, Robust World Models for Continuous Control

作者：DroneVLA Project
"""

import torch
import torch.nn as nn
import numpy as np

# ============================================================
# 1. 世界模型核心组件
# ============================================================

class DynamicsModel(nn.Module):
    """
    动态模型

    预测给定状态和动作下的下一个状态
    s_{t+1} = f(s_t, a_t)
    """

    def __init__(self, state_dim=6, action_dim=3, hidden_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        """
        预测下一个状态

        Args:
            state: [B, state_dim] 当前状态
            action: [B, action_dim] 动作
        Returns:
            next_state: [B, state_dim] 预测的下一个状态
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class RewardModel(nn.Module):
    """
    奖励模型

    预测给定状态和动作的奖励
    r_t = r(s_t, a_t)
    """

    def __init__(self, state_dim=6, action_dim=3, hidden_dim=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        """
        预测奖励

        Args:
            state: [B, state_dim] 状态
            action: [B, action_dim] 动作
        Returns:
            reward: [B, 1] 预测的奖励
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class ValueModel(nn.Module):
    """
    价值模型

    预测状态的价值（未来累积奖励）
    V(s) = E[Σ γ^t r_t]
    """

    def __init__(self, state_dim=6, hidden_dim=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """
        预测状态价值

        Args:
            state: [B, state_dim] 状态
        Returns:
            value: [B, 1] 状态价值
        """
        return self.network(state)


# ============================================================
# 2. 世界模型
# ============================================================

class WorldModel(nn.Module):
    """
    完整的世界模型

    包含：
    - 动态模型：预测状态转移
    - 奖励模型：预测奖励
    - 价值模型：评估状态价值

    用途：
    1. 在想象中训练策略
    2. 评估候选动作
    3. 规划最优动作序列
    """

    def __init__(self, state_dim=6, action_dim=3):
        super().__init__()

        self.dynamics = DynamicsModel(state_dim, action_dim)
        self.reward = RewardModel(state_dim, action_dim)
        self.value = ValueModel(state_dim)

    def imagine(self, state, policy, horizon=10):
        """
        想象未来轨迹

        Args:
            state: [B, state_dim] 初始状态
            policy: 策略网络
            horizon: 想象步数
        Returns:
            trajectory: 想象的轨迹
        """
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': []
        }

        current_state = state

        for _ in range(horizon):
            # 使用策略选择动作
            action = policy(current_state)

            # 预测下一个状态
            next_state = self.dynamics(current_state, action)

            # 预测奖励
            reward = self.reward(current_state, action)

            # 预测价值
            value = self.value(current_state)

            # 记录轨迹
            trajectory['states'].append(current_state)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['values'].append(value)

            current_state = next_state

        return trajectory

    def compute_imagined_return(self, trajectory, gamma=0.99):
        """
        计算想象轨迹的累积回报

        Args:
            trajectory: 想象的轨迹
            gamma: 折扣因子
        Returns:
            returns: 累积回报
        """
        returns = 0
        for t, reward in enumerate(trajectory['rewards']):
            returns += (gamma ** t) * reward

        return returns


# ============================================================
# 3. 策略网络
# ============================================================

class PolicyNetwork(nn.Module):
    """
    策略网络

    输入状态，输出动作
    """

    def __init__(self, state_dim=6, action_dim=3, hidden_dim=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出归一化到[-1, 1]
        )

    def forward(self, state):
        """
        选择动作

        Args:
            state: [B, state_dim] 状态
        Returns:
            action: [B, action_dim] 动作
        """
        return self.network(state)


# ============================================================
# 4. 简单环境
# ============================================================

class SimplePointEnv:
    """
    简单的点环境

    - 2D空间中的点机器人
    - 任务：到达目标位置
    """

    def __init__(self):
        self.pos = None
        self.goal = None
        self.max_steps = 50
        self.step_count = 0

    def reset(self):
        """重置环境"""
        self.pos = np.random.randn(2) * 0.5
        self.goal = np.random.randn(2) * 0.5
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        """执行动作"""
        self.pos = self.pos + action[:2] * 0.1
        self.step_count += 1

        dist = np.linalg.norm(self.pos - self.goal)
        reward = -dist

        done = dist < 0.1 or self.step_count >= self.max_steps

        return self._get_obs(), reward, done

    def _get_obs(self):
        """获取观测"""
        return np.concatenate([self.pos, self.goal, [0, 0]])  # 6维


# ============================================================
# 5. 训练和评估
# ============================================================

def train_world_model():
    """训练世界模型"""
    print("=" * 60)
    print("Demo 04: 世界模型 (World Model)")
    print("=" * 60)

    # 创建环境和模型
    env = SimplePointEnv()
    world_model = WorldModel(state_dim=6, action_dim=3)
    policy = PolicyNetwork(state_dim=6, action_dim=3)

    # 收集真实数据
    print("\n[1/4] 收集真实环境数据...")
    real_data = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': []
    }

    for _ in range(100):
        state = env.reset()

        for _ in range(env.max_steps):
            # 随机动作
            action = np.random.randn(3)

            next_state, reward, done = env.step(action)

            real_data['states'].append(state)
            real_data['actions'].append(action)
            real_data['next_states'].append(next_state)
            real_data['rewards'].append(reward)

            state = next_state

            if done:
                break

    # 转换为张量
    states = torch.FloatTensor(real_data['states'])
    actions = torch.FloatTensor(real_data['actions'])
    next_states = torch.FloatTensor(real_data['next_states'])
    rewards = torch.FloatTensor(real_data['rewards']).unsqueeze(-1)

    print(f"  收集了 {len(states)} 个样本")

    # 训练世界模型
    print("\n[2/4] 训练世界模型...")
    wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=0.001)

    for epoch in range(100):
        # 预测下一个状态
        pred_next_states = world_model.dynamics(states, actions)
        pred_rewards = world_model.reward(states, actions)

        # 计算损失
        dynamics_loss = nn.functional.mse_loss(pred_next_states, next_states)
        reward_loss = nn.functional.mse_loss(pred_rewards, rewards)
        loss = dynamics_loss + reward_loss

        # 反向传播
        wm_optimizer.zero_grad()
        loss.backward()
        wm_optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

    # 训练策略（在想象中）
    print("\n[3/4] 在想象中训练策略...")
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    for epoch in range(50):
        # 随机采样初始状态
        indices = np.random.choice(len(states), 16)
        initial_states = states[indices]

        # 在想象中展开轨迹
        trajectory = world_model.imagine(initial_states, policy, horizon=10)

        # 计算想象的回报
        imagined_return = world_model.compute_imagined_return(trajectory)

        # 最大化回报
        loss = -imagined_return.mean()

        # 反向传播
        policy_optimizer.zero_grad()
        loss.backward()
        policy_optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50, Imagined Return: {-loss.item():.4f}")

    # 评估策略
    print("\n[4/4] 评估策略...")
    success_count = 0
    total_episodes = 20

    for _ in range(total_episodes):
        state = env.reset()

        for _ in range(env.max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action = policy(state_tensor).numpy()[0]

            state, reward, done = env.step(action)

            if done:
                if np.linalg.norm(env.pos - env.goal) < 0.1:
                    success_count += 1
                break

    print(f"\n  成功率: {success_count}/{total_episodes} ({success_count/total_episodes*100:.1f}%)")

    print("\n" + "=" * 60)
    print("Demo完成！")
    print("关键概念：")
    print("  1. 动态模型：学习状态转移 s' = f(s, a)")
    print("  2. 奖励模型：预测奖励 r = r(s, a)")
    print("  3. 价值模型：评估状态价值 V(s)")
    print("  4. 想象训练：在世界模型中训练策略")
    print("  5. 模型预测控制(MPC)：使用世界模型规划")
    print("=" * 60)

    return world_model, policy


if __name__ == "__main__":
    world_model, policy = train_world_model()
