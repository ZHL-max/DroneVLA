"""
DroneVLA 合成数据集生成器

生成用于训练的无人机飞行数据，包括：
- 图像观测（合成的俯视场景）
- 状态向量（位置、速度、姿态）
- 动作序列（速度控制指令）
- 语言指令（自然语言任务描述）

使用方法：
    python scripts/generate_dataset.py --num_episodes 1000 --output data/train
"""

import numpy as np
import os
import argparse
from typing import List, Dict, Tuple
import json


# ============================================================
# 场景定义
# ============================================================

TASKS = {
    "hover": {
        "instructions": [
            "hover at the current position",
            "stay in place",
            "maintain position",
            "hold steady at this location"
        ],
        "description": "保持悬停"
    },
    "navigate": {
        "instructions": [
            "fly to the red building",
            "navigate to the target location",
            "go to the waypoint",
            "move to the destination"
        ],
        "description": "导航到目标"
    },
    "follow": {
        "instructions": [
            "follow the moving object",
            "track the target",
            "keep following the vehicle",
            "pursue the moving target"
        ],
        "description": "跟踪目标"
    },
    "avoid": {
        "instructions": [
            "avoid the obstacles",
            "fly around the obstacles",
            "navigate through the gap",
            "dodge the obstacles ahead"
        ],
        "description": "避障"
    },
    "land": {
        "instructions": [
            "land at the designated area",
            "descend to the landing pad",
            "perform a gentle landing",
            "land safely on the ground"
        ],
        "description": "降落"
    },
    "takeoff": {
        "instructions": [
            "take off from the ground",
            "ascend to hover altitude",
            "launch and reach safe height",
            "rise to operating altitude"
        ],
        "description": "起飞"
    }
}


def generate_synthetic_image(
    drone_pos: np.ndarray,
    goal_pos: np.ndarray,
    obstacles: List[np.ndarray],
    image_size: int = 64
) -> np.ndarray:
    """
    生成合成的俯视图像

    Args:
        drone_pos: 无人机位置 [x, y, z]
        goal_pos: 目标位置 [x, y, z]
        obstacles: 障碍物位置列表
        image_size: 图像尺寸

    Returns:
        RGB图像 [H, W, 3]
    """
    image = np.zeros((image_size, image_size, 3), dtype=np.float32)

    # 背景色（灰色地面）
    image[:, :, :] = 0.3

    # 绘制网格
    for i in range(0, image_size, 8):
        image[i, :, :] = 0.25
        image[:, i, :] = 0.25

    # 绘制障碍物（黑色方块）
    for obs in obstacles:
        # 将世界坐标转换为像素坐标
        px = int(np.clip(obs[0] * image_size / 20, 2, image_size - 6))
        py = int(np.clip(obs[1] * image_size / 20, 2, image_size - 6))
        image[px:px+4, py:py+4, :] = 0.1

    # 绘制目标（红色圆点）
    gx = int(np.clip(goal_pos[0] * image_size / 20, 4, image_size - 5))
    gy = int(np.clip(goal_pos[1] * image_size / 20, 4, image_size - 5))
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx*dx + dy*dy <= 9:
                px, py = gx + dx, gy + dy
                if 0 <= px < image_size and 0 <= py < image_size:
                    image[px, py, 0] = 1.0  # 红色
                    image[px, py, 1] = 0.2
                    image[px, py, 2] = 0.2

    # 绘制无人机（蓝色方块）
    dx = int(np.clip(drone_pos[0] * image_size / 20, 4, image_size - 5))
    dy = int(np.clip(drone_pos[1] * image_size / 20, 4, image_size - 5))
    image[dx-2:dx+3, dy-2:dy+3, 2] = 1.0  # 蓝色
    image[dx-2:dx+3, dy-2:dy+3, 0] = 0.2
    image[dx-2:dx+3, dy-2:dy+3, 1] = 0.2

    return image


def generate_expert_trajectory(
    task: str,
    initial_state: np.ndarray,
    goal_pos: np.ndarray,
    obstacles: List[np.ndarray],
    num_steps: int = 50
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    生成专家轨迹

    Args:
        task: 任务类型
        initial_state: 初始状态 [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        goal_pos: 目标位置
        obstacles: 障碍物列表
        num_steps: 轨迹长度

    Returns:
        states: 状态序列
        actions: 动作序列
        images: 图像序列
    """
    states = []
    actions = []
    images = []

    state = initial_state.copy()

    for step in range(num_steps):
        pos = state[:3]
        vel = state[3:6]

        # 根据任务生成动作
        if task == "hover":
            # PID控制器保持位置
            target_vel = -0.5 * (pos - initial_state[:3])
            action = np.clip(target_vel, -1, 1)

        elif task == "navigate":
            # 向目标移动
            direction = goal_pos - pos
            distance = np.linalg.norm(direction)
            if distance > 0.5:
                target_vel = direction / distance * min(1.0, distance / 5.0)
            else:
                target_vel = np.zeros(3)
            action = np.clip(target_vel, -1, 1)

        elif task == "follow":
            # 跟踪移动目标
            moving_target = goal_pos + np.array([
                np.sin(step * 0.1) * 3,
                np.cos(step * 0.1) * 3,
                0
            ])
            direction = moving_target - pos
            distance = np.linalg.norm(direction)
            if distance > 1.0:
                target_vel = direction / distance * min(1.0, distance / 3.0)
            else:
                target_vel = np.zeros(3)
            action = np.clip(target_vel, -1, 1)

        elif task == "avoid":
            # 避障导航
            direction = goal_pos - pos
            distance = np.linalg.norm(direction)

            # 检查障碍物
            repulsion = np.zeros(3)
            for obs in obstacles:
                obs_dir = pos - obs
                obs_dist = np.linalg.norm(obs_dir)
                if obs_dist < 3.0 and obs_dist > 0.01:
                    repulsion += obs_dir / (obs_dist * obs_dist) * 2.0

            if distance > 0.5:
                target_vel = direction / distance * 0.8 + repulsion
            else:
                target_vel = repulsion
            action = np.clip(target_vel, -1, 1)

        elif task == "land":
            # 降落
            if pos[2] > 0.5:
                target_vel = np.array([0, 0, -0.5])
            else:
                target_vel = np.zeros(3)
            action = np.clip(target_vel, -1, 1)

        elif task == "takeoff":
            # 起飞
            if pos[2] < 5.0:
                target_vel = np.array([0, 0, 0.5])
            else:
                target_vel = np.zeros(3)
            action = np.clip(target_vel, -1, 1)
        else:
            action = np.zeros(3)

        # 添加yaw控制
        yaw_action = 0.0
        if task in ["navigate", "follow", "avoid"]:
            direction = goal_pos - pos
            target_yaw = np.arctan2(direction[1], direction[0])
            current_yaw = state[8]
            yaw_error = target_yaw - current_yaw
            yaw_action = np.clip(yaw_error * 0.5, -1, 1)

        action_4d = np.array([action[0], action[1], action[2], yaw_action])

        # 更新状态（简化动力学）
        dt = 0.1
        new_vel = vel + action_4d[:3] * dt * 5.0  # 加速度
        new_vel = np.clip(new_vel, -2, 2)  # 速度限制
        new_pos = pos + new_vel * dt

        # 边界约束
        new_pos = np.clip(new_pos, 0, 20)
        new_pos[2] = max(0, new_pos[2])  # 不能低于地面

        # 更新状态
        state[:3] = new_pos
        state[3:6] = new_vel
        state[8] += yaw_action * dt  # yaw角

        # 生成图像
        image = generate_synthetic_image(new_pos, goal_pos, obstacles)

        states.append(state.copy())
        actions.append(action_4d.copy())
        images.append(image)

    return states, actions, images


def generate_episode(
    task: str,
    episode_id: int
) -> Dict:
    """
    生成一个完整的演示episode

    Args:
        task: 任务类型
        episode_id: episode编号

    Returns:
        episode数据字典
    """
    # 随机初始位置
    initial_pos = np.random.uniform(2, 18, size=3)
    initial_pos[2] = np.random.uniform(3, 10)  # 高度

    # 随机目标位置
    goal_pos = np.random.uniform(2, 18, size=3)
    goal_pos[2] = np.random.uniform(3, 10)

    # 确保初始位置和目标位置有一定距离
    while np.linalg.norm(goal_pos - initial_pos) < 5:
        goal_pos = np.random.uniform(2, 18, size=3)
        goal_pos[2] = np.random.uniform(3, 10)

    # 随机障碍物
    num_obstacles = np.random.randint(0, 4)
    obstacles = []
    for _ in range(num_obstacles):
        obs = np.random.uniform(3, 17, size=2)
        obs = np.append(obs, np.random.uniform(0, 5))  # 高度
        obstacles.append(obs)

    # 初始状态 [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
    initial_state = np.zeros(12)
    initial_state[:3] = initial_pos

    # 选择指令
    instruction = np.random.choice(TASKS[task]["instructions"])

    # 生成轨迹
    num_steps = np.random.randint(30, 60)
    states, actions, images = generate_expert_trajectory(
        task, initial_state, goal_pos, obstacles, num_steps
    )

    return {
        "task": task,
        "instruction": instruction,
        "images": np.array(images, dtype=np.float32),
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "goal_pos": goal_pos,
        "obstacles": obstacles,
        "initial_pos": initial_pos,
        "episode_id": episode_id
    }


def generate_dataset(
    num_episodes: int,
    output_dir: str,
    tasks: List[str] = None
):
    """
    生成完整数据集

    Args:
        num_episodes: 总episode数量
        output_dir: 输出目录
        tasks: 要包含的任务列表
    """
    if tasks is None:
        tasks = list(TASKS.keys())

    os.makedirs(output_dir, exist_ok=True)

    episodes = []
    episodes_per_task = num_episodes // len(tasks)

    print(f"生成 {num_episodes} 个演示...")
    print(f"任务: {tasks}")
    print(f"每个任务: ~{episodes_per_task} 个episode")

    episode_id = 0
    for task in tasks:
        task_count = episodes_per_task
        if task == tasks[-1]:
            # 最后一个任务补齐剩余
            task_count = num_episodes - episode_id

        for i in range(task_count):
            episode = generate_episode(task, episode_id)
            episodes.append(episode)
            episode_id += 1

            if (episode_id + 1) % 100 == 0:
                print(f"  已生成 {episode_id + 1}/{num_episodes} 个episode")

    # 保存数据
    print(f"\n保存数据到 {output_dir}...")

    # 保存为npz格式
    np.savez_compressed(
        os.path.join(output_dir, "demonstrations.npz"),
        episodes=episodes
    )

    # 保存元数据
    metadata = {
        "num_episodes": len(episodes),
        "tasks": tasks,
        "image_size": 64,
        "state_dim": 12,
        "action_dim": 4,
        "tasks_info": {task: {
            "count": sum(1 for e in episodes if e["task"] == task),
            "description": TASKS[task]["description"]
        } for task in tasks}
    }

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"数据集生成完成!")
    print(f"  总episode数: {len(episodes)}")
    print(f"  保存位置: {output_dir}")
    print(f"  文件大小: {os.path.getsize(os.path.join(output_dir, 'demonstrations.npz')) / 1024 / 1024:.1f} MB")

    return episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DroneVLA 合成数据集生成器")
    parser.add_argument("--num_episodes", type=int, default=1000, help="生成episode数量")
    parser.add_argument("--output", type=str, default="data/train", help="输出目录")
    parser.add_argument("--tasks", nargs="+", default=None, help="要包含的任务")

    args = parser.parse_args()

    generate_dataset(
        num_episodes=args.num_episodes,
        output_dir=args.output,
        tasks=args.tasks
    )
