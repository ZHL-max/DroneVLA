"""
DroneVLA 轻量级评估脚本

使用训练好的轻量级模型在仿真中评估

使用方法：
    python scripts/evaluate_lightweight.py --model logs/best_lightweight.pt --episodes 30
"""

import torch
import numpy as np
import os
import argparse
import sys

sys.path.insert(0, '.')
from scripts.train_lightweight import SimpleDroneVLA


def generate_image(drone_pos, goal_pos, obstacles, size=64):
    """生成合成图像"""
    img = np.ones((size, size, 3), dtype=np.float32) * 0.3

    # 网格
    for i in range(0, size, 8):
        img[i, :] = 0.25
        img[:, i] = 0.25

    # 障碍物
    for obs in obstacles:
        px = int(np.clip(obs[0] * size / 20, 2, size - 6))
        py = int(np.clip(obs[1] * size / 20, 2, size - 6))
        img[px:px+4, py:py+4] = 0.1

    # 目标（红色）
    gx = int(np.clip(goal_pos[0] * size / 20, 4, size - 5))
    gy = int(np.clip(goal_pos[1] * size / 20, 4, size - 5))
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx*dx + dy*dy <= 9:
                px, py = gx + dx, gy + dy
                if 0 <= px < size and 0 <= py < size:
                    img[px, py] = [1.0, 0.2, 0.2]

    # 无人机（蓝色）
    dx = int(np.clip(drone_pos[0] * size / 20, 4, size - 5))
    dy = int(np.clip(drone_pos[1] * size / 20, 4, size - 5))
    img[dx-2:dx+3, dy-2:dy+3] = [0.2, 0.2, 1.0]

    return img


def evaluate_episode(model, task="navigate", max_steps=80):
    """评估单个episode"""
    initial_pos = np.random.uniform(2, 18, size=3)
    initial_pos[2] = np.random.uniform(3, 10)
    goal_pos = np.random.uniform(2, 18, size=3)
    goal_pos[2] = np.random.uniform(3, 10)
    while np.linalg.norm(goal_pos - initial_pos) < 5:
        goal_pos = np.random.uniform(2, 18, size=3)
        goal_pos[2] = np.random.uniform(3, 10)

    num_obs = np.random.randint(0, 3)
    obstacles = [np.random.uniform(3, 17, size=3) for _ in range(num_obs)]

    instructions = {
        "navigate": "fly to the red building",
        "hover": "hover at the current position",
        "avoid": "avoid the obstacles",
        "land": "land at the designated area"
    }
    instruction = instructions.get(task, "navigate to the target")

    state = np.zeros(12, dtype=np.float32)
    state[:3] = initial_pos

    model.eval()
    with torch.no_grad():
        for step in range(max_steps):
            img = generate_image(state[:3], goal_pos, obstacles)
            img_tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            action = model(img_tensor, instruction, state_tensor).numpy()[0]

            dt = 0.1
            vel = state[3:6] + action[:3] * dt * 5.0
            vel = np.clip(vel, -2, 2)
            pos = state[:3] + vel * dt
            pos = np.clip(pos, 0, 20)
            pos[2] = max(0, pos[2])

            state[:3] = pos
            state[3:6] = vel
            state[8] += action[3] * dt

            distance = np.linalg.norm(pos - goal_pos)
            if distance < 1.5:
                return True, step + 1, distance

    return False, max_steps, distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="logs/best_lightweight.pt")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--tasks", nargs="+", default=["navigate", "avoid"])

    args = parser.parse_args()

    print("=" * 60)
    print("DroneVLA 轻量级评估")
    print("=" * 60)

    # 加载模型
    model = SimpleDroneVLA()
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, weights_only=True))
        print(f"模型加载成功: {args.model}")
    else:
        print(f"模型不存在: {args.model}, 使用未训练模型")

    # 评估
    for task in args.tasks:
        print(f"\n评估任务: {task}")
        successes = 0
        total_steps = 0
        episodes = args.episodes // len(args.tasks)

        for ep in range(episodes):
            success, steps, dist = evaluate_episode(model, task)
            if success:
                successes += 1
            total_steps += steps

            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep+1}/{episodes} | "
                      f"Success: {successes}/{ep+1} ({successes/(ep+1)*100:.0f}%)")

        success_rate = successes / episodes * 100
        avg_steps = total_steps / episodes
        print(f"\n  {task} 结果:")
        print(f"    成功率: {success_rate:.1f}%")
        print(f"    平均步数: {avg_steps:.1f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
