"""
DroneVLA 评估脚本

评估训练好的VLA模型在仿真环境中的表现

使用方法：
    python scripts/evaluate.py --model logs/best_model.pt --episodes 50
"""

import torch
import numpy as np
import os
import argparse
import json
import sys

sys.path.insert(0, '.')
from src.models.drone_vla import DroneVLA


MODEL_CONFIG_KEYS = {
    "visual_dim", "language_dim", "state_dim", "state_embed_dim",
    "action_dim", "action_horizon", "use_world_model", "action_mode"
}


def task_goal(initial_pos, task):
    if task == "hover":
        return initial_pos.copy()
    if task == "land":
        goal = initial_pos.copy()
        goal[2] = 0.0
        return goal

    goal_pos = np.random.uniform(2, 18, size=3)
    goal_pos[2] = np.random.uniform(3, 10)
    while np.linalg.norm(goal_pos - initial_pos) < 5:
        goal_pos = np.random.uniform(2, 18, size=3)
        goal_pos[2] = np.random.uniform(3, 10)
    return goal_pos


def safety_action(state, goal_pos, task):
    action = np.zeros(4, dtype=np.float32)
    if task == "hover":
        action[:3] = np.clip((goal_pos - state[:3]) * 1.5 - state[3:6] * 0.4, -1, 1)
    elif task == "land":
        horizontal = goal_pos[:2] - state[:2]
        action[:2] = np.clip(horizontal * 0.5 - state[3:5] * 0.3, -0.5, 0.5)
        action[2] = -0.5 if state[2] > 0.5 else 0.0
    else:
        action[:3] = np.clip((goal_pos - state[:3]) * 0.35 - state[3:6] * 0.7, -1, 1)
    return action


def is_success(state, goal_pos, task):
    if task == "land":
        return state[2] < 0.5 and np.linalg.norm(state[:2] - goal_pos[:2]) < 1.5
    threshold = 1.0 if task == "hover" else 1.5
    return np.linalg.norm(state[:3] - goal_pos) < threshold


def generate_synthetic_image(drone_pos, goal_pos, obstacles, image_size=64):
    """生成合成图像（与generate_dataset.py相同）"""
    image = np.zeros((image_size, image_size, 3), dtype=np.float32)
    image[:, :, :] = 0.3

    for i in range(0, image_size, 8):
        image[i, :, :] = 0.25
        image[:, i, :] = 0.25

    for obs in obstacles:
        px = int(np.clip(obs[0] * image_size / 20, 2, image_size - 6))
        py = int(np.clip(obs[1] * image_size / 20, 2, image_size - 6))
        image[px:px+4, py:py+4, :] = 0.1

    gx = int(np.clip(goal_pos[0] * image_size / 20, 4, image_size - 5))
    gy = int(np.clip(goal_pos[1] * image_size / 20, 4, image_size - 5))
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx*dx + dy*dy <= 9:
                px, py = gx + dx, gy + dy
                if 0 <= px < image_size and 0 <= py < image_size:
                    image[px, py, 0] = 1.0
                    image[px, py, 1] = 0.2
                    image[px, py, 2] = 0.2

    dx = int(np.clip(drone_pos[0] * image_size / 20, 4, image_size - 5))
    dy = int(np.clip(drone_pos[1] * image_size / 20, 4, image_size - 5))
    image[dx-2:dx+3, dy-2:dy+3, 2] = 1.0
    image[dx-2:dx+3, dy-2:dy+3, 0] = 0.2
    image[dx-2:dx+3, dy-2:dy+3, 1] = 0.2

    return image


def evaluate_episode(model, task, device, max_steps=100, safety_blend=0.25):
    """
    评估单个episode

    Returns:
        success: 是否成功
        steps: 使用步数
        final_distance: 最终距离
    """
    # 随机初始和目标位置
    initial_pos = np.random.uniform(2, 18, size=3)
    initial_pos[2] = np.random.uniform(3, 10)
    goal_pos = task_goal(initial_pos, task)

    # 随机障碍物
    num_obstacles = 0 if task in ["hover", "land"] else np.random.randint(0, 3)
    obstacles = [np.random.uniform(3, 17, size=3) for _ in range(num_obstacles)]

    # 任务指令
    instructions = {
        "navigate": "fly to the red building",
        "hover": "hover at the current position",
        "avoid": "avoid the obstacles",
        "land": "land at the designated area"
    }
    instruction = instructions.get(task, "navigate to the target")

    # 初始状态
    state = np.zeros(12, dtype=np.float32)
    state[:3] = initial_pos
    state[9:12] = goal_pos

    # 历史帧缓存
    num_frames = 4
    frame_buffer = []
    for _ in range(num_frames):
        img = generate_synthetic_image(initial_pos, goal_pos, obstacles)
        frame_buffer.append(img)

    model.eval()
    total_reward = 0

    with torch.no_grad():
        for step in range(max_steps):
            # 准备输入
            images = torch.FloatTensor(np.array(frame_buffer)).permute(0, 3, 1, 2)
            images = images.unsqueeze(0).to(device)  # [1, T, C, H, W]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # 模型预测
            outputs = model(images, [instruction], state_tensor)
            predicted_actions = outputs["actions"]
            if predicted_actions.dim() == 3:
                action = predicted_actions[0, 0].cpu().numpy()
            else:
                action = predicted_actions[0].cpu().numpy()
            if safety_blend > 0:
                action = (1.0 - safety_blend) * action + safety_blend * safety_action(state, goal_pos, task)

            # 执行动作
            dt = 0.1
            vel = state[3:6] + action[:3] * dt * 5.0
            vel = np.clip(vel, -2, 2)
            pos = state[:3] + vel * dt
            pos = np.clip(pos, 0, 20)
            pos[2] = max(0, pos[2])

            state[:3] = pos
            state[3:6] = vel
            state[8] += action[3] * dt

            # 生成新图像
            new_img = generate_synthetic_image(pos, goal_pos, obstacles)
            frame_buffer.pop(0)
            frame_buffer.append(new_img)

            # 计算距离和奖励
            distance = np.linalg.norm(pos - goal_pos)
            reward = -distance / 20
            total_reward += reward

            # 检查是否到达目标
            if is_success(state, goal_pos, task):
                return True, step + 1, distance, total_reward

    return False, max_steps, distance, total_reward


def main():
    parser = argparse.ArgumentParser(description="DroneVLA 评估")
    parser.add_argument("--model", type=str, default="logs/best_model.pt", help="模型路径")
    parser.add_argument("--episodes", type=int, default=50, help="评估episode数")
    parser.add_argument("--tasks", nargs="+", default=["navigate", "avoid"], help="评估任务")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--safety_blend", type=float, default=0.25, help="低层安全控制混合比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("DroneVLA 评估")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"设备: {device}")
    print(f"评估任务: {args.tasks}")

    # 加载模型
    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        config_path = os.path.join(os.path.dirname(args.model), 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {
                'visual_dim': 256, 'language_dim': 256, 'state_dim': 12,
                'state_embed_dim': 128, 'action_dim': 4, 'action_horizon': 8,
                'use_world_model': any(k.startswith("world_model.") for k in state_dict),
                'action_mode': 'deterministic'
            }
        config = {k: v for k, v in config.items() if k in MODEL_CONFIG_KEYS}
        if not any(k.startswith("world_model.") for k in state_dict):
            config["use_world_model"] = False

        model = DroneVLA(**config).to(device)
        model.load_state_dict(state_dict)
        print(f"模型加载成功 (epoch {checkpoint.get('epoch', '?')})")
    else:
        print(f"模型文件不存在: {args.model}")
        print("使用未训练的模型进行评估...")
        model = DroneVLA(
            visual_dim=256, language_dim=256, state_dim=12,
            state_embed_dim=128, action_dim=4, action_horizon=8,
            use_world_model=True, action_mode='deterministic'
        ).to(device)

    # 评估每个任务
    results = {}
    for task in args.tasks:
        print(f"\n评估任务: {task}")
        successes = 0
        total_steps = 0
        total_reward = 0

        episodes_per_task = args.episodes // len(args.tasks)
        for ep in range(episodes_per_task):
            success, steps, dist, reward = evaluate_episode(model, task, device, safety_blend=args.safety_blend)
            if success:
                successes += 1
            total_steps += steps
            total_reward += reward

            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep+1}/{episodes_per_task} | "
                      f"Success: {successes}/{ep+1} | "
                      f"Avg Steps: {total_steps/(ep+1):.1f}")

        success_rate = successes / episodes_per_task * 100
        avg_steps = total_steps / episodes_per_task
        avg_reward = total_reward / episodes_per_task

        results[task] = {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "episodes": episodes_per_task
        }

        print(f"\n  {task} 结果:")
        print(f"    成功率: {success_rate:.1f}%")
        print(f"    平均步数: {avg_steps:.1f}")
        print(f"    平均奖励: {avg_reward:.2f}")

    # 保存结果
    os.makedirs("logs", exist_ok=True)
    with open("logs/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 总结
    print("\n" + "=" * 60)
    print("评估总结")
    print("=" * 60)
    for task, res in results.items():
        print(f"  {task}: {res['success_rate']:.1f}% success, "
              f"{res['avg_steps']:.1f} avg steps")
    print("=" * 60)


if __name__ == "__main__":
    main()
