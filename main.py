"""
DroneVLA 主程序

使用示例：
    # 收集演示数据
    python main.py --mode collect --task hover --num_episodes 1000

    # 训练模型
    python main.py --mode train --config configs/default.yaml

    # 评估模型
    python main.py --mode evaluate --model logs/best_model.pt

    # 推理演示
    python main.py --mode demo --model logs/best_model.pt

作者：DroneVLA Project
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from src.environments.drone_env import DroneEnv, DroneLanguageEnv
from src.models.drone_vla import DroneVLA
from src.training.trainer import DroneVLATrainer, collect_demonstrations, DemonstrationDataset


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_data(args, config):
    """收集演示数据"""
    print("=" * 60)
    print("收集演示数据")
    print("=" * 60)

    # 创建环境
    env_config = config.get("environment", {})
    env = DroneLanguageEnv(
        task=args.task or env_config.get("task", "hover"),
        image_obs=env_config.get("image_obs", True),
        image_size=tuple(env_config.get("image_size", [64, 64])),
        max_steps=env_config.get("max_steps", 200)
    )

    # 收集数据
    num_episodes = args.num_episodes or config.get("data", {}).get("num_demonstrations", 100)
    save_dir = args.save_dir or "data/demos"

    episodes = collect_demonstrations(env, num_episodes=num_episodes, save_dir=save_dir)

    print(f"\n收集完成！")
    print(f"  Episodes: {len(episodes)}")
    print(f"  保存位置: {save_dir}")

    env.close()


def train_model(args, config):
    """训练模型"""
    print("=" * 60)
    print("训练DroneVLA模型")
    print("=" * 60)

    # 设置设备
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    model_config = config.get("model", {})
    model = DroneVLA(
        visual_dim=model_config.get("visual_dim", 256),
        language_dim=model_config.get("language_dim", 256),
        state_dim=model_config.get("state_dim", 12),
        state_embed_dim=model_config.get("state_embed_dim", 128),
        action_dim=model_config.get("action_dim", 4),
        action_horizon=model_config.get("action_horizon", 8),
        use_world_model=model_config.get("use_world_model", True),
        action_mode=model_config.get("action_mode", "deterministic")
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建训练器
    training_config = config.get("training", {})
    trainer = DroneVLATrainer(
        model=model,
        config={
            "batch_size": training_config.get("batch_size", 32),
            "learning_rate": training_config.get("learning_rate", 1e-4),
            "weight_decay": training_config.get("weight_decay", 1e-5),
            "num_epochs": training_config.get("num_epochs", 100),
            "log_dir": config.get("logging", {}).get("log_dir", "logs")
        },
        device=device
    )

    # 加载数据
    data_dir = args.data_dir or "data/demos"
    dataset = DemonstrationDataset(data_dir)

    # 划分数据集
    data_config = config.get("data", {})
    train_size = int(len(dataset) * data_config.get("train_split", 0.8))
    val_size = int(len(dataset) * data_config.get("val_split", 0.1))

    train_dataset, val_dataset, _ = torch.utils.data.random_split(
        dataset, [train_size, val_size, len(dataset) - train_size - val_size]
    )

    print(f"训练集: {len(train_dataset)}")
    print(f"验证集: {len(val_dataset)}")

    # 训练
    num_epochs = training_config.get("num_epochs", 100)

    # 阶段1：模仿学习
    trainer.train_imitation(train_dataset, val_dataset, num_epochs=num_epochs)

    # 阶段2：世界模型训练
    if model_config.get("use_world_model", True):
        trainer.train_world_model(train_dataset, num_epochs=50)

    print("\n训练完成！")


def evaluate_model(args, config):
    """评估模型"""
    print("=" * 60)
    print("评估DroneVLA模型")
    print("=" * 60)

    # 设置设备
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model_config = config.get("model", {})
    model = DroneVLA(
        visual_dim=model_config.get("visual_dim", 256),
        language_dim=model_config.get("language_dim", 256),
        state_dim=model_config.get("state_dim", 12),
        state_embed_dim=model_config.get("state_embed_dim", 128),
        action_dim=model_config.get("action_dim", 4),
        action_horizon=model_config.get("action_horizon", 8),
        use_world_model=model_config.get("use_world_model", True),
        action_mode=model_config.get("action_mode", "deterministic")
    )

    # 加载模型
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 创建环境
    env_config = config.get("environment", {})
    env = DroneLanguageEnv(
        task=env_config.get("task", "hover"),
        image_obs=env_config.get("image_obs", True),
        image_size=tuple(env_config.get("image_size", [64, 64])),
        max_steps=env_config.get("max_steps", 200)
    )

    # 评估
    eval_config = config.get("evaluation", {})
    num_episodes = eval_config.get("num_episodes", 100)
    success_threshold = eval_config.get("success_threshold", 0.3)

    successes = 0
    total_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        instruction = info.get("instruction", "")
        total_reward = 0
        done = False

        while not done:
            # 准备输入
            if isinstance(obs, dict):
                images = torch.FloatTensor(obs["image"]).unsqueeze(0).unsqueeze(0)
                state = torch.FloatTensor(obs["state"]).unsqueeze(0)
            else:
                images = torch.zeros(1, 1, 3, 64, 64)
                state = torch.FloatTensor(obs).unsqueeze(0)

            images = images.to(device)
            state = state.to(device)

            # 推理
            with torch.no_grad():
                outputs = model(images, [instruction], state)
                action = outputs["actions"][0].cpu().numpy()

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # 检查是否成功
        if info.get("distance_to_goal", float("inf")) < success_threshold:
            successes += 1

        total_rewards.append(total_reward)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{num_episodes}")

    # 输出结果
    success_rate = successes / num_episodes
    avg_reward = np.mean(total_rewards)

    print(f"\n评估结果:")
    print(f"  成功率: {success_rate:.2%}")
    print(f"  平均奖励: {avg_reward:.4f}")
    print(f"  评估Episodes: {num_episodes}")

    env.close()


def demo_mode(args, config):
    """演示模式"""
    print("=" * 60)
    print("DroneVLA 演示模式")
    print("=" * 60)

    # 设置设备
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model_config = config.get("model", {})
    model = DroneVLA(
        visual_dim=model_config.get("visual_dim", 256),
        language_dim=model_config.get("language_dim", 256),
        state_dim=model_config.get("state_dim", 12),
        state_embed_dim=model_config.get("state_embed_dim", 128),
        action_dim=model_config.get("action_dim", 4),
        action_horizon=model_config.get("action_horizon", 8),
        use_world_model=model_config.get("use_world_model", True),
        action_mode=model_config.get("action_mode", "deterministic")
    )

    # 加载模型
    if args.model:
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"加载模型: {args.model}")

    model = model.to(device)
    model.eval()

    # 创建环境（带渲染）
    env_config = config.get("environment", {})
    env = DroneLanguageEnv(
        task=env_config.get("task", "hover"),
        render_mode="human",
        image_obs=env_config.get("image_obs", True),
        image_size=tuple(env_config.get("image_size", [64, 64])),
        max_steps=env_config.get("max_steps", 200)
    )

    # 运行演示
    print("\n开始演示...")
    print("按 Ctrl+C 退出\n")

    try:
        while True:
            obs, info = env.reset()
            instruction = info.get("instruction", "")
            print(f"指令: {instruction}")

            done = False
            step = 0

            while not done:
                # 准备输入
                if isinstance(obs, dict):
                    images = torch.FloatTensor(obs["image"]).unsqueeze(0).unsqueeze(0)
                    state = torch.FloatTensor(obs["state"]).unsqueeze(0)
                else:
                    images = torch.zeros(1, 1, 3, 64, 64)
                    state = torch.FloatTensor(obs).unsqueeze(0)

                images = images.to(device)
                state = state.to(device)

                # 推理
                with torch.no_grad():
                    outputs = model(images, [instruction], state)
                    action = outputs["actions"][0].cpu().numpy()

                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

                if step % 10 == 0:
                    print(f"  Step {step}, Distance: {info.get('distance_to_goal', 0):.2f}")

            print(f"  完成！总步数: {step}\n")

    except KeyboardInterrupt:
        print("\n演示结束")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="DroneVLA: 无人机视觉-语言-动作模型")

    parser.add_argument("--mode", type=str, required=True,
                       choices=["collect", "train", "evaluate", "demo"],
                       help="运行模式")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="配置文件路径")
    parser.add_argument("--task", type=str, default=None,
                       choices=["hover", "navigate", "track", "avoid"],
                       help="任务类型")
    parser.add_argument("--num_episodes", type=int, default=None,
                       help="收集的episode数量")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="数据保存目录")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="训练数据目录")
    parser.add_argument("--model", type=str, default=None,
                       help="模型路径")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置种子
    set_seed(args.seed or config.get("seed", 42))

    # 运行对应模式
    if args.mode == "collect":
        collect_data(args, config)
    elif args.mode == "train":
        train_model(args, config)
    elif args.mode == "evaluate":
        evaluate_model(args, config)
    elif args.mode == "demo":
        demo_mode(args, config)
    else:
        print(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()
