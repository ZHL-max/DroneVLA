"""
DroneVLA 训练器

支持：
- 模仿学习预训练
- 世界模型训练
- 强化学习微调
- 闭环训练

作者：DroneVLA Project
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Optional, Tuple
import os
import json
from datetime import datetime


class DemonstrationDataset(Dataset):
    """
    演示数据集

    存储和加载专家演示数据
    """

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (64, 64),
        num_frames: int = 4,
        action_horizon: int = 8
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.num_frames = num_frames
        self.action_horizon = action_horizon

        # 加载数据
        self.episodes = self._load_episodes()

    def _load_episodes(self):
        """加载所有演示数据"""
        episodes = []

        data_file = os.path.join(self.data_dir, "demonstrations.npz")
        if os.path.exists(data_file):
            data = np.load(data_file, allow_pickle=True)
            episodes = data["episodes"]

        return episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode = self.episodes[idx]

        # 获取图像序列
        images = episode["images"][-self.num_frames:]
        images = torch.FloatTensor(images).permute(0, 3, 1, 2)  # [T, C, H, W]

        # 获取状态
        state = torch.FloatTensor(episode["states"][-1])

        # 获取动作序列
        actions = torch.FloatTensor(episode["actions"][:self.action_horizon])

        # 获取指令
        instruction = episode["instruction"]

        return {
            "images": images,
            "state": state,
            "actions": actions,
            "instruction": instruction
        }


class DroneVLATrainer:
    """
    DroneVLA训练器

    支持多种训练模式：
    1. 模仿学习预训练
    2. 世界模型训练
    3. 强化学习微调
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5)
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("num_epochs", 100),
            eta_min=1e-6
        )

        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        # 日志
        self.log_dir = config.get("log_dir", "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def train_imitation(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        num_epochs: int = 100
    ):
        """
        模仿学习预训练

        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            num_epochs: 训练轮数
        """
        print("=" * 60)
        print("开始模仿学习预训练")
        print("=" * 60)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
            num_workers=4
        )

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                # 移到设备
                images = batch["images"].to(self.device)
                state = batch["state"].to(self.device)
                actions = batch["actions"].to(self.device)
                instructions = batch["instruction"]

                # 前向传播
                outputs = self.model(images, instructions, state)

                # 计算损失
                targets = {"actions": actions}
                losses = self.model.compute_loss(outputs, targets)

                # 反向传播
                self.optimizer.zero_grad()
                losses["total_loss"].backward()

                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += losses["total_loss"].item()
                self.global_step += 1

            # 更新学习率
            self.scheduler.step()

            # 计算平均损失
            avg_loss = total_loss / len(train_loader)

            # 验证
            val_loss = None
            if val_dataset is not None:
                val_loss = self._validate(val_dataset)

            # 日志
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}")

            # 保存最佳模型
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self._save_checkpoint("best_model.pt")

            self.epoch += 1

    def train_world_model(
        self,
        train_dataset: Dataset,
        num_epochs: int = 50
    ):
        """
        训练世界模型

        Args:
            train_dataset: 训练数据集
            num_epochs: 训练轮数
        """
        print("=" * 60)
        print("开始世界模型训练")
        print("=" * 60)

        if not self.model.use_world_model:
            print("警告：模型未启用世界模型")
            return

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
            num_workers=4
        )

        # 只训练世界模型参数
        wm_params = list(self.model.world_model.parameters())
        wm_optimizer = optim.Adam(wm_params, lr=1e-4)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                images = batch["images"].to(self.device)
                state = batch["state"].to(self.device)
                actions = batch["actions"].to(self.device)
                instructions = batch["instruction"]

                # 计算未来状态（目标）
                # 简化：使用下一个状态作为目标
                future_states = state.unsqueeze(1).expand(-1, actions.shape[1], -1)
                future_rewards = torch.zeros(state.shape[0], actions.shape[1], 1).to(self.device)

                # 前向传播
                outputs = self.model(images, instructions, state, actions)

                # 计算损失
                targets = {
                    "future_states": future_states,
                    "future_rewards": future_rewards
                }
                losses = self.model.compute_loss(outputs, targets)

                # 只计算世界模型损失
                wm_loss = losses.get("state_loss", 0) + losses.get("reward_loss", 0)

                # 反向传播
                wm_optimizer.zero_grad()
                wm_loss.backward()
                wm_optimizer.step()

                total_loss += wm_loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, WM Loss: {avg_loss:.4f}")

    def _validate(self, val_dataset: Dataset) -> float:
        """验证"""
        self.model.eval()
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(self.device)
                state = batch["state"].to(self.device)
                actions = batch["actions"].to(self.device)
                instructions = batch["instruction"]

                outputs = self.model(images, instructions, state)
                targets = {"actions": actions}
                losses = self.model.compute_loss(outputs, targets)

                total_loss += losses["total_loss"].item()

        return total_loss / len(val_loader)

    def _save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config
        }

        path = os.path.join(self.log_dir, filename)
        torch.save(checkpoint, path)
        print(f"  保存检查点到 {path}")

    def load_checkpoint(self, filename: str):
        """加载检查点"""
        path = os.path.join(self.log_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]

        print(f"加载检查点从 {path}")
        print(f"  Epoch: {self.epoch}, Best Loss: {self.best_loss:.4f}")


def collect_demonstrations(
    env,
    num_episodes: int = 100,
    save_dir: str = "data/demos"
):
    """
    收集专家演示数据

    Args:
        env: 环境
        num_episodes: 收集的episode数量
        save_dir: 保存目录
    """
    print(f"收集 {num_episodes} 个episode的演示数据...")

    episodes = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        instruction = info.get("instruction", "")

        episode_data = {
            "images": [],
            "states": [],
            "actions": [],
            "instruction": instruction
        }

        done = False
        while not done:
            # 简单的专家策略（PID控制）
            state = obs["state"] if isinstance(obs, dict) else obs
            pos = state[:3]
            goal = state[9:12]

            # 计算控制动作
            error = goal - pos
            action = np.clip(error * 2.0, -1, 1)

            # 记录数据
            if isinstance(obs, dict) and "image" in obs:
                episode_data["images"].append(obs["image"])
            episode_data["states"].append(state)
            episode_data["actions"].append(action)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        episodes.append(episode_data)

        if (ep + 1) % 10 == 0:
            print(f"  完成 {ep+1}/{num_episodes}")

    # 保存数据
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "demonstrations.npz")
    np.savez(save_path, episodes=episodes)
    print(f"演示数据已保存到 {save_path}")

    return episodes
