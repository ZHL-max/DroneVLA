"""
DroneVLA 训练脚本

使用合成数据集训练完整的VLA模型

使用方法：
    python scripts/train.py --data data/train --epochs 100 --batch_size 32
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import argparse
import json
from datetime import datetime
import sys

sys.path.insert(0, '.')
from src.models.drone_vla import DroneVLA


class DroneVLADataset(Dataset):
    """DroneVLA训练数据集"""

    def __init__(self, data_dir: str, num_frames: int = 4, action_horizon: int = 8):
        self.num_frames = num_frames
        self.action_horizon = action_horizon

        # 加载数据
        data_file = os.path.join(data_dir, "demonstrations.npz")
        data = np.load(data_file, allow_pickle=True)
        self.episodes = data["episodes"]

        # 构建样本索引
        self.samples = []
        for ep_idx, episode in enumerate(self.episodes):
            ep_len = len(episode["actions"])
            for step_idx in range(num_frames, ep_len - action_horizon):
                self.samples.append((ep_idx, step_idx))

        print(f"加载了 {len(self.samples)} 个训练样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, step_idx = self.samples[idx]
        episode = self.episodes[ep_idx]

        # 图像序列 [num_frames, C, H, W]
        images = episode["images"][step_idx - self.num_frames:step_idx]
        images = torch.FloatTensor(images).permute(0, 3, 1, 2)

        # 状态 [state_dim]
        state = torch.FloatTensor(episode["states"][step_idx])

        # 动作序列 [action_horizon, action_dim]
        actions = torch.FloatTensor(
            episode["actions"][step_idx:step_idx + self.action_horizon]
        )
        # 填充或截断到固定长度
        if len(actions) < self.action_horizon:
            pad = torch.zeros(self.action_horizon - len(actions), actions.shape[-1])
            actions = torch.cat([actions, pad], dim=0)
        actions = actions[:self.action_horizon]

        # 指令
        instruction = episode["instruction"]

        return {
            "images": images,
            "state": state,
            "actions": actions,
            "instruction": instruction
        }


def collate_fn(batch):
    """自定义batch整理"""
    images = torch.stack([b["images"] for b in batch])
    states = torch.stack([b["state"] for b in batch])
    actions = torch.stack([b["actions"] for b in batch])
    instructions = [b["instruction"] for b in batch]

    return {
        "images": images,
        "states": states,
        "actions": actions,
        "instructions": instructions
    }


def train_epoch(model, dataloader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        images = batch["images"].to(device)
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        instructions = batch["instructions"]

        # 前向传播
        outputs = model(images, instructions, states, actions)

        # 计算损失
        predicted_actions = outputs["actions"]  # [B, action_dim]
        # 只用目标动作序列的第一个动作进行比较
        target_action = actions[:, 0, :]  # [B, action_dim]
        loss = nn.MSELoss()(predicted_actions, target_action)

        # 世界模型损失（如果存在）
        if "future_states" in outputs and "future_rewards" in outputs:
            # 简单的一步预测损失
            future_states = outputs["future_states"]
            if future_states.dim() == 3:
                # 使用下一个状态作为目标
                next_states = states.unsqueeze(1).expand_as(future_states[:, :1, :])
                wm_loss = nn.MSELoss()(future_states[:, :1, :], next_states)
                loss = loss + 0.1 * wm_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            instructions = batch["instructions"]

            outputs = model(images, instructions, states, actions)
            predicted_actions = outputs["actions"]  # [B, action_dim]
            target_action = actions[:, 0, :]  # [B, action_dim]
            loss = nn.MSELoss()(predicted_actions, target_action)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="DroneVLA 训练")
    parser.add_argument("--data", type=str, default="data/train", help="数据目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_frames", type=int, default=4, help="输入帧数")
    parser.add_argument("--action_horizon", type=int, default=8, help="动作序列长度")
    parser.add_argument("--save_dir", type=str, default="logs", help="保存目录")
    parser.add_argument("--use_world_model", action="store_true", help="使用世界模型")
    parser.add_argument("--device", type=str, default="auto", help="设备")

    args = parser.parse_args()

    # 设备选择
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("DroneVLA 训练")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"数据目录: {args.data}")
    print(f"训练轮数: {args.epochs}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.lr}")

    # 加载数据
    print("\n加载数据集...")
    dataset = DroneVLADataset(
        args.data,
        num_frames=args.num_frames,
        action_horizon=args.action_horizon
    )

    # 划分训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_dataset)}")

    # 创建模型
    print("\n创建模型...")
    model = DroneVLA(
        visual_dim=256,
        language_dim=256,
        state_dim=12,
        state_embed_dim=128,
        action_dim=4,
        action_horizon=args.action_horizon,
        use_world_model=args.use_world_model,
        action_mode='deterministic'
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    print("\n开始训练...")
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.save_dir, 'best_model.pt'))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, os.path.join(args.save_dir, 'final_model.pt'))

    # 保存训练配置
    config = {
        'visual_dim': 256,
        'language_dim': 256,
        'state_dim': 12,
        'state_embed_dim': 128,
        'action_dim': 4,
        'action_horizon': args.action_horizon,
        'use_world_model': args.use_world_model,
        'action_mode': 'deterministic',
        'total_params': total_params,
        'best_val_loss': best_val_loss,
        'epochs': args.epochs
    }
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存位置: {args.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
