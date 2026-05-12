"""
DroneVLA GPU训练脚本

支持完整模型的GPU训练，包含：
- 自动GPU检测
- 混合精度训练
- 实验日志
- 可视化输出

使用方法：
    python scripts/train_gpu.py --data data/train --epochs 100 --experiment iteration_01
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import argparse
import json
import sys
import time
from datetime import datetime

sys.path.insert(0, '.')


class DroneVLADataset(Dataset):
    """DroneVLA训练数据集"""

    def __init__(self, data_dir, num_frames=4):
        self.num_frames = num_frames
        data = np.load(os.path.join(data_dir, "demonstrations.npz"), allow_pickle=True)
        self.episodes = data["episodes"]

        self.samples = []
        for ep_idx, ep in enumerate(self.episodes):
            for step in range(num_frames, len(ep["actions"])):
                self.samples.append((ep_idx, step))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, step = self.samples[idx]
        ep = self.episodes[ep_idx]

        # 图像序列
        images = ep["images"][step-self.num_frames:step]
        images = torch.FloatTensor(images).permute(0, 3, 1, 2)  # [T, C, H, W]

        # 状态
        state = torch.FloatTensor(ep["states"][step])

        # 动作（只用当前步的动作）
        action = torch.FloatTensor(ep["actions"][step])

        # 指令
        instruction = ep["instruction"]

        return images, state, action, instruction


class SimpleVisualEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class SimpleLanguageEncoder(nn.Module):
    VOCAB = {
        "<pad>": 0, "<unk>": 1,
        "hover": 2, "at": 3, "the": 4, "current": 5, "position": 6,
        "stay": 7, "in": 8, "place": 9, "maintain": 10, "hold": 11,
        "steady": 12, "this": 13, "location": 14,
        "fly": 15, "to": 16, "red": 17, "building": 18, "navigate": 19,
        "target": 20, "go": 21, "waypoint": 22, "move": 23, "destination": 24,
        "follow": 25, "moving": 26, "object": 27, "track": 28, "keep": 29,
        "following": 30, "vehicle": 31, "pursue": 32,
        "avoid": 33, "obstacles": 34, "around": 35, "through": 36, "gap": 37,
        "dodge": 38, "ahead": 39,
        "land": 40, "designated": 41, "area": 42, "descend": 43, "landing": 44,
        "pad": 45, "perform": 46, "a": 47, "gentle": 48, "safely": 49,
        "on": 50, "ground": 51,
        "take": 52, "off": 53, "from": 54, "ascend": 55, "altitude": 56,
        "launch": 57, "and": 58, "reach": 59, "safe": 60, "height": 61,
        "rise": 62, "operating": 63,
    }

    def __init__(self, embed_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(100, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, instructions, device):
        tokens = []
        for inst in instructions:
            t = [self.VOCAB.get(w, 1) for w in inst.lower().split()]
            tokens.append(t if t else [0])

        max_len = max(len(t) for t in tokens)
        padded = torch.zeros(len(tokens), max_len, dtype=torch.long, device=device)
        for i, t in enumerate(tokens):
            padded[i, :len(t)] = torch.tensor(t, device=device)

        embeds = self.embedding(padded)
        return self.fc(embeds.mean(dim=1))


class DroneVLA_GPU(nn.Module):
    """支持GPU的DroneVLA模型"""

    def __init__(self, visual_dim=256, state_dim=12, action_dim=4):
        super().__init__()
        self.visual_encoder = SimpleVisualEncoder(visual_dim)
        self.language_encoder = SimpleLanguageEncoder(visual_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, visual_dim)
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(visual_dim * 3, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh()
        )

    def forward(self, images, state, instructions):
        B = images.shape[0]
        # 用最后一帧图像
        visual_feat = self.visual_encoder(images[:, -1])  # [B, visual_dim]
        lang_feat = self.language_encoder(instructions, images.device)
        state_feat = self.state_encoder(state)

        combined = torch.cat([visual_feat, lang_feat, state_feat], dim=-1)
        return self.action_decoder(combined)


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    for images, states, actions, instructions in loader:
        images = images.to(device)
        states = states.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast('cuda'):
                pred = model(images, states, instructions)
                loss = criterion(pred, actions)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(images, states, instructions)
            loss = criterion(pred, actions)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, states, actions, instructions in loader:
            images = images.to(device)
            states = states.to(device)
            actions = actions.to(device)
            pred = model(images, states, instructions)
            loss = criterion(pred, actions)
            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--experiment", default="iteration_01")
    args = parser.parse_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("DroneVLA GPU训练")
    print("=" * 60)
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

    # 实验目录
    exp_dir = f"experiments/{args.experiment}"
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{exp_dir}/visualizations", exist_ok=True)
    os.makedirs(f"{exp_dir}/logs", exist_ok=True)

    # 加载数据
    print("\n加载数据集...")
    dataset = DroneVLADataset(args.data)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"训练样本: {train_size}, 验证样本: {val_size}")

    # 创建模型
    model = DroneVLA_GPU().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    # 训练
    print(f"\n开始训练 ({args.epochs} epochs)...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, f"{exp_dir}/checkpoints/best_model.pt")

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"Time: {elapsed:.0f}s")

    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
    }, f"{exp_dir}/checkpoints/final_model.pt")

    # 保存训练曲线
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{exp_dir}/visualizations/training_curve.png", dpi=150)
    plt.close()

    # 保存指标
    metrics = {
        'total_params': total_params,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'epochs': args.epochs,
        'device': str(device),
        'training_time': time.time() - start_time,
    }
    with open(f"{exp_dir}/logs/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # 保存配置
    config = vars(args)
    config['total_params'] = total_params
    with open(f"{exp_dir}/config.yaml", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"训练时间: {time.time()-start_time:.0f}s")
    print(f"结果保存在: {exp_dir}/")


if __name__ == "__main__":
    main()
