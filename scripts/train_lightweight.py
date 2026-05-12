"""
DroneVLA 轻量级训练脚本

使用简化的模型架构，适合CPU快速验证

使用方法：
    python scripts/train_lightweight.py --data data/train --epochs 20
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

sys.path.insert(0, '.')


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


class SimpleDroneVLA(nn.Module):
    """轻量级DroneVLA模型，适合CPU训练"""

    def __init__(self, state_dim=12, action_dim=4, embed_dim=64):
        super().__init__()

        # 简单的视觉编码器
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embed_dim)
        )

        # 简单的语言编码器
        self.language_encoder = nn.Sequential(
            nn.Embedding(100, embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

        # 动作解码器
        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

        self.word2idx = VOCAB

    def encode_instruction(self, instruction):
        """简单的词袋编码"""
        tokens = []
        for word in instruction.lower().split():
            tokens.append(self.word2idx.get(word, self.word2idx["<unk>"]))
        if not tokens:
            tokens = [self.word2idx["<pad>"]]
        return torch.LongTensor(tokens)

    def forward(self, image, instruction, state):
        """
        Args:
            image: [B, 3, H, W]
            instruction: str or list of str
            state: [B, state_dim]
        """
        B = image.shape[0]

        # 视觉编码
        visual_feat = self.visual_encoder(image)  # [B, embed_dim]

        # 语言编码
        if isinstance(instruction, str):
            instruction = [instruction] * B
        tokens = [self.encode_instruction(inst) for inst in instruction]
        # 填充到相同长度
        max_len = max(len(t) for t in tokens)
        padded = torch.zeros(B, max_len, dtype=torch.long, device=image.device)
        for i, t in enumerate(tokens):
            padded[i, :len(t)] = t.to(image.device)
        lang_embed = self.language_encoder(padded)  # [B, seq_len, embed_dim]
        lang_feat = lang_embed.mean(dim=1)  # [B, embed_dim]

        # 状态编码
        state_feat = self.state_encoder(state)  # [B, embed_dim]

        # 融合
        combined = torch.cat([visual_feat, lang_feat, state_feat], dim=-1)
        action = self.action_decoder(combined)

        return action


class SimpleDataset(Dataset):
    """简化的数据集"""

    def __init__(self, data_dir):
        data = np.load(os.path.join(data_dir, "demonstrations.npz"), allow_pickle=True)
        self.episodes = data["episodes"]

        self.samples = []
        for ep_idx, ep in enumerate(self.episodes):
            for step in range(4, len(ep["actions"])):
                self.samples.append((ep_idx, step))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, step = self.samples[idx]
        ep = self.episodes[ep_idx]

        image = torch.FloatTensor(ep["images"][step]).permute(2, 0, 1)  # [C, H, W]
        state = torch.FloatTensor(ep["states"][step])
        action = torch.FloatTensor(ep["actions"][step])
        instruction = ep["instruction"]

        return image, state, action, instruction


def main():
    parser = argparse.ArgumentParser(description="DroneVLA 轻量级训练")
    parser.add_argument("--data", type=str, default="data/train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="logs")

    args = parser.parse_args()

    print("=" * 60)
    print("DroneVLA 轻量级训练")
    print("=" * 60)

    # 加载数据
    print("\n加载数据集...")
    dataset = SimpleDataset(args.data)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"训练样本: {train_size}, 验证样本: {val_size}")

    # 创建模型
    model = SimpleDroneVLA()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 训练
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0
        for images, states, actions, instructions in train_loader:
            pred_actions = model(images, instructions, states)
            loss = criterion(pred_actions, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, states, actions, instructions in val_loader:
                pred_actions = model(images, instructions, states)
                loss = criterion(pred_actions, actions)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": VOCAB,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, os.path.join(args.save_dir, "best_lightweight.pt"))

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存: {args.save_dir}/best_lightweight.pt")


if __name__ == "__main__":
    main()
