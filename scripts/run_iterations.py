"""
DroneVLA 多轮迭代训练脚本

自动运行10轮训练迭代，每轮有明确的改进策略

使用方法：
    python scripts/run_iterations.py
    python scripts/run_iterations.py --start 2 --end 5
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
import time
import sys
import argparse
from datetime import datetime

sys.path.insert(0, '.')


# ============================================================
# 改进的数据生成器
# ============================================================

def generate_improved_dataset(data_dir, num_episodes=200, focus_task=None):
    """生成改进的训练数据，增加navigate/avoid任务的样本"""
    os.makedirs(data_dir, exist_ok=True)

    episodes = []
    task_counts = {"hover": 0, "navigate": 0, "avoid": 0, "land": 0}

    for i in range(num_episodes):
        # 任务分配：增加navigate和avoid的比例
        if focus_task:
            task = focus_task
        else:
            r = np.random.random()
            if r < 0.15:
                task = "hover"
            elif r < 0.45:
                task = "navigate"  # 增加比例
            elif r < 0.75:
                task = "avoid"  # 增加比例
            else:
                task = "land"

        task_counts[task] += 1

        # 初始状态
        init_pos = np.random.uniform(2, 18, size=3)
        init_pos[2] = np.random.uniform(3, 12)

        if task == "hover":
            goal_pos = init_pos.copy()
        elif task == "land":
            goal_pos = init_pos.copy()
            goal_pos[2] = 0.0
        elif task == "navigate":
            goal_pos = np.random.uniform(2, 18, size=3)
            goal_pos[2] = np.random.uniform(3, 12)
            # 确保距离足够远
            while np.linalg.norm(goal_pos - init_pos) < 4:
                goal_pos = np.random.uniform(2, 18, size=3)
                goal_pos[2] = np.random.uniform(3, 12)
        else:  # avoid
            goal_pos = np.random.uniform(2, 18, size=3)
            goal_pos[2] = np.random.uniform(3, 12)
            while np.linalg.norm(goal_pos - init_pos) < 4:
                goal_pos = np.random.uniform(2, 18, size=3)
                goal_pos[2] = np.random.uniform(3, 12)

        # 障碍物
        if task in ["avoid", "navigate"]:
            num_obs = np.random.randint(1, 5)
        else:
            num_obs = np.random.randint(0, 2)
        obstacles = [np.random.uniform(3, 17, size=3) for _ in range(num_obs)]

        instructions = {
            "hover": "hover at the current position",
            "navigate": "fly to the red building",
            "avoid": "avoid the obstacles",
            "land": "land at the designated area"
        }

        # 生成轨迹
        images, states, actions = generate_trajectory(
            init_pos, goal_pos, obstacles, task, num_steps=60
        )

        episodes.append({
            "images": images,
            "states": states,
            "actions": actions,
            "instruction": instructions[task],
            "task": task,
            "goal": goal_pos.tolist(),
            "obstacles": [o.tolist() for o in obstacles]
        })

    np.savez(os.path.join(data_dir, "demonstrations.npz"), episodes=episodes)
    print(f"  数据集生成: {len(episodes)} episodes")
    print(f"  任务分布: {task_counts}")
    return episodes


def generate_trajectory(init_pos, goal_pos, obstacles, task, num_steps=60):
    """生成单条轨迹（改进版PID控制器）"""
    size = 64
    state = np.zeros(12, dtype=np.float32)
    state[:3] = init_pos

    images = []
    states = []
    actions = []

    # 改进的PID参数
    if task == "hover":
        kp = np.array([0.8, 0.8, 1.0])
        kd = np.array([0.3, 0.3, 0.4])
    elif task == "navigate":
        kp = np.array([1.2, 1.2, 1.0])
        kd = np.array([0.4, 0.4, 0.3])
    elif task == "avoid":
        kp = np.array([1.0, 1.0, 0.8])
        kd = np.array([0.5, 0.5, 0.3])
    else:  # land
        kp = np.array([0.6, 0.6, 1.5])
        kd = np.array([0.2, 0.2, 0.5])

    prev_error = np.zeros(3)

    for step in range(num_steps):
        img = generate_image(state[:3], goal_pos, obstacles, size)
        images.append(img)
        states.append(state.copy())

        # 计算控制指令
        error = goal_pos - state[:3]

        # 避障处理
        if task == "avoid":
            for obs in obstacles:
                diff = state[:3] - obs
                dist = np.linalg.norm(diff)
                if dist < 3.0:
                    # 强排斥力
                    repulsion = diff / (dist + 0.1) * 2.0
                    error += repulsion

        # PID控制
        derivative = error - prev_error
        vel_cmd = kp * error + kd * derivative
        vel_cmd = np.clip(vel_cmd, -2.0, 2.0)

        # 生成动作
        action = np.zeros(4, dtype=np.float32)
        action[:3] = vel_cmd / 5.0  # 归一化到[-0.4, 0.4]
        action[3] = 0.0  # yaw

        actions.append(action)

        # 状态更新
        dt = 0.1
        vel = state[3:6] + vel_cmd * dt
        vel = np.clip(vel, -2, 2)
        pos = state[:3] + vel * dt
        pos = np.clip(pos, 0, 20)
        pos[2] = max(0, pos[2])

        state[:3] = pos
        state[3:6] = vel

        prev_error = error

        # 任务完成检查
        if task == "land" and state[2] < 0.5:
            break
        elif task == "hover" and np.linalg.norm(error) < 0.5:
            break
        elif task in ["navigate", "avoid"] and np.linalg.norm(error) < 1.0:
            break

    return np.array(images), np.array(states), np.array(actions)


def generate_image(drone_pos, goal_pos, obstacles, size=64):
    """生成鸟瞰图"""
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

    # 目标
    gx = int(np.clip(goal_pos[0] * size / 20, 4, size - 5))
    gy = int(np.clip(goal_pos[1] * size / 20, 4, size - 5))
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx*dx + dy*dy <= 9:
                px, py = gx + dx, gy + dy
                if 0 <= px < size and 0 <= py < size:
                    img[px, py] = [1.0, 0.2, 0.2]

    # 无人机
    dx = int(np.clip(drone_pos[0] * size / 20, 4, size - 5))
    dy = int(np.clip(drone_pos[1] * size / 20, 4, size - 5))
    img[dx-2:dx+3, dy-2:dy+3] = [0.2, 0.2, 1.0]

    return img


# ============================================================
# 模型定义（不同变体）
# ============================================================

class VisualEncoder(nn.Module):
    def __init__(self, embed_dim=256, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(128, 1, 1), nn.Sigmoid()
            )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        feat = self.conv(x)
        if self.use_attention:
            attn = self.attention(feat)
            feat = feat * attn
        return self.pool(feat)


class LanguageEncoder(nn.Module):
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


class DroneVLA_V2(nn.Module):
    """改进的DroneVLA模型，支持注意力机制和残差连接"""
    def __init__(self, visual_dim=256, state_dim=12, action_dim=4, use_attention=False):
        super().__init__()
        self.visual_encoder = VisualEncoder(visual_dim, use_attention)
        self.language_encoder = LanguageEncoder(visual_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, visual_dim)
        )
        # 改进：更深的action decoder + 残差连接
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim * 3, 512), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh()
        )

    def forward(self, images, state, instructions):
        visual_feat = self.visual_encoder(images[:, -1])
        lang_feat = self.language_encoder(instructions, images.device)
        state_feat = self.state_encoder(state)
        combined = torch.cat([visual_feat, lang_feat, state_feat], dim=-1)
        fused = self.fusion(combined)
        return self.action_head(fused)


# ============================================================
# 数据集
# ============================================================

class ImprovedDataset(Dataset):
    def __init__(self, data_dir, num_frames=4, augment=False):
        self.num_frames = num_frames
        self.augment = augment
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

        images = ep["images"][step-self.num_frames:step]
        images = torch.FloatTensor(images).permute(0, 3, 1, 2)

        state = torch.FloatTensor(ep["states"][step])
        action = torch.FloatTensor(ep["actions"][step])
        instruction = ep["instruction"]

        # 数据增强
        if self.augment:
            # 图像噪声
            if np.random.random() < 0.3:
                noise = torch.randn_like(images) * 0.02
                images = torch.clamp(images + noise, 0, 1)
            # 状态噪声
            if np.random.random() < 0.3:
                state_noise = torch.randn_like(state) * 0.01
                state = state + state_noise

        return images, state, action, instruction


# ============================================================
# 训练和评估函数
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    for images, states, actions, instructions in loader:
        images, states, actions = images.to(device), states.to(device), actions.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast('cuda'):
                loss = criterion(model(images, states, instructions), actions)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(images, states, instructions), actions)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, states, actions, instructions in loader:
            images, states, actions = images.to(device), states.to(device), actions.to(device)
            loss = criterion(model(images, states, instructions), actions)
            total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_task(model, task, device, episodes=30, max_steps=80):
    """评估单个任务"""
    model.eval()
    successes = 0
    total_steps = 0

    for _ in range(episodes):
        init_pos = np.random.uniform(2, 18, size=3)
        init_pos[2] = np.random.uniform(3, 10)

        if task == "hover":
            goal_pos = init_pos.copy()
        elif task == "land":
            goal_pos = init_pos.copy()
            goal_pos[2] = 0.0
        else:
            goal_pos = np.random.uniform(2, 18, size=3)
            goal_pos[2] = np.random.uniform(3, 10)
            while np.linalg.norm(goal_pos - init_pos) < 5:
                goal_pos = np.random.uniform(2, 18, size=3)
                goal_pos[2] = np.random.uniform(3, 10)

        num_obs = 0 if task in ["hover", "land"] else np.random.randint(1, 4)
        obstacles = [np.random.uniform(3, 17, size=3) for _ in range(num_obs)]

        instructions = {
            "hover": "hover at the current position",
            "navigate": "fly to the red building",
            "avoid": "avoid the obstacles",
            "land": "land at the designated area"
        }

        state = np.zeros(12, dtype=np.float32)
        state[:3] = init_pos
        frame_buffer = [generate_image(init_pos, goal_pos, obstacles) for _ in range(4)]

        with torch.no_grad():
            for step in range(max_steps):
                images = torch.FloatTensor(np.array(frame_buffer)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = model(images, state_tensor, [instructions[task]])[0].cpu().numpy()

                dt = 0.1
                vel = state[3:6] + action[:3] * dt * 5.0
                vel = np.clip(vel, -2, 2)
                pos = state[:3] + vel * dt
                pos = np.clip(pos, 0, 20)
                pos[2] = max(0, pos[2])
                state[:3] = pos
                state[3:6] = vel

                frame_buffer.pop(0)
                frame_buffer.append(generate_image(pos, goal_pos, obstacles))

                if task == "land" and state[2] < 0.5 and np.linalg.norm(state[:2] - goal_pos[:2]) < 1.5:
                    successes += 1
                    total_steps += step + 1
                    break
                elif task == "hover" and np.linalg.norm(state[:3] - goal_pos) < 1.0:
                    successes += 1
                    total_steps += step + 1
                    break
                elif task in ["navigate", "avoid"] and np.linalg.norm(state[:3] - goal_pos) < 1.5:
                    successes += 1
                    total_steps += step + 1
                    break
            else:
                total_steps += max_steps

    return {
        "success_rate": successes / episodes * 100,
        "avg_steps": total_steps / episodes,
        "episodes": episodes
    }


# ============================================================
# 迭代配置
# ============================================================

ITERATION_CONFIGS = {
    1: {
        "name": "Baseline",
        "num_episodes": 100,
        "epochs": 50,
        "batch_size": 32,
        "lr": 1e-3,
        "use_attention": False,
        "augment": False,
        "description": "基线模型，基本配置"
    },
    2: {
        "name": "More Data",
        "num_episodes": 300,
        "epochs": 80,
        "batch_size": 32,
        "lr": 1e-3,
        "use_attention": False,
        "augment": False,
        "description": "增加训练数据量（3倍），平衡任务分布"
    },
    3: {
        "name": "Data Augmentation",
        "num_episodes": 300,
        "epochs": 80,
        "batch_size": 32,
        "lr": 1e-3,
        "use_attention": False,
        "augment": True,
        "description": "启用数据增强（图像噪声+状态扰动）"
    },
    4: {
        "name": "Attention Mechanism",
        "num_episodes": 300,
        "epochs": 100,
        "batch_size": 32,
        "lr": 8e-4,
        "use_attention": True,
        "augment": True,
        "description": "添加视觉注意力机制"
    },
    5: {
        "name": "Larger Model",
        "num_episodes": 400,
        "epochs": 100,
        "batch_size": 32,
        "lr": 8e-4,
        "use_attention": True,
        "augment": True,
        "visual_dim": 384,
        "description": "增大模型容量（visual_dim=384）"
    },
    6: {
        "name": "Learning Rate Tuning",
        "num_episodes": 400,
        "epochs": 120,
        "batch_size": 32,
        "lr": 5e-4,
        "use_attention": True,
        "augment": True,
        "visual_dim": 384,
        "description": "降低学习率，增加训练轮数"
    },
    7: {
        "name": "Focus on Weak Tasks",
        "num_episodes": 500,
        "epochs": 120,
        "batch_size": 32,
        "lr": 5e-4,
        "use_attention": True,
        "augment": True,
        "focus_tasks": ["navigate", "avoid"],
        "description": "重点增加navigate/avoid任务数据"
    },
    8: {
        "name": "Deeper Decoder",
        "num_episodes": 500,
        "epochs": 150,
        "batch_size": 32,
        "lr": 3e-4,
        "use_attention": True,
        "augment": True,
        "deeper_decoder": True,
        "description": "加深动作解码器"
    },
    9: {
        "name": "Curriculum Learning",
        "num_episodes": 600,
        "epochs": 150,
        "batch_size": 32,
        "lr": 3e-4,
        "use_attention": True,
        "augment": True,
        "curriculum": True,
        "description": "课程学习：从简单到复杂"
    },
    10: {
        "name": "Final Optimized",
        "num_episodes": 800,
        "epochs": 200,
        "batch_size": 32,
        "lr": 2e-4,
        "use_attention": True,
        "augment": True,
        "visual_dim": 384,
        "deeper_decoder": True,
        "description": "最终优化：所有最佳配置组合"
    }
}


# ============================================================
# 主训练循环
# ============================================================

def run_iteration(iter_num, device):
    """运行单次迭代"""
    config = ITERATION_CONFIGS[iter_num]
    exp_dir = f"experiments/iteration_{iter_num:02d}"

    print(f"\n{'='*60}")
    print(f"迭代 {iter_num:02d}: {config['name']}")
    print(f"描述: {config['description']}")
    print(f"{'='*60}")

    # 创建目录
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{exp_dir}/visualizations", exist_ok=True)
    os.makedirs(f"{exp_dir}/logs", exist_ok=True)

    # 保存配置
    with open(f"{exp_dir}/config.yaml", 'w') as f:
        json.dump(config, f, indent=2)

    start_time = time.time()

    # 1. 生成数据
    print("\n[1/4] 生成训练数据...")
    data_dir = f"data/iter_{iter_num:02d}"
    if config.get("focus_tasks"):
        # 为弱任务生成额外数据
        all_episodes = []
        for task in config["focus_tasks"]:
            eps = generate_improved_dataset(
                f"{data_dir}_{task}",
                num_episodes=config["num_episodes"] // 3,
                focus_task=task
            )
            all_episodes.extend(eps)
        # 通用数据
        eps = generate_improved_dataset(data_dir, num_episodes=config["num_episodes"] // 3)
        all_episodes.extend(eps)
        # 保存合并数据
        os.makedirs(data_dir, exist_ok=True)
        np.savez(os.path.join(data_dir, "demonstrations.npz"), episodes=all_episodes)
    else:
        generate_improved_dataset(data_dir, num_episodes=config["num_episodes"])

    # 2. 训练
    print("\n[2/4] 开始训练...")
    visual_dim = config.get("visual_dim", 256)
    model = DroneVLA_V2(
        visual_dim=visual_dim,
        use_attention=config["use_attention"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数: {total_params:,}")

    dataset = ImprovedDataset(data_dir, augment=config["augment"])
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)
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

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Best: {best_val_loss:.4f}")

    training_time = time.time() - start_time

    # 保存最终模型
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
    }, f"{exp_dir}/checkpoints/final_model.pt")

    # 3. 评估
    print("\n[3/4] 评估模型...")
    # 加载最佳模型
    checkpoint = torch.load(f"{exp_dir}/checkpoints/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    eval_results = {}
    for task in ["navigate", "avoid", "hover", "land"]:
        print(f"  评估: {task}...")
        result = evaluate_task(model, task, device, episodes=30)
        eval_results[task] = result
        print(f"    成功率: {result['success_rate']:.1f}%, 平均步数: {result['avg_steps']:.1f}")

    with open(f"{exp_dir}/logs/eval_results.json", 'w') as f:
        json.dump(eval_results, f, indent=2)

    # 4. 保存指标
    metrics = {
        'total_params': total_params,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'epochs': config['epochs'],
        'device': str(device),
        'training_time': training_time,
        'num_episodes': config['num_episodes'],
        'iteration_name': config['name'],
        'task_success_rates': {t: r['success_rate'] for t, r in eval_results.items()},
        'overall_success_rate': np.mean([r['success_rate'] for r in eval_results.values()])
    }
    with open(f"{exp_dir}/logs/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # 5. 生成可视化
    print("\n[4/4] 生成可视化...")
    generate_iteration_plots(exp_dir, train_losses, val_losses, eval_results, config)

    print(f"\n迭代 {iter_num} 完成!")
    print(f"  最佳验证损失: {best_val_loss:.4f}")
    print(f"  整体成功率: {metrics['overall_success_rate']:.1f}%")
    print(f"  训练时间: {training_time:.0f}s")

    return metrics


def generate_iteration_plots(exp_dir, train_losses, val_losses, eval_results, config):
    """生成迭代的可视化图表"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 训练曲线
    axes[0, 0].plot(train_losses, label='Train Loss', color='#2196F3')
    axes[0, 0].plot(val_losses, label='Val Loss', color='#FF5722')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 任务成功率
    tasks = list(eval_results.keys())
    success_rates = [eval_results[t]['success_rate'] for t in tasks]
    colors = ['#4CAF50' if r > 50 else '#FF9800' if r > 0 else '#F44336' for r in success_rates]
    bars = axes[0, 1].bar(tasks, success_rates, color=colors, width=0.5)
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].set_title('Task Success Rate')
    axes[0, 1].set_ylim(0, 110)
    for bar, val in zip(bars, success_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=11)

    # 3. 平均步数
    avg_steps = [eval_results[t]['avg_steps'] for t in tasks]
    bars = axes[1, 0].bar(tasks, avg_steps, color='#2196F3', width=0.5)
    axes[1, 0].set_ylabel('Average Steps')
    axes[1, 0].set_title('Average Steps to Complete')
    for bar, val in zip(bars, avg_steps):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1f}', ha='center', va='bottom', fontsize=11)

    # 4. 配置信息
    info = f"""Iteration: {config['name']}
Description: {config['description']}

Training Config:
  Episodes: {config['num_episodes']}
  Epochs: {config['epochs']}
  Learning Rate: {config['lr']}
  Batch Size: {config['batch_size']}
  Attention: {config['use_attention']}
  Augmentation: {config['augment']}

Results:
  Best Val Loss: {min(val_losses):.4f}
  Overall Success: {np.mean(success_rates):.1f}%"""
    axes[1, 1].text(0.05, 0.5, info, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')

    plt.suptitle(f'Iteration {exp_dir.split("_")[-1]}: {config["name"]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{exp_dir}/visualizations/results.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_comparison_plot(all_metrics):
    """生成所有迭代的对比图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    iterations = sorted(all_metrics.keys())
    names = [all_metrics[i].get('iteration_name', f'Iter {i}') for i in iterations]

    # 1. 验证损失对比
    val_losses = [all_metrics[i]['best_val_loss'] for i in iterations]
    axes[0, 0].plot(iterations, val_losses, 'o-', color='#FF5722', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Best Validation Loss')
    axes[0, 0].set_title('Validation Loss Progress')
    axes[0, 0].grid(True, alpha=0.3)
    for i, v in zip(iterations, val_losses):
        axes[0, 0].annotate(f'{v:.4f}', (i, v), textcoords="offset points", xytext=(0, 10), ha='center')

    # 2. 整体成功率对比
    overall = [all_metrics[i]['overall_success_rate'] for i in iterations]
    axes[0, 1].bar(iterations, overall, color='#4CAF50', width=0.6)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Overall Success Rate (%)')
    axes[0, 1].set_title('Overall Success Rate Progress')
    axes[0, 1].set_ylim(0, 110)
    for i, v in zip(iterations, overall):
        axes[0, 1].text(i, v, f'{v:.1f}%', ha='center', va='bottom')

    # 3. 各任务成功率趋势
    tasks = ["navigate", "avoid", "hover", "land"]
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    for task, color in zip(tasks, colors):
        rates = [all_metrics[i]['task_success_rates'].get(task, 0) for i in iterations]
        axes[1, 0].plot(iterations, rates, 'o-', label=task, color=color, linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Success Rate (%)')
    axes[1, 0].set_title('Task-wise Success Rate Progress')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 110)

    # 4. 训练时间
    times = [all_metrics[i]['training_time'] / 60 for i in iterations]
    axes[1, 1].bar(iterations, times, color='#9C27B0', width=0.6)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Training Time (min)')
    axes[1, 1].set_title('Training Time per Iteration')
    for i, v in zip(iterations, times):
        axes[1, 1].text(i, v, f'{v:.1f}m', ha='center', va='bottom')

    plt.suptitle('DroneVLA Training Iterations Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("experiments/iterations_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("迭代对比图已保存: experiments/iterations_comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, help="起始迭代")
    parser.add_argument("--end", type=int, default=10, help="结束迭代")
    parser.add_argument("--device", default="auto", help="设备 (auto/cpu/cuda)")
    args = parser.parse_args()

    # 设备选择
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("DroneVLA 多轮迭代训练")
    print("=" * 60)
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"迭代范围: {args.start} - {args.end}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_metrics = {}

    for iter_num in range(args.start, args.end + 1):
        try:
            metrics = run_iteration(iter_num, device)
            all_metrics[iter_num] = metrics
        except Exception as e:
            print(f"\n迭代 {iter_num} 出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 生成对比图
    if len(all_metrics) > 1:
        print("\n生成迭代对比图...")
        generate_comparison_plot(all_metrics)

    # 总结
    print("\n" + "=" * 60)
    print("训练总结")
    print("=" * 60)
    for iter_num, metrics in sorted(all_metrics.items()):
        print(f"迭代 {iter_num:02d} ({metrics.get('iteration_name', 'N/A')}): "
              f"成功率={metrics['overall_success_rate']:.1f}%, "
              f"验证损失={metrics['best_val_loss']:.4f}")

    best_iter = max(all_metrics.items(), key=lambda x: x[1]['overall_success_rate'])
    print(f"\n最佳迭代: {best_iter[0]} ({best_iter[1].get('iteration_name', 'N/A')})")
    print(f"  整体成功率: {best_iter[1]['overall_success_rate']:.1f}%")
    print(f"  验证损失: {best_iter[1]['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
