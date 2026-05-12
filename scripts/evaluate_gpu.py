"""
DroneVLA GPU评估脚本

与train_gpu.py配套使用

使用方法：
    python scripts/evaluate_gpu.py --model experiments/iteration_01/checkpoints/best_model.pt --episodes 50
"""

import torch
import numpy as np
import os
import argparse
import json
import sys

sys.path.insert(0, '.')


class SimpleVisualEncoder(torch.nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
            torch.nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class SimpleLanguageEncoder(torch.nn.Module):
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
        self.embedding = torch.nn.Embedding(100, embed_dim)
        self.fc = torch.nn.Linear(embed_dim, embed_dim)

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


class DroneVLA_GPU(torch.nn.Module):
    def __init__(self, visual_dim=256, state_dim=12, action_dim=4):
        super().__init__()
        self.visual_encoder = SimpleVisualEncoder(visual_dim)
        self.language_encoder = SimpleLanguageEncoder(visual_dim)
        self.state_encoder = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, visual_dim)
        )
        self.action_decoder = torch.nn.Sequential(
            torch.nn.Linear(visual_dim * 3, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim), torch.nn.Tanh()
        )

    def forward(self, images, state, instructions):
        visual_feat = self.visual_encoder(images[:, -1])
        lang_feat = self.language_encoder(instructions, images.device)
        state_feat = self.state_encoder(state)
        combined = torch.cat([visual_feat, lang_feat, state_feat], dim=-1)
        return self.action_decoder(combined)


def generate_image(drone_pos, goal_pos, obstacles, size=64):
    img = np.ones((size, size, 3), dtype=np.float32) * 0.3
    for i in range(0, size, 8):
        img[i, :] = 0.25
        img[:, i] = 0.25
    for obs in obstacles:
        px = int(np.clip(obs[0] * size / 20, 2, size - 6))
        py = int(np.clip(obs[1] * size / 20, 2, size - 6))
        img[px:px+4, py:py+4] = 0.1
    gx = int(np.clip(goal_pos[0] * size / 20, 4, size - 5))
    gy = int(np.clip(goal_pos[1] * size / 20, 4, size - 5))
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx*dx + dy*dy <= 9:
                px, py = gx + dx, gy + dy
                if 0 <= px < size and 0 <= py < size:
                    img[px, py] = [1.0, 0.2, 0.2]
    dx = int(np.clip(drone_pos[0] * size / 20, 4, size - 5))
    dy = int(np.clip(drone_pos[1] * size / 20, 4, size - 5))
    img[dx-2:dx+3, dy-2:dy+3] = [0.2, 0.2, 1.0]
    return img


def evaluate_episode(model, task, device, max_steps=80):
    initial_pos = np.random.uniform(2, 18, size=3)
    initial_pos[2] = np.random.uniform(3, 10)

    if task == "hover":
        goal_pos = initial_pos.copy()
    elif task == "land":
        goal_pos = initial_pos.copy()
        goal_pos[2] = 0.0
    else:
        goal_pos = np.random.uniform(2, 18, size=3)
        goal_pos[2] = np.random.uniform(3, 10)
        while np.linalg.norm(goal_pos - initial_pos) < 5:
            goal_pos = np.random.uniform(2, 18, size=3)
            goal_pos[2] = np.random.uniform(3, 10)

    num_obs = 0 if task in ["hover", "land"] else np.random.randint(0, 3)
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

    num_frames = 4
    frame_buffer = [generate_image(initial_pos, goal_pos, obstacles) for _ in range(num_frames)]

    model.eval()
    with torch.no_grad():
        for step in range(max_steps):
            images = torch.FloatTensor(np.array(frame_buffer)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            action = model(images, state_tensor, [instruction])[0].cpu().numpy()

            dt = 0.1
            vel = state[3:6] + action[:3] * dt * 5.0
            vel = np.clip(vel, -2, 2)
            pos = state[:3] + vel * dt
            pos = np.clip(pos, 0, 20)
            pos[2] = max(0, pos[2])

            state[:3] = pos
            state[3:6] = vel
            state[8] += action[3] * dt

            frame_buffer.pop(0)
            frame_buffer.append(generate_image(pos, goal_pos, obstacles))

            if task == "land":
                if state[2] < 0.5 and np.linalg.norm(state[:2] - goal_pos[:2]) < 1.5:
                    return True, step + 1, np.linalg.norm(state[:3] - goal_pos)
            elif task == "hover":
                if np.linalg.norm(state[:3] - goal_pos) < 1.0:
                    return True, step + 1, np.linalg.norm(state[:3] - goal_pos)
            else:
                if np.linalg.norm(state[:3] - goal_pos) < 1.5:
                    return True, step + 1, np.linalg.norm(state[:3] - goal_pos)

    return False, max_steps, np.linalg.norm(state[:3] - goal_pos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="experiments/iteration_01/checkpoints/best_model.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--tasks", nargs="+", default=["navigate", "avoid", "hover", "land"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("DroneVLA GPU评估")
    print("=" * 60)
    print(f"设备: {device}")

    model = DroneVLA_GPU().to(device)
    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型加载成功 (epoch {checkpoint.get('epoch', '?')})")

    results = {}
    for task in args.tasks:
        print(f"\n评估任务: {task}")
        successes = 0
        total_steps = 0
        episodes = args.episodes // len(args.tasks)

        for ep in range(episodes):
            success, steps, dist = evaluate_episode(model, task, device)
            if success:
                successes += 1
            total_steps += steps
            if (ep + 1) % 5 == 0:
                print(f"  {ep+1}/{episodes} | Success: {successes}/{ep+1}")

        results[task] = {
            "success_rate": successes / episodes * 100,
            "avg_steps": total_steps / episodes,
            "episodes": episodes
        }
        print(f"  {task}: {results[task]['success_rate']:.1f}% success, {results[task]['avg_steps']:.1f} avg steps")

    os.makedirs("experiments/iteration_01/logs", exist_ok=True)
    with open("experiments/iteration_01/logs/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    for task, res in results.items():
        print(f"  {task}: {res['success_rate']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
