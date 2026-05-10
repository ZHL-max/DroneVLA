# DroneVLA API 参考手册
## 核心模块与接口文档

---

## 目录

1. [模型模块](#1-模型模块)
2. [环境模块](#2-环境模块)
3. [相机模块](#3-相机模块)
4. [训练模块](#4-训练模块)

---

## 1. 模型模块

### 1.1 DroneVLA

主模型类，整合视觉、语言、状态编码和动作解码。

```python
from src.models.drone_vla import DroneVLA

model = DroneVLA(
    visual_dim=256,        # 视觉特征维度
    language_dim=256,      # 语言特征维度
    state_dim=12,          # 状态维度
    state_embed_dim=128,   # 状态嵌入维度
    action_dim=4,          # 动作维度
    action_horizon=8,      # 动作序列长度
    use_world_model=True,  # 是否使用世界模型
    action_mode='deterministic'  # 动作模式: 'deterministic' 或 'diffusion'
)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| visual_dim | int | 256 | 视觉编码器输出维度 |
| language_dim | int | 256 | 语言编码器输出维度 |
| state_dim | int | 12 | 无人机状态维度 [x,y,z,vx,vy,vz,roll,pitch,yaw,wx,wy,wz] |
| state_embed_dim | int | 128 | 状态编码器输出维度 |
| action_dim | int | 4 | 动作维度 [vx,vy,vz,yaw_rate] |
| action_horizon | int | 8 | 动作序列长度 |
| use_world_model | bool | True | 是否启用世界模型 |
| action_mode | str | 'deterministic' | 动作解码模式 |

**前向传播：**

```python
outputs = model(
    images,           # [B, T, C, H, W] 图像序列
    instructions,     # List[str] 语言指令
    state,            # [B, state_dim] 无人机状态
    actions=None      # [B, T, action_dim] 真实动作（训练时）
)

# 输出字典
outputs['actions']          # [B, action_dim] 预测动作
outputs['visual_features']  # [B, visual_dim] 视觉特征
outputs['language_features'] # [B, language_dim] 语言特征
outputs['state_features']   # [B, state_embed_dim] 状态特征
outputs['future_states']    # [B, T, state_dim] 预测未来状态（如果使用世界模型）
outputs['future_rewards']   # [B, T, 1] 预测未来奖励（如果使用世界模型）
```

---

### 1.2 VisualEncoder

视觉编码器，处理图像输入。

```python
from src.models.drone_vla import VisualEncoder

encoder = VisualEncoder(
    embed_dim=256,     # 输出维度
    temporal=True      # 是否使用时序注意力
)

# 输入: [B, T, C, H, W]
# 输出: [B, embed_dim]
features = encoder(images)
```

**支持的骨干网络：**
- EfficientNet-B0 (默认)
- ResNet-18/34/50

---

### 1.3 LanguageEncoder

语言编码器，基于BERT。

```python
from src.models.drone_vla import LanguageEncoder

encoder = LanguageEncoder(
    embed_dim=256,           # 输出维度
    model_name='bert-base-uncased'  # 预训练模型
)

# 输入: List[str]
# 输出: [B, embed_dim]
features = encoder(["fly to the red building", "hover at position"])
```

---

### 1.4 StateEncoder

状态编码器，处理无人机状态向量。

```python
from src.models.drone_vla import StateEncoder

encoder = StateEncoder(
    state_dim=12,     # 输入状态维度
    embed_dim=128     # 输出维度
)

# 输入: [B, state_dim]
# 输出: [B, embed_dim]
features = encoder(state)
```

---

### 1.5 WorldModel

世界模型，预测未来状态和奖励。

```python
from src.models.drone_vla import WorldModel

model = WorldModel(
    state_dim=12,     # 状态维度
    action_dim=4      # 动作维度
)

# 预测未来
future_states, future_rewards = model.predict_future(
    state,            # [B, state_dim] 当前状态
    actions           # [B, T, action_dim] 动作序列
)

# 想象轨迹
trajectory = model.imagine_trajectory(
    state,            # [B, state_dim] 初始状态
    policy,           # 策略网络
    horizon=10        # 想象步数
)
```

---

### 1.6 ActionDecoder

动作解码器，支持确定性和扩散两种模式。

```python
from src.models.drone_vla import ActionDecoder

# 确定性模式
decoder = ActionDecoder(
    input_dim=256,         # 输入维度
    action_dim=4,          # 动作维度
    action_horizon=8,      # 动作序列长度
    mode='deterministic'
)

# 扩散模式
decoder = ActionDecoder(
    input_dim=256,
    action_dim=4,
    action_horizon=8,
    mode='diffusion',
    diffusion_steps=100    # 扩散步数
)

# 输入: [B, input_dim]
# 输出: [B, action_dim] 或 [B, action_horizon, action_dim]
action = decoder(features)
```

---

## 2. 环境模块

### 2.1 DroneEnv

基于PyBullet的无人机仿真环境。

```python
from src.environments.drone_env import DroneEnv

env = DroneEnv(
    task='hover',           # 任务类型: 'hover', 'navigate', 'track', 'avoid'
    max_steps=200,          # 最大步数
    image_size=64,          # 图像尺寸
    render_mode='rgb_array' # 渲染模式
)

# 重置环境
obs, info = env.reset()

# 执行动作
action = [vx, vy, vz, yaw_rate]  # 速度控制
obs, reward, done, truncated, info = env.step(action)
```

**任务类型：**

| 任务 | 说明 | 奖励函数 |
|------|------|----------|
| hover | 保持悬停 | -distance_to_target |
| navigate | 导航到目标 | -distance + goal_bonus |
| track | 跟踪移动目标 | -tracking_error |
| avoid | 避障导航 | -distance - collision_penalty |

---

### 2.2 DroneLanguageEnv

带语言指令的无人机环境。

```python
from src.environments.drone_env import DroneLanguageEnv

env = DroneLanguageEnv(max_steps=200)

# 重置，获取语言指令
obs, instruction = env.reset()
print(f"指令: {instruction}")

# 执行动作
obs, reward, done, info = env.step(action)
```

---

## 3. 相机模块

### 3.1 CameraBase

相机基类，定义通用接口。

```python
from src.camera.camera_base import CameraBase

class MyCamera(CameraBase):
    def connect(self) -> bool:
        # 连接相机
        pass

    def disconnect(self):
        # 断开连接
        pass

    def get_frame(self) -> np.ndarray:
        # 获取RGB图像 [H, W, 3]
        pass

    def get_depth(self) -> np.ndarray:
        # 获取深度图 [H, W]
        pass
```

---

### 3.2 RealSenseCamera

Intel RealSense相机驱动。

```python
from src.camera.realsense_camera import RealSenseCamera

camera = RealSenseCamera(config={
    'width': 640,
    'height': 480,
    'fps': 30,
    'enable_depth': True,
    'enable_imu': True
})

# 连接
camera.connect()

# 获取图像
rgb = camera.get_frame()      # [480, 640, 3]
depth = camera.get_depth()    # [480, 640]
imu = camera.get_imu()        # dict with accel and gyro

# 断开
camera.disconnect()
```

---

### 3.3 USBCamera

通用USB摄像头驱动。

```python
from src.camera.usb_camera import USBCamera

camera = USBCamera(config={
    'device_id': 0,
    'width': 640,
    'height': 480,
    'fps': 30
})

camera.connect()
rgb = camera.get_frame()
camera.disconnect()
```

---

### 3.4 OAKDCamera

OAK-D相机驱动。

```python
from src.camera.oakd_camera import OAKDCamera

camera = OAKDCamera(config={
    'width': 640,
    'height': 480,
    'fps': 30
})

camera.connect()
rgb = camera.get_frame()
depth = camera.get_depth()
camera.disconnect()
```

---

## 4. 训练模块

### 4.1 DemonstrationDataset

演示数据集类。

```python
from src.training.trainer import DemonstrationDataset

dataset = DemonstrationDataset(
    data_dir='data/train',
    image_size=(64, 64),
    num_frames=4,
    action_horizon=8
)

# 获取样本
sample = dataset[0]
images = sample['images']      # [T, C, H, W]
state = sample['state']        # [state_dim]
actions = sample['actions']    # [action_horizon, action_dim]
instruction = sample['instruction']  # str
```

---

### 4.2 DroneVLATrainer

训练器类。

```python
from src.training.trainer import DroneVLATrainer

trainer = DroneVLATrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    learning_rate=1e-4,
    batch_size=32,
    device='cuda'
)

# 训练
trainer.train(num_epochs=100)

# 评估
metrics = trainer.evaluate()
```

---

### 4.3 collect_demonstrations

收集专家演示数据。

```python
from src.training.trainer import collect_demonstrations

demos = collect_demonstrations(
    env=env,
    num_episodes=100,
    expert_policy=expert_policy
)
```

---

## 5. 工具函数

### 5.1 图像处理

```python
import numpy as np

def preprocess_image(image: np.ndarray, size: int = 64) -> np.ndarray:
    """预处理图像"""
    # 调整大小
    image = cv2.resize(image, (size, size))
    # 归一化
    image = image.astype(np.float32) / 255.0
    # 转换为CHW格式
    image = image.transpose(2, 0, 1)
    return image
```

### 5.2 状态处理

```python
def normalize_state(state: np.ndarray) -> np.ndarray:
    """归一化状态向量"""
    # 位置: 0-20 -> -1 to 1
    state[:3] = state[:3] / 10.0 - 1.0
    # 速度: -2 to 2 -> -1 to 1
    state[3:6] = state[3:6] / 2.0
    # 姿态: -pi to pi -> -1 to 1
    state[6:9] = state[6:9] / np.pi
    return state
```

---

*最后更新：2026-05-11*
