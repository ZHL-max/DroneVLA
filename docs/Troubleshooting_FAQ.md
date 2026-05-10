# 故障排除与常见问题
## DroneVLA 问题诊断指南

---

## 目录

1. [安装问题](#1-安装问题)
2. [训练问题](#2-训练问题)
3. [模型问题](#3-模型问题)
4. [相机问题](#4-相机问题)
5. [飞控问题](#5-飞控问题)
6. [性能问题](#6-性能问题)

---

## 1. 安装问题

### Q: conda创建环境失败

**症状：**
```
ResolvePackageNotFound: python=3.10
```

**解决方案：**
```bash
# 更新conda
conda update conda

# 使用更宽松的版本约束
conda create -n dronevla python=3.10.*

# 或使用mamba（更快）
conda install mamba -c conda-forge
mamba create -n dronevla python=3.10
```

---

### Q: PyTorch安装后无法使用GPU

**症状：**
```python
import torch
print(torch.cuda.is_available())  # False
```

**解决方案：**
```bash
# 检查CUDA版本
nvidia-smi

# 安装对应版本的PyTorch
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 验证
python -c "import torch; print(torch.cuda.is_available())"
```

---

### Q: pip install -e . 失败

**症状：**
```
error: metadata-generation-failed
```

**解决方案：**
```bash
# 升级pip和setuptools
pip install --upgrade pip setuptools wheel

# 重新安装
pip install -e .

# 如果仍然失败，尝试
pip install -e . --no-build-isolation
```

---

## 2. 训练问题

### Q: 训练loss不下降

**可能原因：**
1. 学习率不合适
2. 数据未归一化
3. 模型架构问题

**解决方案：**
```bash
# 尝试不同的学习率
python scripts/train.py --lr 1e-3   # 更高
python scripts/train.py --lr 1e-5   # 更低

# 检查数据
python -c "
import numpy as np
data = np.load('data/train/demonstrations.npz', allow_pickle=True)
episodes = data['episodes']
print(f'样本数: {len(episodes)}')
print(f'动作范围: {episodes[0]["actions"].min():.3f} to {episodes[0]["actions"].max():.3f}')
"
```

---

### Q: 内存不足 (OOM)

**症状：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**
```bash
# 减小批大小
python scripts/train.py --batch_size 8

# 使用梯度累积
python scripts/train.py --gradient_accumulation_steps 4

# 使用混合精度
python scripts/train.py --use_amp

# 减小模型
python scripts/train.py --visual_dim 128 --language_dim 128
```

---

### Q: 训练速度太慢

**解决方案：**
```bash
# 使用GPU
python scripts/train.py --device cuda

# 使用轻量级模型
python scripts/train_lightweight.py

# 减少数据量
python scripts/generate_dataset.py --num_episodes 100

# 使用更小的批大小（减少每步计算）
python scripts/train.py --batch_size 4 --epochs 10
```

---

## 3. 模型问题

### Q: 模型输出全为0或NaN

**可能原因：**
1. 梯度爆炸
2. 输入未归一化
3. 权重初始化问题

**解决方案：**
```python
# 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")

# 添加梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 检查输入
print(f"图像范围: [{images.min():.3f}, {images.max():.3f}]")
print(f"状态范围: [{states.min():.3f}, {states.max():.3f}]")
```

---

### Q: 模型加载失败

**症状：**
```
RuntimeError: Error(s) in loading state_dict
```

**解决方案：**
```python
# 使用strict=False加载
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# 检查键不匹配
model_keys = set(model.state_dict().keys())
ckpt_keys = set(checkpoint['model_state_dict'].keys())
print("缺失的键:", model_keys - ckpt_keys)
print("多余的键:", ckpt_keys - model_keys)
```

---

## 4. 相机问题

### Q: RealSense相机无法连接

**症状：**
```
RuntimeError: No device connected
```

**解决方案：**
```bash
# 检查USB连接
lsusb | grep Intel

# 安装驱动
sudo apt install librealsense2-dkms librealsense2-utils

# 测试
realsense-viewer

# 权限问题
sudo usermod -a -G video $USER
# 重新登录
```

---

### Q: 相机图像卡顿

**解决方案：**
```python
# 降低分辨率
camera = RealSenseCamera(config={
    'width': 320,    # 从640降到320
    'height': 240,   # 从480降到240
    'fps': 15        # 从30降到15
})

# 使用多线程
import threading

class AsyncCamera:
    def __init__(self, camera):
        self.camera = camera
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            self.frame = self.camera.get_frame()

    def get_frame(self):
        return self.frame
```

---

## 5. 飞控问题

### Q: MAVLink连接失败

**症状：**
```
mavutil.mavlink_connection() timeout
```

**解决方案：**
```bash
# 检查串口
ls /dev/ttyUSB*
ls /dev/ttyACM*

# 检查权限
sudo usermod -a -G dialout $USER
# 重新登录

# 测试连接
python -c "
from pymavlink import mavutil
master = mavutil.mavlink_connection('/dev/ttyUSB0', baud=57600)
master.wait_heartbeat()
print('Connected!')
"
```

---

### Q: 飞控不响应速度指令

**解决方案：**
```python
# 确保在正确的模式下
# 设置模式为GUIDED
master.set_mode_apm('GUIDED')

# 发送心跳
master.mav.heartbeat_send(
    mavutil.mavlink.MAV_TYPE_GCS,
    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
    0, 0, 0
)

# 检查解锁状态
master.arducopter_arm()
```

---

## 6. 性能问题

### Q: 推理延迟太高

**目标：** < 100ms (10Hz控制)

**解决方案：**
```python
# 1. 使用TensorRT优化
python scripts/export_onnx.py --model logs/best_model.pt
# 然后使用TensorRT转换

# 2. 减小输入尺寸
images = torch.randn(1, 4, 3, 32, 32)  # 从64x64降到32x32

# 3. 使用更小的模型
model = SimpleDroneVLA()  # 55K参数 vs 115M参数

# 4. 使用GPU推理
model = model.cuda()
images = images.cuda()

# 5. 批量推理（如果有多帧）
with torch.no_grad():
    outputs = model(images, instructions, states)
```

---

### Q: GPU利用率低

**检查方法：**
```bash
# 监控GPU使用
nvidia-smi -l 1

# 检查是否在CPU上
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Current device: {torch.cuda.current_device()}')
print(f'Device name: {torch.cuda.get_device_name(0)}')
"
```

**解决方案：**
```python
# 确保数据在GPU上
device = torch.device('cuda')
model = model.to(device)
images = images.to(device)
states = states.to(device)

# 增加批大小
batch_size = 32  # 从4增加到32

# 使用pin_memory加速数据加载
DataLoader(dataset, pin_memory=True, num_workers=4)
```

---

## 获取帮助

如果以上方法都无法解决问题：

1. **查看日志：** `logs/` 目录下的训练日志
2. **搜索Issue：** [GitHub Issues](https://github.com/ZHL-max/DroneVLA/issues)
3. **提交Issue：** 包含完整的错误信息和环境信息
4. **社区讨论：** [GitHub Discussions](https://github.com/ZHL-max/DroneVLA/discussions)

---

*最后更新：2026-05-11*
