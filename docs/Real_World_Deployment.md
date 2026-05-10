# 真实无人机部署指南
## 从仿真到实飞的完整流程

---

## 目录

1. [部署前准备](#1-部署前准备)
2. [硬件组装](#2-硬件组装)
3. [软件环境配置](#3-软件环境配置)
4. [模型部署](#4-模型部署)
5. [飞行测试](#5-飞行测试)
6. [安全注意事项](#6-安全注意事项)

---

## 1. 飞行前检查清单

### 1.1 硬件检查

```
□ 飞控固件已更新到最新版本
□ 电调校准完成
□ 电机转向正确（参考飞控文档）
□ 螺旋桨安装正确（CW/CCW）
□ 电池充满电
□ GPS信号良好（至少10颗卫星）
□ 遥控器已校准
□ 失控保护已设置
```

### 1.2 软件检查

```
□ 飞控参数已保存备份
□ 伴飞电脑系统正常
□ 相机驱动已安装
□ VLA模型已加载
□ MAVLink连接正常
□ 数据链路稳定
```

---

## 2. 硬件组装

### 2.1 典型组装方案

```
┌─────────────────────────────────────────────────────┐
│                    无人机平台                         │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │              机架 (F450)                      │   │
│  │                                             │   │
│  │    [电机1]                     [电机2]       │   │
│  │       \                         /           │   │
│  │        \                       /            │   │
│  │         ┌───────────────────┐              │   │
│  │         │     Pixhawk 6X    │              │   │
│  │         │     (飞控)        │              │   │
│  │         └───────────────────┘              │   │
│  │        /                       \            │   │
│  │       /                         \           │   │
│  │    [电机3]                     [电机4]       │   │
│  │                                             │   │
│  │  ┌──────────────────────────────────────┐  │   │
│  │  │        Jetson Orin Nano              │  │   │
│  │  │        (伴飞电脑)                     │  │   │
│  │  │                                      │  │   │
│  │  │  ┌────────────┐  ┌──────────────┐   │  │   │
│  │  │  │ RealSense  │  │   电池       │   │  │   │
│  │  │  │ D435i      │  │   4S 5000mAh │   │  │   │
│  │  │  │ (相机)     │  │              │   │  │   │
│  │  │  └────────────┘  └──────────────┘   │  │   │
│  │  └──────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 2.2 重量分配

| 组件 | 重量 | 位置 |
|------|------|------|
| 机架 | 300g | 中心 |
| 飞控 | 50g | 中心 |
| Jetson Orin Nano | 100g | 中心偏后 |
| RealSense D435i | 75g | 前方 |
| 电池 | 400g | 中心 |
| 电机+螺旋桨 | 200g | 四角 |
| **总计** | ~1125g | - |

---

## 3. 软件环境配置

### 3.1 Jetson Orin Nano 配置

```bash
# 1. 刷写JetPack
# 使用NVIDIA SDK Manager刷写JetPack 5.1+

# 2. 设置性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

# 3. 安装Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh

# 4. 创建环境
conda create -n dronevla python=3.10 -y
conda activate dronevla

# 5. 安装PyTorch (Jetson版本)
# 从NVIDIA论坛下载预编译版本
pip install torch-2.0.0+nv23.05-cp310-cp310-linux_aarch64.whl

# 6. 安装依赖
pip install pyserial numpy opencv-python-headless
```

### 3.2 MAVLink连接设置

```python
# 连接到Pixhawk飞控
from pymavlink import mavutil

# 串口连接（Jetson <-> Pixhawk）
master = mavutil.mavlink_connection(
    '/dev/ttyUSB0',  # 或 /dev/ttyACM0
    baud=57600
)

# 等待心跳
master.wait_heartbeat()
print(f"已连接到飞控 (system {master.target_system})")
```

---

## 4. 模型部署

### 4.1 模型优化

```bash
# 1. 导出为ONNX格式
python scripts/export_onnx.py --model logs/best_model.pt

# 2. 使用TensorRT优化（Jetson专用）
python scripts/optimize_tensorrt.py --onnx logs/model.onnx --output logs/model.trt

# 3. 验证优化后的模型
python scripts/verify_trt.py --model logs/model.trt
```

### 4.2 实时推理代码

```python
import torch
import cv2
import numpy as np
from pymavlink import mavutil
import time

class DroneVLAController:
    def __init__(self, model_path, camera_id=0):
        # 加载模型
        self.model = self.load_model(model_path)
        self.model.eval()

        # 初始化相机
        self.cap = cv2.VideoCapture(camera_id)

        # 连接飞控
        self.master = mavutil.mavlink_connection('/dev/ttyUSB0', baud=57600)
        self.master.wait_heartbeat()

        # 状态缓存
        self.state = np.zeros(12, dtype=np.float32)
        self.frame_buffer = []

    def load_model(self, path):
        """加载VLA模型"""
        from src.models.drone_vla import DroneVLA
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = DroneVLA(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def get_frame(self):
        """获取相机帧"""
        ret, frame = self.cap.read()
        if ret:
            # 调整大小
            frame = cv2.resize(frame, (64, 64))
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame.astype(np.float32) / 255.0
        return None

    def get_state(self):
        """从飞控获取状态"""
        # 获取位置
        pos = self.master.recv_match(type='LOCAL_POSITION_NED', blocking=True)
        # 获取姿态
        att = self.master.recv_match(type='ATTITUDE', blocking=True)

        if pos and att:
            self.state[:3] = [pos.x, pos.y, pos.z]
            self.state[3:6] = [pos.vx, pos.vy, pos.vz]
            self.state[6:9] = [att.roll, att.pitch, att.yaw]

        return self.state

    def send_velocity_command(self, vx, vy, vz, yaw_rate):
        """发送速度指令到飞控"""
        self.master.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,  # type_mask: 只使用速度
            0, 0, 0,  # x, y, z (忽略)
            vx, vy, vz,  # vx, vy, vz
            0, 0, 0,  # ax, ay, az (忽略)
            yaw_rate, 0  # yaw_rate, yaw (忽略)
        )

    def predict_action(self, instruction):
        """使用VLA模型预测动作"""
        # 准备输入
        frame = self.get_frame()
        if frame is None:
            return np.zeros(4)

        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > 4:
            self.frame_buffer.pop(0)

        # 填充到4帧
        while len(self.frame_buffer) < 4:
            self.frame_buffer.insert(0, self.frame_buffer[0])

        images = torch.FloatTensor(np.array(self.frame_buffer)).permute(0, 3, 1, 2).unsqueeze(0)
        state = torch.FloatTensor(self.state).unsqueeze(0)

        # 模型推理
        with torch.no_grad():
            outputs = self.model(images, [instruction], state)
            action = outputs['actions'][0].numpy()

        return action

    def run(self, instruction="hover at current position", duration=60):
        """运行VLA控制循环"""
        print(f"开始执行: {instruction}")
        print(f"持续时间: {duration}秒")

        start_time = time.time()
        dt = 0.1  # 10Hz控制频率

        while time.time() - start_time < duration:
            loop_start = time.time()

            # 获取状态
            self.get_state()

            # 预测动作
            action = self.predict_action(instruction)

            # 发送指令
            self.send_velocity_command(
                action[0], action[1], action[2], action[3]
            )

            # 保持控制频率
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

        # 停止
        self.send_velocity_command(0, 0, 0, 0)
        print("执行完成")

    def cleanup(self):
        """清理资源"""
        self.cap.release()
        self.master.close()


if __name__ == "__main__":
    controller = DroneVLAController("logs/best_model.pt")

    try:
        # 示例：执行悬停任务
        controller.run("hover at current position", duration=30)
    finally:
        controller.cleanup()
```

---

## 5. 飞行测试

### 5.1 测试流程

```
阶段1：地面测试
├── 确认MAVLink连接正常
├── 验证传感器数据
├── 测试电机响应
└── 检查遥控器覆盖

阶段2：低空悬停测试
├── 手动起飞到1米
├── 切换到VLA模式
├── 测试悬停稳定性
└── 观察10秒

阶段3：简单导航测试
├── 设置简单目标点
├── 观察VLA响应
├── 手动覆盖测试
└── 记录飞行日志

阶段4：复杂任务测试
├── 避障测试
├── 跟踪测试
├── 多目标切换
└── 异常处理测试
```

### 5.2 参数调优

| 参数 | 初始值 | 调优建议 |
|------|--------|----------|
| 控制频率 | 10Hz | 根据模型推理速度调整 |
| 最大速度 | 2 m/s | 室内降低到0.5 m/s |
| 到达阈值 | 1.5 m | 根据任务精度调整 |
| 超时时间 | 100步 | 根据任务复杂度调整 |

---

## 6. 安全注意事项

### 6.1 必须遵守的安全规则

```
1. 始终保持遥控器在手，随时可以手动覆盖
2. 设置失控保护：信号丢失时自动悬停或返航
3. 首次飞行在开阔场地，远离人群和建筑
4. 电池电量低于20%时立即返航
5. 风速超过5 m/s时不要飞行
6. 每次飞行前检查螺旋桨和电机
7. 不要在室内使用GPS模式
8. 保持视距飞行（VLOS）
```

### 6.2 紧急处理

```python
# 紧急情况处理代码
def emergency_stop(self):
    """紧急停止"""
    # 发送悬停指令
    self.send_velocity_command(0, 0, 0, 0)

    # 尝试降落
    self.master.mav.command_long_send(
        self.master.target_system,
        self.master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0
    )

    print("紧急降落指令已发送")
```

---

## 常见问题

### Q: 模型推理太慢怎么办？
A: 使用TensorRT优化，或减小模型尺寸。目标是控制频率>5Hz。

### Q: 飞行不稳定怎么办？
A: 检查PID参数，确保传感器数据正常，降低VLA控制权重。

### Q: 如何记录飞行数据用于后续训练？
A: 使用MAVLink日志记录，或自定义数据收集脚本。

---

*最后更新：2026-05-11*
