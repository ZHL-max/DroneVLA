# 无人机控制基础
## 理解无人机飞行控制原理

---

## 1. 坐标系

### 机体坐标系 (Body Frame)

```
        机头方向 (X轴)
            ↑
            │
            │
    ────────┼────────→ Y轴 (右翼方向)
            │
            │
            ↓
        Z轴 (向下)

特点：
- 原点在无人机重心
- X轴指向机头
- Y轴指向右翼
- Z轴垂直向下（符合右手定则）
```

### 世界坐标系 (World Frame)

```
        Z轴 (向上)
            ↑
            │
            │
            │
            └──────────→ X轴 (北)
           /
          /
         ↓
        Y轴 (东)

特点：
- 固定不动
- 通常使用NED（北东地）或ENU（东北天）坐标系
```

### 坐标转换

```python
import numpy as np
from scipy.spatial.transform import Rotation

def body_to_world(body_pos, body_attitude, point_in_body):
    """
    将机体坐标系的点转换到世界坐标系

    Args:
        body_pos: 无人机在世界坐标系的位置 [x, y, z]
        body_attitude: 无人机姿态 [roll, pitch, yaw] (弧度)
        point_in_body: 机体坐标系中的点 [x, y, z]

    Returns:
        世界坐标系中的点
    """
    # 旋转矩阵
    r = Rotation.from_euler('xyz', body_attitude)
    rotation_matrix = r.as_matrix()

    # 旋转 + 平移
    world_point = rotation_matrix @ point_in_body + body_pos
    return world_point
```

---

## 2. 姿态表示

### 欧拉角 (Euler Angles)

```python
# 欧拉角定义
roll = 0.1    # 滚转角 (绕X轴旋转), 弧度
pitch = -0.05 # 俯仰角 (绕Y轴旋转), 弧度
yaw = 1.57    # 偏航角 (绕Z轴旋转), 弧度

# 直观理解
# roll: 左右倾斜（翻滚）
# pitch: 前后倾斜（俯仰）
# yaw: 左右转向（偏航）

# 欧拉角的缺点：万向锁（Gimbal Lock）
# 当pitch接近±90°时，roll和yaw会重合
```

### 四元数 (Quaternion)

```python
# 四元数表示：q = [w, x, y, z]
# 优点：无万向锁，计算效率高

from scipy.spatial.transform import Rotation

# 欧拉角转四元数
euler = [roll, pitch, yaw]
r = Rotation.from_euler('xyz', euler)
quaternion = r.as_quat()  # [x, y, z, w] 格式

# 四元数转欧拉角
r_back = Rotation.from_quat(quaternion)
euler_back = r_back.as_euler('xyz')
```

### 旋转矩阵

```python
# 3x3旋转矩阵
# 描述从机体坐标系到世界坐标系的旋转

def euler_to_rotation_matrix(roll, pitch, yaw):
    """欧拉角转旋转矩阵"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])
    return R
```

---

## 3. 运动学模型

### 位置运动学

```python
def update_position(state, action, dt):
    """
    更新无人机位置

    Args:
        state: 当前状态 [x, y, z, vx, vy, vz, ...]
        action: 速度指令 [vx_cmd, vy_cmd, vz_cmd, yaw_rate]
        dt: 时间步长 (秒)

    Returns:
        更新后的状态
    """
    x, y, z = state[:3]
    vx, vy, vz = state[3:6]

    # 速度指令
    vx_cmd, vy_cmd, vz_cmd = action[:3]

    # 简单动力学模型（一阶响应）
    tau = 0.3  # 时间常数，控制响应速度
    vx_new = vx + (vx_cmd - vx) * dt / tau
    vy_new = vy + (vy_cmd - vy) * dt / tau
    vz_new = vz + (vz_cmd - vz) * dt / tau

    # 速度限幅
    max_vel = 2.0  # m/s
    vx_new = np.clip(vx_new, -max_vel, max_vel)
    vy_new = np.clip(vy_new, -max_vel, max_vel)
    vz_new = np.clip(vz_new, -max_vel, max_vel)

    # 位置更新
    x_new = x + vx_new * dt
    y_new = y + vy_new * dt
    z_new = z + vz_new * dt

    # 位置约束
    x_new = np.clip(x_new, 0, 20)
    y_new = np.clip(y_new, 0, 20)
    z_new = max(0, z_new)  # 不能低于地面

    return np.array([x_new, y_new, z_new, vx_new, vy_new, vz_new])
```

### 姿态运动学

```python
def update_attitude(state, action, dt):
    """
    更新无人机姿态

    Args:
        state: 当前状态 [..., roll, pitch, yaw, wx, wy, wz]
        action: 姿态指令 [..., yaw_rate]
        dt: 时间步长

    Returns:
        更新后的姿态
    """
    roll, pitch, yaw = state[6:9]
    wx, wy, wz = state[9:12]

    yaw_rate = action[3]  # 偏航角速度

    # 姿态更新（简化模型）
    yaw_new = yaw + yaw_rate * dt

    # 滚转和俯仰通常由底层控制器处理
    # 这里简化为0（假设水平飞行）
    roll_new = roll * 0.9  # 趋向水平
    pitch_new = pitch * 0.9

    # 角速度
    wz_new = yaw_rate

    return np.array([roll_new, pitch_new, yaw_new, wx, wy, wz_new])
```

---

## 4. PID控制器

### PID原理

```
误差 = 目标值 - 当前值

P（比例）：误差越大，输出越大
I（积分）：消除稳态误差
D（微分）：预测误差变化趋势，减少超调

输出 = Kp * e + Ki * ∫e + Kd * de/dt
```

### 代码实现

```python
class PIDController:
    def __init__(self, kp, ki, kd, limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limits = limits  # 输出限幅

        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        """计算PID输出"""
        # 比例项
        p_term = self.kp * error

        # 积分项（带抗饱和）
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10, 10)  # 抗饱和
        i_term = self.ki * self.integral

        # 微分项
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        # 总输出
        output = p_term + i_term + d_term

        # 限幅
        if self.limits:
            output = np.clip(output, -self.limits, self.limits)

        # 保存误差
        self.prev_error = error

        return output

    def reset(self):
        """重置控制器"""
        self.prev_error = 0
        self.integral = 0
```

### 无人机PID控制示例

```python
def pid_control(current_pos, goal_pos, dt):
    """
    PID位置控制

    Returns:
        速度指令 [vx, vy, vz]
    """
    # 位置误差
    error = goal_pos - current_pos

    # PID参数（每个轴独立）
    kp = np.array([1.0, 1.0, 1.2])  # z轴响应更快
    ki = np.array([0.1, 0.1, 0.15])
    kd = np.array([0.3, 0.3, 0.4])

    # 计算控制量
    velocity_cmd = kp * error

    # 限幅
    max_vel = 2.0
    velocity_cmd = np.clip(velocity_cmd, -max_vel, max_vel)

    return velocity_cmd
```

---

## 5. 避障控制

### 势场法 (Potential Field)

```python
def potential_field_avoidance(current_pos, goal_pos, obstacles, k_att=1.0, k_rep=5.0, d0=3.0):
    """
    势场法避障

    Args:
        current_pos: 当前位置
        goal_pos: 目标位置
        obstacles: 障碍物位置列表
        k_att: 引力系数
        k_rep: 斥力系数
        d0: 斥力作用范围

    Returns:
        合力方向（速度指令）
    """
    # 引力（指向目标）
    att_force = k_att * (goal_pos - current_pos)

    # 斥力（远离障碍物）
    rep_force = np.zeros(3)
    for obs in obstacles:
        diff = current_pos - obs
        dist = np.linalg.norm(diff)

        if dist < d0 and dist > 0:
            # 斥力公式：F = k_rep * (1/d - 1/d0) * (1/d^2) * direction
            magnitude = k_rep * (1/dist - 1/d0) * (1/dist**2)
            direction = diff / dist
            rep_force += magnitude * direction

    # 合力
    total_force = att_force + rep_force

    # 归一化到最大速度
    max_vel = 2.0
    if np.linalg.norm(total_force) > max_vel:
        total_force = total_force / np.linalg.norm(total_force) * max_vel

    return total_force
```

---

## 6. 常见飞行模式

### 模式说明

```
1. 手动模式 (Manual)
   - 飞手完全控制
   - 适合有经验的操作者

2. 稳定模式 (Stabilize)
   - 自动保持水平
   - 手动控制位置

3. 定高模式 (Altitude Hold)
   - 自动保持高度
   - 手动控制水平位置

4. 定点模式 (Position Hold)
   - 自动保持位置
   - 需要GPS或视觉定位

5. 自动模式 (Auto)
   - 按预设航线飞行
   - 由VLA模型控制

6. 返回模式 (Return to Launch)
   - 自动返回起飞点
   - 安全模式
```

---

## 7. 安全注意事项

### 飞行前检查

```python
preflight_checklist = {
    "电池电量": "> 50%",
    "GPS信号": ">= 10颗卫星",
    "遥控器连接": "已配对",
    "螺旋桨": "无损坏，安装牢固",
    "电机": "运转正常，无异响",
    "传感器": "IMU、气压计、磁力计正常",
    "飞行区域": "无障碍物，无禁飞区",
    "天气条件": "风速 < 5级，无降雨"
}
```

### 紧急处理

```python
emergency_procedures = {
    "电量低": "立即降落或返航",
    "信号丢失": "启动返航模式",
    "GPS丢失": "切换手动模式，降落",
    "电机故障": "立即降落",
    "碰撞": "检查损伤，评估是否可飞"
}
```

---

## 8. 仿真环境

### PyBullet无人机仿真

```python
import pybullet as p
import pybullet_data

def setup_drone_simulation():
    """设置无人机仿真环境"""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # 加载地面
    p.loadURDF("plane.urdf")

    # 加载无人机（简化为立方体）
    drone_id = p.loadURDF("cube.urdf", [0, 0, 1])

    return drone_id

def apply_velocity_control(drone_id, velocity_cmd):
    """应用速度控制"""
    vx, vy, vz = velocity_cmd
    p.resetBaseVelocity(drone_id, [vx, vy, vz])
```

---

*上一节：[无人机vs机械臂](01_Drone_vs_Manipulator.md) | 下一节：[传感器融合](03_Sensor_Fusion.md)*
