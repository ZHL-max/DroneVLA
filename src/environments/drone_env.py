"""
DroneVLA 无人机环境

基于Gymnasium的无人机仿真环境，支持：
- 多种飞行任务（悬停、导航、跟踪）
- 视觉观测（第一人称视角）
- 语言指令条件
- 安全约束和边界检查

作者：DroneVLA Project
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from typing import Optional, Tuple, Dict, Any


class DroneEnv(gym.Env):
    """
    无人机仿真环境

    动作空间：
    - vx, vy, vz: 机体坐标系速度 (m/s)
    - yaw_rate: 偏航角速度 (rad/s)

    观测空间：
    - 图像: RGB相机图像 (可选)
    - 状态: [位置(3), 速度(3), 姿态(3), 目标位置(3)]

    奖励：
    - 距离奖励：越接近目标奖励越高
    - 安全惩罚：碰撞或超出边界
    - 任务奖励：完成特定任务目标
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: str = "hover",
        render_mode: Optional[str] = None,
        image_obs: bool = False,
        image_size: Tuple[int, int] = (64, 64),
        max_steps: int = 200,
        boundary: float = 5.0,
        **kwargs
    ):
        super().__init__()

        self.task = task
        self.render_mode = render_mode
        self.image_obs = image_obs
        self.image_size = image_size
        self.max_steps = max_steps
        self.boundary = boundary

        # 动作空间：[vx, vy, vz, yaw_rate]
        self.action_space = spaces.Box(
            low=np.array([-2.0, -2.0, -1.0, -1.0]),
            high=np.array([2.0, 2.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # 观测空间
        if image_obs:
            self.observation_space = spaces.Dict({
                "image": spaces.Box(
                    low=0, high=255,
                    shape=(image_size[0], image_size[1], 3),
                    dtype=np.uint8
                ),
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(12,),  # pos(3) + vel(3) + att(3) + goal(3)
                    dtype=np.float32
                )
            })
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(12,),
                dtype=np.float32
            )

        # PyBullet物理引擎
        self.physics_client = None
        self.drone_id = None
        self.goal_pos = None
        self.step_count = 0

        # 无人机参数
        self.mass = 0.5  # kg
        self.arm_length = 0.1  # m
        self.max_thrust = 2.0  # N

        # 任务特定的目标位置
        self._setup_task()

    def _setup_task(self):
        """根据任务类型设置目标"""
        if self.task == "hover":
            # 悬停任务：保持在随机目标位置
            self.goal_pos = np.array([0.0, 0.0, 2.0])
        elif self.task == "navigate":
            # 导航任务：到达指定位置
            self.goal_pos = np.random.uniform(-3, 3, size=3)
            self.goal_pos[2] = np.abs(self.goal_pos[2]) + 1.0  # 确保高度为正
        elif self.task == "track":
            # 跟踪任务：跟踪移动目标
            self.goal_pos = np.array([0.0, 0.0, 2.0])
            self.target_velocity = np.random.uniform(-0.5, 0.5, size=3)
        elif self.task == "avoid":
            # 避障任务：到达目标同时避开障碍物
            self.goal_pos = np.array([3.0, 0.0, 2.0])
            self.obstacles = [
                {"pos": np.array([1.5, 0.0, 2.0]), "radius": 0.3},
                {"pos": np.array([0.0, 1.0, 1.5]), "radius": 0.3},
            ]
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)

        # 初始化PyBullet
        if self.physics_client is not None:
            p.disconnect(self.physics_client)

        self.physics_client = p.connect(p.DIRECT if self.render_mode != "human" else p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)

        # 加载地面
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

        # 创建无人机（简化为球体）
        drone_pos = [0, 0, 0.5]
        drone_col = p.createCollisionShape(
            p.GEOM_SPHERE, radius=0.1,
            physicsClientId=self.physics_client
        )
        drone_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.1,
            rgbaColor=[0, 0, 1, 1],
            physicsClientId=self.physics_client
        )
        self.drone_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=drone_col,
            baseVisualShapeIndex=drone_vis,
            basePosition=drone_pos,
            physicsClientId=self.physics_client
        )

        # 重置任务
        self._setup_task()
        self.step_count = 0

        # 创建目标可视化
        goal_col = p.createCollisionShape(
            p.GEOM_SPHERE, radius=0.05,
            physicsClientId=self.physics_client
        )
        goal_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.05,
            rgbaColor=[1, 0, 0, 1],
            physicsClientId=self.physics_client
        )
        self.goal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=goal_col,
            baseVisualShapeIndex=goal_vis,
            basePosition=self.goal_pos.tolist(),
            physicsClientId=self.physics_client
        )

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步动作"""
        self.step_count += 1

        # 解析动作
        vx, vy, vz, yaw_rate = action

        # 获取当前状态
        pos, orn = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self.physics_client
        )
        vel, ang_vel = p.getBaseVelocity(
            self.drone_id, physicsClientId=self.physics_client
        )

        # 应用速度控制
        # 简化：直接设置速度（实际应该使用力/力矩控制）
        new_vel = [vx, vy, vz]
        p.resetBaseVelocity(
            self.drone_id,
            linearVelocity=new_vel,
            angularVelocity=[0, 0, yaw_rate],
            physicsClientId=self.physics_client
        )

        # 物理仿真步进
        p.stepSimulation(physicsClientId=self.physics_client)

        # 计算奖励
        reward = self._compute_reward(pos, vel)

        # 检查终止条件
        terminated = self._check_terminated(pos)
        truncated = self.step_count >= self.max_steps

        # 获取新的观测
        obs = self._get_obs()

        info = {
            "position": np.array(pos),
            "velocity": np.array(vel),
            "distance_to_goal": np.linalg.norm(np.array(pos) - self.goal_pos)
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """获取观测"""
        pos, orn = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self.physics_client
        )
        vel, ang_vel = p.getBaseVelocity(
            self.drone_id, physicsClientId=self.physics_client
        )

        # 欧拉角
        euler = p.getEulerFromQuaternion(orn)

        state = np.array([
            *pos,      # 位置 (3)
            *vel,      # 速度 (3)
            *euler,    # 姿态 (3)
            *self.goal_pos  # 目标位置 (3)
        ], dtype=np.float32)

        if self.image_obs:
            # 获取相机图像
            image = self._get_camera_image()
            return {"image": image, "state": state}
        else:
            return state

    def _get_camera_image(self) -> np.ndarray:
        """获取第一人称相机图像"""
        pos, orn = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self.physics_client
        )

        # 计算相机参数
        cam_pos = [pos[0], pos[1], pos[2] + 0.1]
        cam_target = [pos[0] + 1.0, pos[1], pos[2]]
        cam_up = [0, 0, 1]

        view_matrix = p.computeViewMatrix(
            cam_pos, cam_target, cam_up,
            physicsClientId=self.physics_client
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=100,
            physicsClientId=self.physics_client
        )

        # 渲染图像
        width, height = self.image_size
        _, _, img, _, _ = p.getCameraImage(
            width, height,
            view_matrix, proj_matrix,
            physicsClientId=self.physics_client
        )

        # 转换为RGB
        img = np.array(img).reshape(height, width, 4)[:, :, :3]

        return img

    def _compute_reward(self, position, velocity) -> float:
        """计算奖励"""
        pos = np.array(position)
        vel = np.array(velocity)

        # 距离奖励
        distance = np.linalg.norm(pos - self.goal_pos)
        distance_reward = -distance

        # 速度惩罚（鼓励平稳飞行）
        velocity_penalty = -0.1 * np.linalg.norm(vel)

        # 任务特定奖励
        task_reward = 0.0

        if self.task == "hover":
            # 悬停任务：到达目标后保持稳定
            if distance < 0.2:
                task_reward = 1.0
                # 额外奖励：保持稳定
                if np.linalg.norm(vel) < 0.1:
                    task_reward += 0.5

        elif self.task == "navigate":
            # 导航任务：到达目标
            if distance < 0.3:
                task_reward = 5.0

        elif self.task == "track":
            # 跟踪任务：跟踪移动目标
            self.goal_pos += self.target_velocity * 0.02
            # 边界检查
            if np.any(np.abs(self.goal_pos) > self.boundary):
                self.target_velocity *= -1
            task_reward = -distance * 0.5

        elif self.task == "avoid":
            # 避障任务：避开障碍物
            for obs in self.obstacles:
                obs_dist = np.linalg.norm(pos - obs["pos"])
                if obs_dist < obs["radius"] + 0.15:
                    task_reward -= 2.0  # 碰撞惩罚

            if distance < 0.3:
                task_reward += 10.0  # 到达目标奖励

        # 安全惩罚
        safety_penalty = 0.0
        if pos[2] < 0.1:  # 太低
            safety_penalty -= 1.0
        if np.any(np.abs(pos[:2]) > self.boundary):  # 超出边界
            safety_penalty -= 2.0

        total_reward = distance_reward + velocity_penalty + task_reward + safety_penalty

        return total_reward

    def _check_terminated(self, position) -> bool:
        """检查是否应该终止"""
        pos = np.array(position)

        # 碰撞地面
        if pos[2] < 0.05:
            return True

        # 超出边界
        if np.any(np.abs(pos[:2]) > self.boundary + 1.0):
            return True

        # 避障任务：碰撞障碍物
        if self.task == "avoid":
            for obs in self.obstacles:
                if np.linalg.norm(pos - obs["pos"]) < obs["radius"] + 0.1:
                    return True

        return False

    def render(self):
        """渲染环境"""
        if self.render_mode == "rgb_array":
            return self._get_camera_image()

    def close(self):
        """关闭环境"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


class DroneLanguageEnv(DroneEnv):
    """
    支持语言指令的无人机环境

    在基础环境上增加：
    - 语言指令条件
    - 更复杂的任务描述
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 语言指令模板
        self.instruction_templates = {
            "hover": [
                "hover at position {x:.1f}, {y:.1f}, {z:.1f}",
                "stay at {x:.1f}, {y:.1f}, {z:.1f}",
                "maintain position at {x:.1f}, {y:.1f}, {z:.1f}"
            ],
            "navigate": [
                "fly to {x:.1f}, {y:.1f}, {z:.1f}",
                "navigate to position {x:.1f}, {y:.1f}, {z:.1f}",
                "go to {x:.1f}, {y:.1f}, {z:.1f}"
            ],
            "track": [
                "follow the moving target",
                "track the object",
                "keep following the target"
            ],
            "avoid": [
                "reach the goal while avoiding obstacles",
                "fly to the target safely",
                "navigate to the goal without hitting obstacles"
            ]
        }

        self.current_instruction = ""

    def reset(self, **kwargs):
        """重置环境并生成新的语言指令"""
        obs, info = super().reset(**kwargs)

        # 生成语言指令
        self.current_instruction = self._generate_instruction()
        info["instruction"] = self.current_instruction

        return obs, info

    def _generate_instruction(self) -> str:
        """生成语言指令"""
        templates = self.instruction_templates[self.task]
        template = np.random.choice(templates)

        instruction = template.format(
            x=self.goal_pos[0],
            y=self.goal_pos[1],
            z=self.goal_pos[2]
        )

        return instruction

    def _get_obs(self):
        """获取观测（包含语言指令信息）"""
        obs = super()._get_obs()

        # 在实际应用中，这里会将语言指令编码为向量
        # 并与状态观测拼接

        return obs


# 注册环境
gym.register(
    id="DroneEnv-v0",
    entry_point="src.environments.drone_env:DroneEnv",
    max_episode_steps=200,
)

gym.register(
    id="DroneLanguageEnv-v0",
    entry_point="src.environments.drone_env:DroneLanguageEnv",
    max_episode_steps=200,
)
