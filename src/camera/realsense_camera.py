"""
Intel RealSense 相机驱动

支持 RealSense D435i, D455, L515 等型号
提供RGB图像、深度图、IMU数据

安装依赖：
    pip install pyrealsense2
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from .camera_base import CameraBase


class RealSenseCamera(CameraBase):
    """
    Intel RealSense 相机

    使用示例：
        with RealSenseCamera() as camera:
            rgb = camera.get_frame()
            depth = camera.get_depth()
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化 RealSense 相机

        Args:
            config: 配置参数
                - width: 图像宽度 (默认640)
                - height: 图像高度 (默认480)
                - fps: 帧率 (默认30)
                - enable_depth: 是否启用深度 (默认True)
                - enable_imu: 是否启用IMU (默认False)
                - serial_number: 相机序列号 (默认None，使用第一个)
        """
        super().__init__(config)

        self.width = self.config.get("width", 640)
        self.height = self.config.get("height", 480)
        self.fps = self.config.get("fps", 30)
        self.enable_depth = self.config.get("enable_depth", True)
        self.enable_imu = self.config.get("enable_imu", False)
        self.serial_number = self.config.get("serial_number", None)

        self.pipeline = None
        self.align = None
        self.intrinsics = None

    def connect(self) -> bool:
        """连接 RealSense 相机"""
        try:
            import pyrealsense2 as rs

            self.pipeline = rs.pipeline()
            config = rs.config()

            # 配置序列号
            if self.serial_number:
                config.enable_device(self.serial_number)

            # 配置RGB流
            config.enable_stream(
                rs.stream.color,
                self.width, self.height,
                rs.format.bgr8,
                self.fps
            )

            # 配置深度流
            if self.enable_depth:
                config.enable_stream(
                    rs.stream.depth,
                    self.width, self.height,
                    rs.format.z16,
                    self.fps
                )

            # 启动管道
            profile = self.pipeline.start(config)

            # 获取内参
            color_stream = profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

            # 对齐到RGB
            if self.enable_depth:
                align_to = rs.stream.color
                self.align = rs.align(align_to)

            # 等待几帧稳定
            for _ in range(30):
                self.pipeline.wait_for_frames()

            self.is_connected = True
            print(f"RealSense 连接成功: {self.width}x{self.height} @ {self.fps}fps")
            return True

        except ImportError:
            print("错误：请安装 pyrealsense2: pip install pyrealsense2")
            return False
        except Exception as e:
            print(f"RealSense 连接失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        self.is_connected = False
        self.is_streaming = False
        print("RealSense 已断开")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        获取RGB图像

        Returns:
            np.ndarray: RGB图像 (H, W, 3)
        """
        if not self.is_connected:
            return None

        try:
            import pyrealsense2 as rs

            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                return None

            # 转换为numpy数组 (BGR -> RGB)
            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = color_image[:, :, ::-1].copy()

            return rgb_image

        except Exception as e:
            print(f"获取图像失败: {e}")
            return None

    def get_depth(self) -> Optional[np.ndarray]:
        """
        获取深度图

        Returns:
            np.ndarray: 深度图 (H, W)，单位：米
        """
        if not self.is_connected or not self.enable_depth:
            return None

        try:
            import pyrealsense2 as rs

            frames = self.pipeline.wait_for_frames()

            if self.align:
                frames = self.align.process(frames)

            depth_frame = frames.get_depth_frame()

            if not depth_frame:
                return None

            # 转换为米
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
            depth_meters = depth_image * depth_scale

            return depth_meters

        except Exception as e:
            print(f"获取深度图失败: {e}")
            return None

    def get_imu(self) -> Optional[Dict[str, np.ndarray]]:
        """
        获取IMU数据（仅D435i支持）

        Returns:
            Dict: 包含accel和gyro数据
        """
        if not self.enable_imu:
            return None

        # D435i IMU需要单独配置
        # 这里返回None，实际使用需要额外配置
        return None

    def get_camera_info(self) -> Dict[str, Any]:
        """获取相机详细信息"""
        info = super().get_camera_info()
        info.update({
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "enable_depth": self.enable_depth,
            "enable_imu": self.enable_imu,
            "serial_number": self.serial_number
        })

        if self.intrinsics:
            info["intrinsics"] = {
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "ppx": self.intrinsics.ppx,
                "ppy": self.intrinsics.ppy
            }

        return info
