"""
OAK-D 相机驱动

支持 Luxonis OAK-D 系列相机
提供RGB图像、深度图、神经计算

安装依赖：
    pip install depthai
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from .camera_base import CameraBase


class OAKDCamera(CameraBase):
    """
    Luxonis OAK-D 相机

    使用示例：
        with OAKDCamera() as camera:
            rgb = camera.get_frame()
            depth = camera.get_depth()
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化 OAK-D 相机

        Args:
            config: 配置参数
                - width: 图像宽度 (默认640)
                - height: 图像高度 (默认480)
                - fps: 帧率 (默认30)
                - enable_depth: 是否启用深度 (默认True)
        """
        super().__init__(config)

        self.width = self.config.get("width", 640)
        self.height = self.config.get("height", 480)
        self.fps = self.config.get("fps", 30)
        self.enable_depth = self.config.get("enable_depth", True)

        self.device = None
        self.rgb_queue = None
        self.depth_queue = None

    def connect(self) -> bool:
        """连接 OAK-D 相机"""
        try:
            import depthai as dai

            # 创建pipeline
            pipeline = dai.Pipeline()

            # RGB相机
            cam_rgb = pipeline.createColorCamera()
            cam_rgb.setPreviewSize(self.width, self.height)
            cam_rgb.setInterleaved(False)
            cam_rgb.setFps(self.fps)

            xout_rgb = pipeline.createXLinkOut()
            xout_rgb.setStreamName("rgb")
            cam_rgb.preview.link(xout_rgb.input)

            # 深度相机
            if self.enable_depth:
                mono_left = pipeline.createMonoCamera()
                mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
                mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)

                mono_right = pipeline.createMonoCamera()
                mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
                mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

                depth = pipeline.createStereoDepth()
                depth.setConfidenceThreshold(200)
                mono_left.out.link(depth.left)
                mono_right.out.link(depth.right)

                xout_depth = pipeline.createXLinkOut()
                xout_depth.setStreamName("depth")
                depth.disparity.link(xout_depth.input)

            # 启动设备
            self.device = dai.Device(pipeline)
            self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            if self.enable_depth:
                self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            self.is_connected = True
            print(f"OAK-D 连接成功: {self.width}x{self.height} @ {self.fps}fps")
            return True

        except ImportError:
            print("错误：请安装 depthai: pip install depthai")
            return False
        except Exception as e:
            print(f"OAK-D 连接失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self.device:
            self.device.close()
            self.device = None
        self.is_connected = False
        self.is_streaming = False
        print("OAK-D 已断开")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        获取RGB图像

        Returns:
            np.ndarray: RGB图像 (H, W, 3)
        """
        if not self.is_connected or not self.rgb_queue:
            return None

        try:
            in_rgb = self.rgb_queue.get()
            rgb_image = in_rgb.getCvFrame()
            return rgb_image

        except Exception as e:
            print(f"获取图像失败: {e}")
            return None

    def get_depth(self) -> Optional[np.ndarray]:
        """
        获取深度图

        Returns:
            np.ndarray: 深度图 (H, W)
        """
        if not self.is_connected or not self.depth_queue:
            return None

        try:
            in_depth = self.depth_queue.get()
            depth_image = in_depth.getFrame()
            return depth_image

        except Exception as e:
            print(f"获取深度图失败: {e}")
            return None
