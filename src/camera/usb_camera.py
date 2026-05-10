"""
USB摄像头驱动

支持通用USB摄像头
提供RGB图像

安装依赖：
    pip install opencv-python
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from .camera_base import CameraBase


class USBCamera(CameraBase):
    """
    USB摄像头

    使用示例：
        with USBCamera(device_id=0) as camera:
            rgb = camera.get_frame()
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化USB摄像头

        Args:
            config: 配置参数
                - device_id: 设备ID (默认0)
                - width: 图像宽度 (默认640)
                - height: 图像高度 (默认480)
                - fps: 帧率 (默认30)
        """
        super().__init__(config)

        self.device_id = self.config.get("device_id", 0)
        self.width = self.config.get("width", 640)
        self.height = self.config.get("height", 480)
        self.fps = self.config.get("fps", 30)

        self.cap = None

    def connect(self) -> bool:
        """连接USB摄像头"""
        try:
            import cv2

            self.cap = cv2.VideoCapture(self.device_id)

            if not self.cap.isOpened():
                print(f"无法打开摄像头 {self.device_id}")
                return False

            # 设置分辨率和帧率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # 读取一帧验证
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取图像")
                return False

            self.is_connected = True
            print(f"USB摄像头 连接成功: {self.width}x{self.height} @ {self.fps}fps")
            return True

        except ImportError:
            print("错误：请安装 opencv-python: pip install opencv-python")
            return False
        except Exception as e:
            print(f"USB摄像头 连接失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        self.is_streaming = False
        print("USB摄像头 已断开")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        获取RGB图像

        Returns:
            np.ndarray: RGB图像 (H, W, 3)
        """
        if not self.is_connected or not self.cap:
            return None

        try:
            import cv2

            ret, frame = self.cap.read()

            if not ret:
                return None

            # BGR -> RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            return rgb_image

        except Exception as e:
            print(f"获取图像失败: {e}")
            return None

    def get_depth(self) -> Optional[np.ndarray]:
        """USB摄像头不支持深度"""
        return None
