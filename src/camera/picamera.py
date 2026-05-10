"""
Raspberry Pi Camera 驱动

支持 Raspberry Pi Camera Module v2/v3
通过CSI接口连接

安装依赖：
    pip install picamera2
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from .camera_base import CameraBase


class PiCamera(CameraBase):
    """
    Raspberry Pi Camera

    使用示例：
        with PiCamera() as camera:
            rgb = camera.get_frame()
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化 Pi Camera

        Args:
            config: 配置参数
                - width: 图像宽度 (默认640)
                - height: 图像高度 (默认480)
                - fps: 帧率 (默认30)
                - camera_id: 相机ID (默认0)
        """
        super().__init__(config)

        self.width = self.config.get("width", 640)
        self.height = self.config.get("height", 480)
        self.fps = self.config.get("fps", 30)
        self.camera_id = self.config.get("camera_id", 0)

        self.camera = None

    def connect(self) -> bool:
        """连接 Pi Camera"""
        try:
            from picamera2 import Picamera2

            self.camera = Picamera2(self.camera_id)

            # 配置相机
            config = self.camera.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameRate": self.fps}
            )
            self.camera.configure(config)

            # 启动相机
            self.camera.start()

            self.is_connected = True
            print(f"Pi Camera 连接成功: {self.width}x{self.height} @ {self.fps}fps")
            return True

        except ImportError:
            print("错误：请安装 picamera2: pip install picamera2")
            print("注意：picamera2 仅在 Raspberry Pi 上可用")
            return False
        except Exception as e:
            print(f"Pi Camera 连接失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self.camera:
            self.camera.stop()
            self.camera.close()
            self.camera = None
        self.is_connected = False
        self.is_streaming = False
        print("Pi Camera 已断开")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        获取RGB图像

        Returns:
            np.ndarray: RGB图像 (H, W, 3)
        """
        if not self.is_connected or not self.camera:
            return None

        try:
            # picamera2 直接返回RGB
            frame = self.camera.capture_array()
            return frame

        except Exception as e:
            print(f"获取图像失败: {e}")
            return None

    def get_depth(self) -> Optional[np.ndarray]:
        """Pi Camera 不支持深度"""
        return None
