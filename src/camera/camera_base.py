"""
相机基类

定义所有相机的统一接口
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, Dict, Any


class CameraBase(ABC):
    """
    相机基类

    所有相机实现都应继承此类，确保统一的接口
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化相机

        Args:
            config: 相机配置字典
        """
        self.config = config or {}
        self.is_connected = False
        self.is_streaming = False

    @abstractmethod
    def connect(self) -> bool:
        """
        连接相机

        Returns:
            bool: 连接是否成功
        """
        pass

    @abstractmethod
    def disconnect(self):
        """断开相机连接"""
        pass

    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """
        获取一帧图像

        Returns:
            np.ndarray: RGB图像 (H, W, 3)，失败返回None
        """
        pass

    @abstractmethod
    def get_depth(self) -> Optional[np.ndarray]:
        """
        获取深度图（如果支持）

        Returns:
            np.ndarray: 深度图 (H, W)，单位：米，不支持返回None
        """
        pass

    def get_rgbd(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取RGB+深度图

        Returns:
            Tuple[rgb, depth]: RGB和深度图，不支持返回(None, None)
        """
        rgb = self.get_frame()
        depth = self.get_depth()
        return rgb, depth

    def get_camera_info(self) -> Dict[str, Any]:
        """
        获取相机信息

        Returns:
            Dict: 包含内参、分辨率等信息
        """
        return {
            "type": self.__class__.__name__,
            "is_connected": self.is_connected,
            "is_streaming": self.is_streaming,
            "config": self.config
        }

    def start_streaming(self):
        """开始连续采集"""
        self.is_streaming = True

    def stop_streaming(self):
        """停止连续采集"""
        self.is_streaming = False

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
        return False
