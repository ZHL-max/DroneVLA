# 多平台安装配置手册
## Windows / Linux Ubuntu 22.04 / macOS

---

## 目录

1. [Windows 安装指南](#1-windows-安装指南)
2. [Linux Ubuntu 22.04 安装指南](#2-linux-ubuntu-2204-安装指南)
3. [macOS 安装指南](#3-macos-安装指南)
4. [Jetson Orin Nano 安装指南](#4-jetson-orin-nano-安装指南)
5. [常见问题排查](#5-常见问题排查)

---

## 1. Windows 安装指南

### 1.1 系统要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| **操作系统** | Windows 10 64位 | Windows 11 |
| **CPU** | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| **内存** | 16GB | 32GB |
| **GPU** | NVIDIA GTX 1060 6GB | NVIDIA RTX 3080+ |
| **存储** | 50GB SSD | 100GB NVMe SSD |
| **CUDA** | 11.8 | 12.1+ |

### 1.2 安装步骤

#### 步骤1：安装 Miniconda

```powershell
# 下载 Miniconda
# https://docs.conda.io/en/latest/miniconda.html

# 或使用 winget
winget install CondaForge.Miniconda

# 验证安装
conda --version
```

#### 步骤2：安装 CUDA（如果使用GPU）

```powershell
# 下载 CUDA Toolkit
# https://developer.nvidia.com/cuda-toolkit-archive

# 推荐版本：CUDA 12.1
# 安装后验证
nvcc --version
nvidia-smi
```

#### 步骤3：安装 Git

```powershell
# 使用 winget
winget install Git.Git

# 或下载安装
# https://git-scm.com/download/win

# 验证
git --version
```

#### 步骤4：克隆项目

```powershell
# 打开 PowerShell 或 Git Bash
cd D:\BH\github
git clone https://github.com/ZHL-max/DroneVLA.git
cd DroneVLA
```

#### 步骤5：创建 Conda 环境

```powershell
# 创建环境
conda create -n dronevla python=3.10 -y
conda activate dronevla

# 安装 PyTorch (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装项目依赖
pip install -e .

# 安装额外依赖
pip install pyrealsense2  # RealSense相机
pip install depthai       # OAK-D相机
```

#### 步骤6：安装仿真环境

```powershell
# 安装 PyBullet
pip install pybullet

# 安装 Gymnasium
pip install gymnasium

# 验证安装
python -c "import pybullet; print('PyBullet OK')"
python -c "import gymnasium; print('Gymnasium OK')"
```

#### 步骤7：验证安装

```powershell
# 运行测试
python tests/test_models.py

# 运行Demo
cd demos/01_simple_vla
python simple_vla.py
```

### 1.3 Windows 特殊配置

#### PowerShell 执行策略

```powershell
# 如果遇到脚本执行错误
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 环境变量

```powershell
# 添加 CUDA 到 PATH
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
```

---

## 2. Linux Ubuntu 22.04 安装指南

### 2.1 系统要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| **操作系统** | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| **CPU** | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| **内存** | 16GB | 32GB |
| **GPU** | NVIDIA GTX 1060 6GB | NVIDIA RTX 3080+ |
| **存储** | 50GB SSD | 100GB NVMe SSD |

### 2.2 安装步骤

#### 步骤1：系统更新

```bash
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y git curl wget build-essential cmake
sudo apt install -y python3-pip python3-venv
```

#### 步骤2：安装 NVIDIA 驱动

```bash
# 查看推荐驱动
ubuntu-drivers devices

# 安装推荐驱动
sudo ubuntu-drivers autoinstall

# 重启
sudo reboot

# 验证
nvidia-smi
```

#### 步骤3：安装 CUDA Toolkit

```bash
# 下载 CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

# 安装
sudo sh cuda_12.1.0_530.30.02_linux.run

# 添加环境变量
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvcc --version
```

#### 步骤4：安装 Miniconda

```bash
# 下载
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh

# 重新加载shell
source ~/.bashrc

# 验证
conda --version
```

#### 步骤5：克隆项目

```bash
cd ~/github
git clone https://github.com/ZHL-max/DroneVLA.git
cd DroneVLA
```

#### 步骤6：创建 Conda 环境

```bash
# 创建环境
conda create -n dronevla python=3.10 -y
conda activate dronevla

# 安装 PyTorch (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装项目依赖
pip install -e .

# 安装相机驱动
pip install pyrealsense2  # RealSense
pip install depthai       # OAK-D
```

#### 步骤7：安装 RealSense 驱动

```bash
# 添加密钥
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE

# 添加仓库
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo jammy main"

# 安装
sudo apt update
sudo apt install -y librealsense2-dkms librealsense2-utils librealsense2-dev

# 验证
realsense-viewer
```

#### 步骤8：安装仿真环境

```bash
# PyBullet
pip install pybullet

# Gymnasium
pip install gymnasium

# ROS2 (可选，用于真实无人机)
sudo apt install -y ros-humble-desktop
```

#### 步骤9：验证安装

```bash
# 运行测试
python tests/test_models.py

# 运行Demo
cd demos/01_simple_vla
python simple_vla.py
```

### 2.3 Ubuntu 特殊配置

#### 用户权限

```bash
# 将用户添加到dialout组（串口访问）
sudo usermod -a -G dialout $USER

# 将用户添加到video组（摄像头访问）
sudo usermod -a -G video $USER

# 重新登录生效
```

#### udev 规则（RealSense）

```bash
# 创建udev规则
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && udevadm trigger
```

---

## 3. macOS 安装指南

### 3.1 系统要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| **操作系统** | macOS 12 Monterey | macOS 14 Sonoma |
| **芯片** | Intel / Apple M1 | Apple M2+ |
| **内存** | 16GB | 32GB |
| **存储** | 50GB | 100GB SSD |

### 3.2 安装步骤

#### 步骤1：安装 Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 验证
brew --version
```

#### 步骤2：安装基础工具

```bash
# Git
brew install git

# CMake
brew install cmake

# Python
brew install python@3.10
```

#### 步骤3：安装 Miniconda

```bash
# 下载
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# 安装 (Apple Silicon)
bash Miniconda3-latest-MacOSX-arm64.sh

# 或安装 (Intel)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# 重新加载shell
source ~/.zshrc
```

#### 步骤4：克隆项目

```bash
cd ~/github
git clone https://github.com/ZHL-max/DroneVLA.git
cd DroneVLA
```

#### 步骤5：创建 Conda 环境

```bash
# 创建环境
conda create -n dronevla python=3.10 -y
conda activate dronevla

# 安装 PyTorch (Apple Silicon)
conda install pytorch torchvision torchaudio -c pytorch -y

# 或安装 PyTorch (Intel)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 安装项目依赖
pip install -e .
```

#### 步骤6：安装相机驱动

```bash
# RealSense (macOS支持有限)
brew install librealsense

# USB摄像头（使用OpenCV）
pip install opencv-python
```

#### 步骤7：验证安装

```bash
# 运行测试
python tests/test_models.py

# 运行Demo
cd demos/01_simple_vla
python simple_vla.py
```

### 3.3 macOS 特殊配置

#### Apple Silicon 加速

```bash
# PyTorch MPS 加速 (Apple Silicon)
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

#### 权限设置

```bash
# 摄像头权限
# 系统设置 → 隐私与安全 → 摄像头 → 允许终端访问

# 串口权限（用于飞控连接）
# 系统设置 → 隐私与安全 → 完全磁盘访问权限 → 添加终端
```

---

## 4. Jetson Orin Nano 安装指南

### 4.1 系统要求

| 项目 | 要求 |
|------|------|
| **JetPack** | 5.1+ |
| **CUDA** | 11.4+ |
| **cuDNN** | 8.6+ |
| **TensorRT** | 8.5+ |

### 4.2 安装步骤

#### 步骤1：刷写系统

```bash
# 使用 NVIDIA SDK Manager 刷写 JetPack
# https://developer.nvidia.com/embedded/jetpack

# 或使用 SD 卡镜像
# https://developer.nvidia.com/embedded/jetpack-sdk-sd-card
```

#### 步骤2：系统配置

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y git curl wget python3-pip

# 设置性能模式
sudo nvpmodel -m 0  # MAXN模式
sudo jetson_clocks
```

#### 步骤3：安装 Miniconda

```bash
# 下载 (aarch64版本)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh

# 安装
bash Miniconda3-latest-Linux-aarch64.sh
```

#### 步骤4：创建环境

```bash
# 创建环境
conda create -n dronevla python=3.10 -y
conda activate dronevla

# 安装 PyTorch (Jetson)
# 从 NVIDIA 下载预编译版本
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

# 安装项目依赖
pip install -e .
```

#### 步骤5：安装相机驱动

```bash
# RealSense
sudo apt install -y librealsense2-utils librealsense2-dev

# CSI摄像头 (Pi Camera)
sudo apt install -y libcamera-dev
pip install picamera2
```

#### 步骤6：优化推理

```bash
# 安装 TensorRT
sudo apt install -y tensorrt

# 安装 ONNX Runtime (GPU)
pip install onnxruntime-gpu

# 模型转换
python scripts/export_onnx.py --model logs/best_model.pt
```

---

## 5. 常见问题排查

### 5.1 CUDA 相关

| 问题 | 解决方案 |
|------|----------|
| CUDA不可用 | 检查 `nvidia-smi` 输出 |
| 版本不匹配 | 确保PyTorch CUDA版本与系统一致 |
| 内存不足 | 减小batch size或使用混合精度 |

### 5.2 相机相关

| 问题 | 解决方案 |
|------|----------|
| 相机无法识别 | 检查USB连接、安装驱动 |
| 权限被拒 | 添加用户到video组 |
| 图像卡顿 | 降低分辨率或帧率 |

### 5.3 依赖相关

| 问题 | 解决方案 |
|------|----------|
| pip安装失败 | 升级pip: `pip install --upgrade pip` |
| 版本冲突 | 使用新的conda环境 |
| 编译错误 | 安装build-essential |

---

*最后更新：2026-05-11*
