# 环境配置教程
## 从零搭建VLA开发环境

---

## 1. 硬件需求

### 最低配置
```
CPU: Intel i5 / AMD Ryzen 5
RAM: 16GB
GPU: NVIDIA GTX 1060 6GB（可选，CPU也能跑轻量模型）
硬盘: 20GB可用空间
```

### 推荐配置
```
CPU: Intel i7 / AMD Ryzen 7
RAM: 32GB
GPU: NVIDIA RTX 3060 12GB 或更高
硬盘: 50GB SSD
```

### 本项目实测环境
```
CPU: Intel i7
RAM: 32GB
GPU: NVIDIA RTX 4060 Laptop 8GB
系统: Windows 11
```

---

## 2. 软件环境

### 2.1 安装Conda

```bash
# 下载Miniconda
# https://docs.conda.io/en/latest/miniconda.html

# 或下载Anaconda
# https://www.anaconda.com/products/distribution

# 验证安装
conda --version
```

### 2.2 创建虚拟环境

```bash
# 创建DroneVLA专用环境
conda create -n dronevla python=3.10 -y

# 激活环境
conda activate dronevla

# 验证Python版本
python --version  # 应该是3.10.x
```

### 2.3 安装PyTorch

```bash
# 方式1：CUDA 12.1（推荐，匹配RTX 4060）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 方式2：CUDA 11.8（旧GPU）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 方式3：仅CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 验证安装
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 2.4 安装项目依赖

```bash
# 进入项目目录
cd D:\BH\github\DroneVLA

# 安装所有依赖
pip install -r requirements.txt

# 或手动安装核心包
pip install numpy matplotlib tqdm pyyaml
pip install transformers  # 用于BERT编码器
pip install onnx onnxruntime  # 用于模型导出
```

---

## 3. 验证安装

### 3.1 运行验证脚本

```python
# test_environment.py
import torch
import numpy as np
import matplotlib

print("=" * 50)
print("环境验证")
print("=" * 50)

# Python版本
import sys
print(f"Python: {sys.version}")

# PyTorch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# NumPy
print(f"NumPy: {np.__version__}")

# Matplotlib
print(f"Matplotlib: {matplotlib.__version__}")

print("\n环境配置完成！")
```

### 3.2 运行测试

```bash
conda activate dronevla
python test_environment.py
```

预期输出：
```
==================================================
环境验证
==================================================
Python: 3.10.12
PyTorch: 2.5.1+cu121
CUDA可用: True
CUDA版本: 12.1
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
显存: 8.0 GB
NumPy: 1.24.3
Matplotlib: 3.7.2

环境配置完成！
```

---

## 4. 常见问题

### Q1: torch.cuda.is_available()返回False

```bash
# 检查NVIDIA驱动
nvidia-smi

# 如果驱动正常，重装PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 如果nvidia-smi报错，更新驱动
# https://www.nvidia.com/Download/index.aspx
```

### Q2: CUDA版本不匹配

```bash
# 查看CUDA版本
nvidia-smi  # 右上角显示CUDA Version

# 选择对应PyTorch版本
# CUDA 12.x → cu121
# CUDA 11.x → cu118
```

### Q3: 内存不足

```bash
# 减小批大小
python scripts/train_gpu.py --batch_size 16

# 或使用轻量模型
python scripts/train_lightweight.py
```

### Q4: 模块导入错误

```bash
# 确保在项目根目录
cd D:\BH\github\DroneVLA

# 确保激活了正确环境
conda activate dronevla

# 检查包是否安装
pip list | grep torch
```

---

## 5. IDE配置

### VS Code

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "C:/Users/用户名/miniconda3/envs/dronevla/python.exe",
    "python.terminal.activateEnvironment": true,
    "python.condaPath": "C:/Users/用户名/miniconda3/condabin/conda.bat"
}
```

### PyCharm

1. File → Settings → Project → Python Interpreter
2. 添加 Conda Interpreter
3. 选择 dronevla 环境

---

## 6. GPU监控

```bash
# 实时监控GPU使用
watch -n 1 nvidia-smi

# Windows PowerShell
while ($true) { nvidia-smi; Start-Sleep 1; Clear-Host }
```

```python
# Python中监控
import torch

def print_gpu_status():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU显存: {allocated:.1f}GB / {total:.1f}GB (已分配/总计)")
```

---

*下一节：[从零构建VLA](02_Build_VLA_From_Scratch.md)*
