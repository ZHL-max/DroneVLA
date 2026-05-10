"""
DroneVLA 模型导出脚本

将PyTorch模型导出为ONNX格式，用于TensorRT优化和部署

使用方法：
    python scripts/export_onnx.py --model logs/best_model.pt --output logs/model.onnx
"""

import torch
import torch.onnx
import os
import argparse
import json
import sys

sys.path.insert(0, '.')


def export_to_onnx(model_path, output_path, opset_version=11):
    """
    将PyTorch模型导出为ONNX格式

    Args:
        model_path: PyTorch模型路径
        output_path: ONNX输出路径
        opset_version: ONNX opset版本
    """
    from src.models.drone_vla import DroneVLA

    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # 加载配置
    config_path = os.path.join(os.path.dirname(model_path), 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            'visual_dim': 256, 'language_dim': 256, 'state_dim': 12,
            'state_embed_dim': 128, 'action_dim': 4, 'action_horizon': 8,
            'use_world_model': False, 'action_mode': 'deterministic'
        }

    # 创建模型
    model = DroneVLA(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 创建示例输入
    batch_size = 1
    num_frames = 4
    image_size = 64

    dummy_images = torch.randn(batch_size, num_frames, 3, image_size, image_size)
    dummy_state = torch.randn(batch_size, 12)
    dummy_instructions = ["hover at current position"]

    # 导出模型
    print(f"导出模型到 {output_path}...")

    # 由于模型包含字符串输入，我们需要创建一个包装器
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images, state):
            # 简化：使用固定的指令
            instruction = ["hover at current position"]
            outputs = self.model(images, instruction, state)
            return outputs['actions']

    wrapper = ONNXWrapper(model)

    # 导出
    torch.onnx.export(
        wrapper,
        (dummy_images, dummy_state),
        output_path,
        opset_version=opset_version,
        input_names=['images', 'state'],
        output_names=['actions'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'state': {0: 'batch_size'},
            'actions': {0: 'batch_size'}
        }
    )

    print(f"ONNX模型已保存到: {output_path}")
    print(f"模型大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    # 验证ONNX模型
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX模型验证通过")
    except ImportError:
        print("注意：未安装onnx库，跳过验证")
    except Exception as e:
        print(f"ONNX验证失败: {e}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="DroneVLA ONNX导出")
    parser.add_argument("--model", type=str, required=True, help="PyTorch模型路径")
    parser.add_argument("--output", type=str, default=None, help="ONNX输出路径")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset版本")

    args = parser.parse_args()

    if args.output is None:
        args.output = args.model.replace('.pt', '.onnx')

    export_to_onnx(args.model, args.output, args.opset)


if __name__ == "__main__":
    main()
