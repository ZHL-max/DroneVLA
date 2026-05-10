"""
DroneVLA 模型测试
"""

import torch
import numpy as np
from src.models.drone_vla import (
    DroneVLA,
    VisualEncoder,
    LanguageEncoder,
    StateEncoder,
    WorldModel,
    ActionDecoder
)


def test_visual_encoder():
    """测试视觉编码器"""
    print("测试视觉编码器...")

    encoder = VisualEncoder(embed_dim=256, temporal=True)

    # 创建测试数据
    images = torch.randn(2, 4, 3, 64, 64)  # [B, T, C, H, W]

    # 前向传播
    features = encoder(images)

    assert features.shape == (2, 256), f"期望形状 (2, 256)，得到 {features.shape}"
    print("  通过！")


def test_language_encoder():
    """测试语言编码器"""
    print("测试语言编码器...")

    encoder = LanguageEncoder(embed_dim=256)

    # 创建测试数据
    instructions = ["hover at position 1, 2, 3", "fly to the red building"]

    # 前向传播
    features = encoder(instructions)

    assert features.shape == (2, 256), f"期望形状 (2, 256)，得到 {features.shape}"
    print("  通过！")


def test_state_encoder():
    """测试状态编码器"""
    print("测试状态编码器...")

    encoder = StateEncoder(state_dim=12, embed_dim=128)

    # 创建测试数据
    state = torch.randn(2, 12)

    # 前向传播
    features = encoder(state)

    assert features.shape == (2, 128), f"期望形状 (2, 128)，得到 {features.shape}"
    print("  通过！")


def test_world_model():
    """测试世界模型"""
    print("测试世界模型...")

    model = WorldModel(state_dim=12, action_dim=4)

    # 创建测试数据
    state = torch.randn(2, 12)
    actions = torch.randn(2, 10, 4)  # 10步动作序列

    # 预测未来
    future_states, future_rewards = model.predict_future(state, actions)

    assert future_states.shape == (2, 10, 12), f"期望形状 (2, 10, 12)，得到 {future_states.shape}"
    assert future_rewards.shape == (2, 10, 1), f"期望形状 (2, 10, 1)，得到 {future_rewards.shape}"
    print("  通过！")


def test_action_decoder():
    """测试动作解码器"""
    print("测试动作解码器...")

    # 测试确定性模式
    decoder = ActionDecoder(input_dim=256, action_dim=4, mode="deterministic")
    features = torch.randn(2, 256)
    action = decoder(features)

    assert action.shape == (2, 4), f"期望形状 (2, 4)，得到 {action.shape}"
    print("  确定性模式通过！")

    # 测试扩散模式
    decoder = ActionDecoder(input_dim=256, action_dim=4, action_horizon=8, mode="diffusion")
    action = decoder(features, num_diffusion_steps=10)

    assert action.shape == (2, 8, 4), f"期望形状 (2, 8, 4)，得到 {action.shape}"
    print("  扩散模式通过！")


def test_drone_vla():
    """测试完整的DroneVLA模型"""
    print("测试DroneVLA模型...")

    model = DroneVLA(
        visual_dim=256,
        language_dim=256,
        state_dim=12,
        state_embed_dim=128,
        action_dim=4,
        action_horizon=8,
        use_world_model=True,
        action_mode="deterministic"
    )

    # 创建测试数据
    images = torch.randn(2, 4, 3, 64, 64)
    instructions = ["hover at position 1, 2, 3", "fly to the red building"]
    state = torch.randn(2, 12)
    actions = torch.randn(2, 8, 4)

    # 前向传播（带世界模型）
    outputs = model(images, instructions, state, actions)

    assert "actions" in outputs
    assert "visual_features" in outputs
    assert "language_features" in outputs
    assert "state_features" in outputs
    assert "future_states" in outputs
    assert "future_rewards" in outputs

    print("  通过！")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("DroneVLA 模型测试")
    print("=" * 60)

    test_visual_encoder()
    test_language_encoder()
    test_state_encoder()
    test_world_model()
    test_action_decoder()
    test_drone_vla()

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
