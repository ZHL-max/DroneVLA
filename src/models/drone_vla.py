"""
DroneVLA 无人机视觉-语言-动作模型

面向无人机的VLA模型架构：
- 视觉编码：处理机载相机图像
- 语言编码：理解自然语言指令
- 世界模型：预测未来状态
- 动作解码：生成飞行控制指令

作者：DroneVLA Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class VisualEncoder(nn.Module):
    """
    视觉编码器

    处理机载相机图像，提取视觉特征
    支持多帧时序融合
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        embed_dim: int = 256,
        temporal: bool = True,
        num_frames: int = 4
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.temporal = temporal
        self.num_frames = num_frames

        # 主干网络
        if backbone == "efficientnet_b0":
            import timm
            self.backbone = timm.create_model(
                "efficientnet_b0", pretrained=pretrained, num_classes=0
            )
            backbone_dim = 1280
        elif backbone == "resnet18":
            import torchvision.models as models
            resnet = models.resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            backbone_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # 特征投影
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        # 时序融合
        if temporal:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=8,
                batch_first=True
            )
            self.temporal_norm = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像序列

        Args:
            images: [B, T, C, H, W] 图像序列
        Returns:
            features: [B, embed_dim] 视觉特征
        """
        B, T, C, H, W = images.shape

        # 提取每帧特征
        images_flat = images.view(B * T, C, H, W)
        features_flat = self.backbone(images_flat)

        if features_flat.dim() > 2:
            features_flat = features_flat.mean(dim=[2, 3])

        features = features_flat.view(B, T, -1)

        # 特征投影
        features = self.projection(features)  # [B, T, embed_dim]

        # 时序融合
        if self.temporal:
            attended, _ = self.temporal_attention(
                features, features, features
            )
            features = self.temporal_norm(features + attended)

        # 全局池化
        features = features.mean(dim=1)  # [B, embed_dim]

        return features


class LanguageEncoder(nn.Module):
    """
    语言编码器

    使用预训练语言模型编码自然语言指令
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embed_dim: int = 256,
        freeze: bool = False
    ):
        super().__init__()

        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        # 特征投影
        self.projection = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

    def forward(self, instructions: list) -> torch.Tensor:
        """
        编码语言指令

        Args:
            instructions: List[str] 自然语言指令列表
        Returns:
            features: [B, embed_dim] 语言特征
        """
        # 分词
        inputs = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.model.device)

        # 编码
        outputs = self.model(**inputs)

        # 使用[CLS] token
        cls_features = outputs.last_hidden_state[:, 0]

        # 投影
        features = self.projection(cls_features)

        return features


class StateEncoder(nn.Module):
    """
    状态编码器

    编码无人机的低级状态信息：
    - 位置 (3D)
    - 速度 (3D)
    - 姿态 (3D)
    - 目标位置 (3D)
    """

    def __init__(self, state_dim: int = 12, embed_dim: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        编码状态

        Args:
            state: [B, state_dim] 状态向量
        Returns:
            features: [B, embed_dim] 状态特征
        """
        return self.encoder(state)


class WorldModel(nn.Module):
    """
    无人机世界模型

    预测给定动作下的未来状态
    用于：
    1. 策略训练（在想象中训练）
    2. 策略评估（预测行为后果）
    3. 规划（选择最优动作序列）
    """

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 4,
        hidden_dim: int = 256,
        prediction_horizon: int = 10
    ):
        super().__init__()

        self.prediction_horizon = prediction_horizon

        # 状态转移模型
        self.dynamics = nn.GRU(
            input_size=state_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # 状态解码器
        self.state_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )

        # 奖励预测器
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def predict_future(
        self,
        state: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测未来状态序列

        Args:
            state: [B, state_dim] 当前状态
            actions: [B, T, action_dim] 动作序列
        Returns:
            future_states: [B, T, state_dim] 预测的未来状态
            future_rewards: [B, T, 1] 预测的未来奖励
        """
        B = state.shape[0]
        T = actions.shape[1]

        # 准备输入序列
        state_expanded = state.unsqueeze(1).expand(-1, T, -1)
        inputs = torch.cat([state_expanded, actions], dim=-1)

        # GRM预测
        hidden_states, _ = self.dynamics(inputs)

        # 解码状态和奖励
        future_states = self.state_decoder(hidden_states)
        future_rewards = self.reward_predictor(hidden_states)

        return future_states, future_rewards

    def imagine_trajectory(
        self,
        state: torch.Tensor,
        policy: nn.Module,
        horizon: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        想象未来轨迹

        Args:
            state: [B, state_dim] 初始状态
            policy: 策略网络
            horizon: 想象步数
        Returns:
            trajectory: 包含状态、动作、奖励的字典
        """
        trajectory = {
            "states": [],
            "actions": [],
            "rewards": []
        }

        current_state = state

        for _ in range(horizon):
            # 使用策略选择动作
            action = policy(current_state)

            # 预测下一个状态
            next_state, reward = self.predict_future(
                current_state.unsqueeze(1),
                action.unsqueeze(1)
            )
            next_state = next_state.squeeze(1)
            reward = reward.squeeze(1)

            # 记录轨迹
            trajectory["states"].append(current_state)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)

            current_state = next_state

        return trajectory


class ActionDecoder(nn.Module):
    """
    动作解码器

    将多模态特征解码为无人机控制指令

    支持两种模式：
    1. 确定性：直接输出动作
    2. 扩散：通过去噪生成动作序列
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 4,
        action_horizon: int = 8,
        mode: str = "deterministic"
    ):
        super().__init__()

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.mode = mode

        if mode == "deterministic":
            # 确定性动作解码
            self.decoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        elif mode == "diffusion":
            # 扩散动作解码
            self.diffusion_decoder = DiffusionActionDecoder(
                input_dim=input_dim,
                action_dim=action_dim,
                action_horizon=action_horizon
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(
        self,
        features: torch.Tensor,
        num_diffusion_steps: int = 20
    ) -> torch.Tensor:
        """
        解码动作

        Args:
            features: [B, input_dim] 多模态融合特征
            num_diffusion_steps: 扩散步数（仅扩散模式）
        Returns:
            action: [B, action_dim] 或 [B, T, action_dim] 动作
        """
        if self.mode == "deterministic":
            action = self.decoder(features)
            # 缩放到实际范围
            action = action * torch.tensor([2.0, 2.0, 1.0, 1.0]).to(action.device)
            return action
        else:
            return self.diffusion_decoder.generate(features, num_diffusion_steps)


class DiffusionActionDecoder(nn.Module):
    """
    扩散动作解码器

    通过去噪过程生成动作序列
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 4,
        action_horizon: int = 8,
        hidden_dim: int = 256,
        num_diffusion_steps: int = 50
    ):
        super().__init__()

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_diffusion_steps = num_diffusion_steps

        # 噪声调度
        self.betas = torch.linspace(0.0001, 0.02, num_diffusion_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        # 条件编码
        self.condition_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )

        # 去噪网络
        self.denoiser = nn.Sequential(
            nn.Linear(action_horizon * action_dim + 128 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_horizon * action_dim)
        )

    def add_noise(self, x0, noise, t):
        """前向扩散"""
        sqrt_alpha = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha = torch.sqrt(1 - self.alpha_cumprod[t]).view(-1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def generate(self, condition, num_steps=20):
        """生成动作序列"""
        B = condition.shape[0]
        device = condition.device

        # 编码条件
        cond_embed = self.condition_encoder(condition)

        # 从噪声开始
        x = torch.randn(B, self.action_horizon, self.action_dim, device=device)

        # 逐步去噪
        step_size = self.num_diffusion_steps // num_steps

        for t in reversed(range(0, self.num_diffusion_steps, step_size)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            t_embed = self.time_embed(t_batch.float().unsqueeze(-1))

            # 预测噪声
            x_flat = x.reshape(B, -1)
            combined = torch.cat([x_flat, cond_embed, t_embed], dim=-1)
            noise_pred = self.denoiser(combined)
            noise_pred = noise_pred.reshape(B, self.action_horizon, self.action_dim)

            # 去噪步骤
            beta = self.betas[t]
            alpha = self.alphas[t]
            alpha_cumprod = self.alpha_cumprod[t]

            x0_pred = (x - torch.sqrt(1 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha_cumprod)
            x0_pred = torch.clamp(x0_pred, -1, 1)

            if t > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha) * x0_pred + torch.sqrt(beta) * noise
            else:
                x = x0_pred

        # 缩放到实际范围
        x = x * torch.tensor([2.0, 2.0, 1.0, 1.0], device=device)

        return x


class DroneVLA(nn.Module):
    """
    DroneVLA主模型

    整合视觉、语言、状态编码和动作解码
    支持世界模型辅助训练
    """

    def __init__(
        self,
        visual_dim: int = 256,
        language_dim: int = 256,
        state_dim: int = 12,
        state_embed_dim: int = 128,
        action_dim: int = 4,
        action_horizon: int = 8,
        use_world_model: bool = True,
        action_mode: str = "deterministic"
    ):
        super().__init__()

        self.use_world_model = use_world_model

        # 编码器
        self.visual_encoder = VisualEncoder(embed_dim=visual_dim)
        self.language_encoder = LanguageEncoder(embed_dim=language_dim)
        self.state_encoder = StateEncoder(
            state_dim=state_dim,
            embed_dim=state_embed_dim
        )

        # 多模态融合
        fusion_dim = visual_dim + language_dim + state_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        # 动作解码器
        self.action_decoder = ActionDecoder(
            input_dim=256,
            action_dim=action_dim,
            action_horizon=action_horizon,
            mode=action_mode
        )

        # 世界模型
        if use_world_model:
            self.world_model = WorldModel(
                state_dim=state_dim,
                action_dim=action_dim
            )

    def forward(
        self,
        images: torch.Tensor,
        instructions: list,
        state: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            images: [B, T, C, H, W] 图像序列
            instructions: List[str] 语言指令
            state: [B, state_dim] 无人机状态
            actions: [B, T, action_dim] 真实动作（训练时）
        Returns:
            outputs: 包含动作预测和世界模型预测的字典
        """
        # 编码各模态
        visual_features = self.visual_encoder(images)
        language_features = self.language_encoder(instructions)
        state_features = self.state_encoder(state)

        # 多模态融合
        combined = torch.cat([
            visual_features,
            language_features,
            state_features
        ], dim=-1)
        fused = self.fusion(combined)

        # 动作解码
        predicted_actions = self.action_decoder(fused)

        outputs = {
            "actions": predicted_actions,
            "visual_features": visual_features,
            "language_features": language_features,
            "state_features": state_features
        }

        # 世界模型预测
        if self.use_world_model and actions is not None:
            future_states, future_rewards = self.world_model.predict_future(
                state, actions
            )
            outputs["future_states"] = future_states
            outputs["future_rewards"] = future_rewards

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失

        Args:
            outputs: 模型输出
            targets: 目标值
        Returns:
            losses: 各项损失的字典
        """
        losses = {}

        # 动作损失
        if "actions" in targets:
            action_loss = F.mse_loss(outputs["actions"], targets["actions"])
            losses["action_loss"] = action_loss

        # 世界模型损失
        if self.use_world_model and "future_states" in outputs:
            state_loss = F.mse_loss(
                outputs["future_states"],
                targets["future_states"]
            )
            reward_loss = F.mse_loss(
                outputs["future_rewards"],
                targets["future_rewards"]
            )
            losses["state_loss"] = state_loss
            losses["reward_loss"] = reward_loss

        # 总损失
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss

        return losses
