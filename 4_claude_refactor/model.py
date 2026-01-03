"""ResNet-10 Policy-Value Network for AlphaZero."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import config


class ResidualBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class PolicyValueNet(nn.Module):
    """ResNet-based policy-value network for AlphaZero.

    Architecture:
        Input: (batch, 5, 15, 15)
        -> Conv2d(5, 128) + BN + ReLU
        -> 10x ResidualBlock(128)
        -> Policy Head: Conv2d(128, 32) + BN + ReLU + Linear(7200, 225) + Softmax
        -> Value Head: Conv2d(128, 32) + BN + ReLU + Linear(7200, 256) + ReLU + Linear(256, 1) + Tanh
    """

    def __init__(
        self,
        board_size: int = config.board_size,
        num_channels: int = config.num_channels,
        num_filters: int = config.num_filters,
        num_res_blocks: int = config.num_res_blocks,
    ):
        super().__init__()

        self.board_size = board_size
        self.num_positions = board_size * board_size

        # Initial conv block
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_channels, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # Policy head
        policy_filters = 32
        self.policy_conv = nn.Conv2d(num_filters, policy_filters, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_filters)
        self.policy_fc = nn.Linear(policy_filters * board_size * board_size, self.num_positions)

        # Value head
        value_filters = 32
        self.value_conv = nn.Conv2d(num_filters, value_filters, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_filters)
        self.value_fc1 = nn.Linear(value_filters * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            policy: Action probabilities (batch, 225)
            value: State value (batch, 1) in range [-1, 1]
        """
        # Shared backbone
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)  # Use reshape for non-contiguous tensors
        p = self.policy_fc(p)
        policy = F.softmax(p, dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)  # Use reshape for non-contiguous tensors
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value

    def infer(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """Single state inference for gameplay.

        Args:
            state: Board state (15, 15, 5) numpy array

        Returns:
            policy: Action probabilities (225,) numpy array
            value: State value scalar
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            x = torch.from_numpy(state).float()
            x = x.permute(2, 0, 1).unsqueeze(0)  # (1, 5, 15, 15)
            x = x.to(next(self.parameters()).device)

            policy, value = self.forward(x)

            return policy[0].cpu().numpy(), value[0, 0].cpu().item()

    def infer_batch(self, states: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Batch inference for training.

        Args:
            states: List of board states, each (15, 15, 5)

        Returns:
            policies: (batch, 225) numpy array
            values: (batch,) numpy array
        """
        self.eval()
        with torch.no_grad():
            # Stack and convert
            x = np.stack(states, axis=0)  # (batch, 15, 15, 5)
            x = torch.from_numpy(x).float()
            x = x.permute(0, 3, 1, 2)  # (batch, 5, 15, 15)
            x = x.to(next(self.parameters()).device)

            policies, values = self.forward(x)

            return policies.cpu().numpy(), values.squeeze(-1).cpu().numpy()


def create_model(device: str = config.device) -> PolicyValueNet:
    """Create and initialize model on specified device."""
    model = PolicyValueNet()
    model = model.to(device)
    return model


def load_model(path: str, device: str = config.device) -> PolicyValueNet:
    """Load model from checkpoint."""
    model = PolicyValueNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model
