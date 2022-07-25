from numpy import identity
import torch
from torch.nn import (
    Linear,
    Conv2d,
    BatchNorm2d,
    ReLU,
    Softmax,
    Sequential,
    Flatten,
    Tanh,
)


class ResidualBlock(torch.nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv1 = Conv2d(chs, chs, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = BatchNorm2d(chs)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = Conv2d(chs, chs, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn(out)

        out += identity
        out = self.relu(out)

        return out


# fmt: off
class Policy_Value(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Sequential(
            #15x15
            Conv2d(in_channels=4, out_channels=128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),

            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        # self.policy_head = Sequential(
        #     Conv2d(in_channels=128, out_channels=2, kernel_size=1, stride=1),
        #     BatchNorm2d(2),
        #     ReLU(),
        #     Flatten(),
        #     Linear(15* 15 * 2, 15*15),
        #     Softmax(-1)
        # )
        self.policy_head = Sequential(
            Conv2d(in_channels=128, out_channels=4, kernel_size=1, stride=1),
            BatchNorm2d(4),
            ReLU(),
            Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1),
            Flatten(),
            Softmax(-1)
        )
        self.value_head = Sequential(
            Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1),
            BatchNorm2d(1),
            ReLU(),
            Flatten(),
            Linear(15*15, 128),
            ReLU(),
            Linear(128, 1),
            Tanh()
        )

    def forward(self, x):
        x = self.backbone(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v

    def infer(self, x):
        self.eval()
        # x = torch.tensor(x, dtype=torch.float32).to('cuda:0')
        x = torch.tensor(x, dtype=torch.float32)
        x.unsqueeze_(0)
        x = x.permute(0, 3, 1, 2)
        with torch.no_grad():
            p, v = self.forward(x)
            return p[0].to('cpu'), v[0].to('cpu')

# fmt: on
