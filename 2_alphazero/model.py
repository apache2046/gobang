import torch
from torch.nn import Linear, Conv2d, BatchNorm2d, ReLU, Softmax, Sequential, Flatten

# fmt: off
class Policy_Value(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Sequential(
            #15x15
            Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),

            #15x15
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            BatchNorm2d(64),
            ReLU(),

            #13x13
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            BatchNorm2d(128),
            ReLU(),

            #11x11
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            BatchNorm2d(256),
            ReLU(),

            #9x9
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0),
            BatchNorm2d(512),
            ReLU(),

            #7x7
            Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0),

            #5x5
            Flatten()
        )
        self.policy_head = Sequential(
            Linear(25 * 1024, 1024),
            ReLU(),
            Linear(1024, 225),
            Softmax()
        )
        self.value_head = Sequential(
            Linear(25 * 1024, 1024),
            ReLU(),
            Linear(1024, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v
# fmt: on
