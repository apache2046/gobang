import torch
from torch.nn import Linear, Conv2d, BatchNorm2d, ReLU, Softmax, Sequential, Flatten

# fmt: off
class Policy_Value(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Sequential(
            #15x15
            Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1),
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
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            BatchNorm2d(256),
            ReLU(),

            #7x7
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),

            #5x5
            Flatten()
        )
        self.policy_head = Sequential(
            Linear(25 * 256, 1024),
            ReLU(),
            Linear(1024, 225),
            Softmax(-1)
        )
        self.value_head = Sequential(
            Linear(25 * 256, 1024),
            ReLU(),
            Linear(1024, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v

    def infer(self, x):
        self.eval()
        x = torch.tensor(x, dtype=torch.float32).to('cuda:0')
        x.unsqueeze_(0)
        x = x.permute(0, 3, 1, 2)
        with torch.no_grad():
            p, v = self.forward(x)
            return p[0].to('cpu'), v[0].to('cpu')

# fmt: on
