import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, in_channels: int = 3, num_actions: int = 5):
        super().__init__()
        # Input: (C, 96, 96)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # compute feature size after convs for linear layer
        self._conv_output_size = self._get_conv_output((in_channels, 96, 96))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.zeros(bs, *shape)
        output_feat = self.features(input)
        n_size = output_feat.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        # x: (B, C, H, W), values in [0,1]
        feat = self.features(x)
        return self.head(feat)


def create_model(device, in_channels=3, num_actions=5):
    model = QNetwork(in_channels=in_channels, num_actions=num_actions).to(device)
    return model
