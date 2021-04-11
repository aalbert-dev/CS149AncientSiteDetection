import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channels, s=1):
        super(ResBlock, self).__init__()
        out_channel_1, out_channel_2 = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel, out_channels=out_channel_1, kernel_size=1
            ),
            nn.BatchNorm2d(out_channel_1),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channel_1,
                out_channels=out_channel_2,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channel_2),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channel_2, out_channels=in_channel, kernel_size=1
            ),
            nn.BatchNorm2d(in_channel),
        )

        self.relu_out = nn.ReLU()

    def forward(self, x):
        x_orig = x

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x += x_orig

        x = self.relu_out(x)

        return x


class SimpleModel(nn.Module):
    # please modify the following code for training model
    def __init__(self, num_class=2):
        super(SimpleModel, self).__init__()

        self.num_class = num_class

        self.block1 = nn.Sequential(
            ResBlock(in_channel=1, out_channels=[4, 4], s=1),
            ResBlock(in_channel=1, out_channels=[4, 4], s=1),
        )

        self.fc = nn.Linear(129 * 129, self.num_class)

    def forward(self, x):
        x = self.block1(x)

        x = x.flatten(start_dim=1)

        x = self.fc(x)

        return x
