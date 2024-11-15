import torch.nn as nn


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SmallCNN(nn.Module):
    """A small CNN model inspired by ResNet"""

    def __init__(self, in_channels, num_classes, h_1=128, h_2=512, p_drop=0.2):
        super().__init__()

        self.conv1 = conv_block(in_channels, h_1 // 2)
        self.conv2 = conv_block(h_1 // 2, h_1, pool=True)  # 128
        self.res1 = nn.Sequential(
            conv_block(h_1, h_1), conv_block(h_1, h_1)
        )  # 128, 128 128, 128

        self.conv3 = conv_block(h_1, h_1 * 2, pool=True)  # 128, 256
        self.conv4 = conv_block(h_1 * 2, h_2, pool=True)  # 256, 512
        self.res2 = nn.Sequential(
            conv_block(h_2, h_2), conv_block(h_2, h_2)
        )  # 512, 512 512, 512

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(h_2, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x

        x = self.classifier(x)
        return x
