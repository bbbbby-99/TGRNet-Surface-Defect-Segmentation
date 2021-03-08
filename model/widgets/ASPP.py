import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, channel=256):
        super(ASPP, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=6, dilation=6, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )

    def forward(self, x):

        feature_size = x.shape[-2:]
        global_feature = F.avg_pool2d(x, kernel_size=feature_size)

        global_feature = self.layer0(global_feature)

        global_feature = global_feature.expand(-1, -1, feature_size[0], feature_size[1])
        out = torch.cat([global_feature, self.layer1(x), self.layer2(x), self.layer3(x),self.layer4(x)], dim = 1)

        out = self.layer5(out)

        return out