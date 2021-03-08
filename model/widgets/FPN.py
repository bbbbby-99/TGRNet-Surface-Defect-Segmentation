import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, channel):
        super(FPN, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(channel, 2*channel, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))
        self.conv1 = nn.Sequential(
            nn.Conv2d(2*channel, 2*channel, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))
        self.conv2 = nn.Sequential(
            nn.Conv2d(4*channel, 2*channel, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))

    def forward(self, x0, x1, x2):
        # x0[b,2048,15,15] x1[b,1024,30,30] x2[b,512,60,60]
        _, _, H, W = x0.size()

        x0_conv = self.conv0(x0)
        x1_conv = self.conv1(x1)
        x2_conv = self.conv2(x2)

        x0_out = F.interpolate(x2_conv, size=(H,W), mode='bilinear', align_corners=True)
        x1_out = F.interpolate(x1_conv, size=(H,W), mode='bilinear', align_corners=True) + x0_out
        x2_out = F.interpolate(x0_conv, size=(H,W), mode='bilinear', align_corners=True) + x1_out

        return x0_out, x1_out, x2_out